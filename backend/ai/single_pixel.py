import os
import time
import json
from datetime import datetime

import torch
import torch.profiler as profiler
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import save_image

from torch.utils.data import DataLoader, TensorDataset
from piq import ssim, SSIMLoss, psnr
from torch.utils.tensorboard import SummaryWriter

from decoders import VFXSpiralNetDecoder, SpecificDecoder, BigDecoder, CoordFlowish, TestDecoder, TLearnedWaveVectorDecoder, TModulatedDecoder, DualDecoder
from encoding_utils import sample_fourier_transforms
from losses import DCTLoss, GradientLoss
from make_shader import decoder_to_glsl, compare_decoder_and_shader, save_weights_to_exr
from image_utils import load_images, save_images
from soap import SOAP


LATENT_IMAGE_CHANNELS = 4 # RGBA


class VFXNet(nn.Module):
    def __init__(self, height, width, device, decoder_type, decoder_config, experiment_name=None, freeze_decoder=False):
        super().__init__()
        self.device = device
        self.experiment_name = experiment_name
        self.height = height
        self.width = width
        decoder_config = decoder_config or {}
        freq_init = decoder_config.pop("freq_init", None)
        low_freqs = decoder_config.pop("low_freq_init", None)
        high_freqs = decoder_config.pop("high_freq_init", None)
        if decoder_type == "SpiralNet":
            self.decoder = VFXSpiralNetDecoder(device, **decoder_config)
        elif decoder_type == "Specific":
            self.decoder = SpecificDecoder(device, **decoder_config)
        elif decoder_type == "Big":
            self.decoder = BigDecoder(device, **decoder_config)
        elif decoder_type == "CoordFlowish":
            self.decoder = CoordFlowish(device, **decoder_config)
        elif decoder_type == "Test":
            self.decoder = TestDecoder(device, freq_init, **decoder_config)
        elif decoder_type == "TLearnedWaveVector":
            self.decoder = TLearnedWaveVectorDecoder(device, **decoder_config)
        elif decoder_type == "TModulated":
            self.decoder = TModulatedDecoder(device, freq_init, **decoder_config)
        elif decoder_type == "Dual":
            self.decoder = DualDecoder(device, low_freqs, high_freqs, **decoder_config)
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")
        if getattr(self.decoder, "latent_dim", 0) > 0:
            self.latent = nn.Embedding(self.height * self.width, LATENT_IMAGE_CHANNELS)
        self.output_channels = self.decoder.output_channels
        self.freeze_decoder = freeze_decoder
        if self.freeze_decoder:
            for param in self.decoder.parameters():
                param.requires_grad = False
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


    def forward(self, raw_pos, control):
        if getattr(self.decoder, "latent_dim", 0) > 0:
            return self.decoder(raw_pos, control[:, 0:1], self.latent)
        return self.decoder(raw_pos, control[:, 0:1])
    
    def run_patch(self, pos_patch, time_patch):
        flat_raw_pos = pos_patch.view(-1, pos_patch.shape[-1])
        flat_control = time_patch.view(-1, time_patch.shape[-1])

        return self.decoder(flat_raw_pos, flat_control).view(*pos_patch.shape[:-1], self.output_channels)

    def full_image(self, time, H=512, W=512):
        safe_batch_size = 512 * 512
        x_coords = torch.linspace(-1, 1, W, device=self.device)
        y_coords = torch.linspace(-1, 1, H, device=self.device)
        grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
        raw_pos = torch.stack([grid_x, grid_y], dim=-1)
        expanded_time = time.repeat(H, W, 1)

        flat_pos = raw_pos.view(-1, 2).to(device=self.device)
        flat_time = expanded_time.view(-1, 1).to(device=self.device)
        if flat_pos.shape[0] > safe_batch_size:
            chunks = []
            for i in range(0, flat_pos.shape[0], safe_batch_size):
                chunk_pos = flat_pos[i:i+safe_batch_size]
                chunk_time = flat_time[i:i+safe_batch_size]
                chunk_out = self.decoder(chunk_pos, chunk_time)
                chunks.append(chunk_out)
            response = torch.cat(chunks, dim=0)
        else:
            response = self.decoder(flat_pos, flat_time)
        shaped_image = response.view(H, W, self.output_channels)
        return shaped_image


def compute_ssim_scores(
    reconstructed_batch,
    sampled_controls,
    image_tensor,
    shape,
    control_tensor, 
    update=False,
    ssim_loss_fn=None,
    optimizer=None
):
    H, W = shape
    pixels_per_frame = H * W
    T = control_tensor.shape[0] // pixels_per_frame
    ssim_scores = []

    if not update:
        torch.set_grad_enabled(False)  # Ensure gradients off for pure evaluation
    else:
        assert ssim_loss_fn is not None, "Must provide SSIM loss function when update=True"
        torch.set_grad_enabled(True)

    for recon, control in zip(reconstructed_batch, sampled_controls):
        t = control[0].item()
        frame_idx = min(int(t * (T - 1)), T - 1)

        start = frame_idx * pixels_per_frame
        end = start + pixels_per_frame
        gt_frame = image_tensor[start:end].view(H, W, -1)

        recon_img = recon.permute(2, 0, 1).unsqueeze(0).detach()
        gt_img = gt_frame.permute(2, 0, 1).unsqueeze(0).detach()

        if update:
            recon_img.requires_grad_(True)
            gt_img.requires_grad_(True)
            loss = ssim_loss_fn(recon_img, gt_img)
            print(f"SSIM Loss: {loss.item()}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ssim_scores.append(loss.item())
        else:
            score = 1 - ssim_loss_fn(recon_img, gt_img).item()
            ssim_scores.append(score)

    torch.set_grad_enabled(True)  # reset default
    return ssim_scores


def compute_gradient_scores(
    reconstructed_batch,
    sampled_controls,
    image_tensor,
    shape,
    control_tensor,
    update=False,
    grad_loss_fn=None,
    optimizer=None
):
    """
    Same API as compute_ssim_scores but runs your gradient loss.
    Returns a list of per-frame grad losses; if update=True, does one
    backward()/step() on the mean grad loss.
    """
    H, W = shape
    pixels = H * W
    T = control_tensor.shape[0] // pixels

    if update:
        assert grad_loss_fn is not None, "Need grad_loss_fn when update=True"
        assert optimizer     is not None, "Need optimizer when update=True"
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)

    total_grad = 0.0
    grad_scores = []

    for recon, control in zip(reconstructed_batch, sampled_controls):
        # pick matching GT frame
        t = control[0].item()
        idx = min(int(t * (T - 1)), T - 1)
        start = idx * pixels
        gt = image_tensor[start:start+pixels].view(H, W, -1)

        # NCHW tensors
        p = recon.permute(2, 0, 1).unsqueeze(0)
        g = gt.permute(2, 0, 1).unsqueeze(0).detach()

        loss_g = grad_loss_fn(p, g)
        grad_scores.append(loss_g.item())
        if update:
            total_grad += loss_g

    if update:
        mean_grad = total_grad / len(reconstructed_batch)
        optimizer.zero_grad()
        mean_grad.backward()
        optimizer.step()

    torch.set_grad_enabled(True)
    return grad_scores


def subsample_random_pixels(num_samples, image_tensor, raw_pos, control_tensor):
    total_pixels = image_tensor.shape[0]
    indices = torch.randperm(total_pixels)[:num_samples]
    return (
        image_tensor[indices],
        raw_pos[indices],
        control_tensor[indices],
    )


class PatchSampler(torch.utils.data.IterableDataset):
    def __init__(self, image_tensor, device, tile_size, batch_size, threshold_psnr=35, dtype=torch.float32):
        self.device = device
        self.tile_size = tile_size
        self.margin_size = self.tile_size // 2
        self.image_tensor = image_tensor.permute(0, 3, 1, 2)  # Change to NCHW format
        self.T, self.C, self.H, self.W = self.image_tensor.shape
        self.current_T = 0
        self.threshold_psnr = threshold_psnr

        dx = self.margin_size / self.W
        dy = self.margin_size / self.H
        self.kernel_x = torch.linspace(-dx, dx, tile_size, dtype=image_tensor.dtype, device=device)
        self.kernel_y = torch.linspace(-dy, dy, tile_size, dtype=image_tensor.dtype, device=device)

        self.patch_kernel = torch.stack(torch.meshgrid(self.kernel_y, self.kernel_x, indexing='ij'), dim=-1)  # (tile_size, tile_size, 2)
        self.batch_size = batch_size
        self.dtype = dtype

    def __iter__(self):
        while True:
            times = torch.randint(0, self.current_T + 1, (self.batch_size,), device=self.image_tensor.device)

            image_batch = self.image_tensor[times].to(device=self.device)  # Move to GPU
            patch_centers = torch.rand((self.batch_size, 2), dtype=self.image_tensor.dtype, device=self.device) * 2 - 1
            patch_grid = self.patch_kernel.unsqueeze(0) + patch_centers[:, None, None, :]  # [B, tile, tile, 2]
            
            patch_batch = F.grid_sample(image_batch, patch_grid).requires_grad_(True)  # [B, C, tile, tile]
            # Keep patch_grid in [-1, 1] range to match Fourier frequency assumptions
            patch_grid = patch_grid.clamp(-1.0, 1.0)

            times_patch = times.float().view(-1, 1, 1, 1).expand(-1, self.tile_size, self.tile_size, 1)
            yield (
                patch_batch.to(dtype=self.dtype, device=self.device),
                patch_grid.to(dtype=self.dtype, device=self.device),
                (times_patch / self.T).to(dtype=self.dtype, device=self.device)
            )
    
    def update_time(self):
        if self.current_T < self.T - 1:
            self.current_T += 1
        print(f"PatchSampler current_T updated to {self.current_T}")


class PSNRLoss(nn.Module):
    def __init__(self, data_range=1.0, eps=1e-8):
        super().__init__()
        self.data_range = data_range
        self.eps = eps

    def forward(self, pred, target):
        mse = torch.mean((pred - target) ** 2)
        psnr = 10 * torch.log10((self.data_range ** 2) / (mse + self.eps))
        # negate so lower is worse, higher is better (like SSIMLoss)
        return -psnr


def train_vfx_model(
    image_dir,
    device,
    batch_size,
    patch_size,
    epochs,
    batches_per_epoch,
    experiment_name=None,
    decoder_type="SpiralNet",
    decoder_config=None
):
    image_tensor = load_images(image_dir)
    T, H, W, C = image_tensor.shape

    psnr_loss = PSNRLoss(data_range=1.0).to(device)

    freq_init, _ = sample_fourier_transforms(image_tensor, decoder_config.get("pos_encoding_len", 512), device)
    # sort and split frequencies for Dual decoder by magnitude
    if decoder_type == "Dual":
        # Sort by magnitude (L2 norm) instead of raw values
        freq_magnitudes = torch.linalg.norm(freq_init, dim=1)
        sorted_indices = torch.argsort(freq_magnitudes)
        freq_init_sorted = freq_init[sorted_indices]
        
        mid = freq_init_sorted.shape[0] // 2
        low_freq_init = freq_init_sorted[:mid]  # Lower magnitude frequencies
        high_freq_init = freq_init_sorted[mid:]  # Higher magnitude frequencies
        
        print(f"low freq magnitudes: {torch.linalg.norm(low_freq_init, dim=1).min().item():.4f} to {torch.linalg.norm(low_freq_init, dim=1).max().item():.4f}")
        print(f"high freq magnitudes: {torch.linalg.norm(high_freq_init, dim=1).min().item():.4f} to {torch.linalg.norm(high_freq_init, dim=1).max().item():.4f}")
        decoder_config["low_freq_init"] = low_freq_init
        decoder_config["high_freq_init"] = high_freq_init
    else:
        decoder_config["freq_init"] = freq_init
    model = VFXNet(H, W, device, decoder_type=decoder_type, decoder_config=decoder_config)
    model.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    optimizer = SOAP(
        model.parameters(),
        weight_decay=0,
    )

    base_path = f"anim_tests/{model.experiment_name}"
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    dataset = PatchSampler(image_tensor, device, tile_size=patch_size, batch_size=batch_size)
    dataloader = DataLoader(dataset, batch_size=None)
    # Logging epochs: more frequent at start, less frequent later
    log_epochs = (
        list(range(0, 11, 1))   # 0-10 every epoch
        + list(range(10, 51, 5))   # 10-50 every 5 epochs
        + list(range(50, 101, 10)) # 50-100 every 10 epochs
        + list(range(100, 501, 50)) # 100-500 every 50 epochs
        + list(range(500, 1001, 100)) # 500-1000 every 100 epochs
    )
    if epochs > 1000:
        log_epochs += list(range(1000, epochs, 250))

    writer = SummaryWriter(f'runs/{model.experiment_name}')
    writer.add_text('Config/Decoder', json.dumps(decoder_config or {}, indent=2), 0)
    writer.add_graph(
        model,
        (
            torch.zeros((7, 2), device=device),
            torch.zeros((7, 1), device=device),
        )
    )
    
    epoch_numbers = torch.arange(epochs)
    first_batch_sizes = torch.linspace(100, 1000, steps=250)
    remainder_sizes = torch.full((750,), 1000)
    full_batch_sizes = torch.cat([first_batch_sizes, remainder_sizes]).long()

    total_patch_loss = 0.0
    global_step = 0
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for epoch in range(epochs):
        batches_per_epoch = full_batch_sizes[epoch].item()
        model.train()
        batch_num = 0
        total_pixel_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for image_patches, pos_patches, time_patches in dataloader:
            # Permute reconstructed_image from [B, H, W, C] to [B, C, H, W] and select first 3 channels
            reconstructed_image = model.run_patch(pos_patches, time_patches)[..., :3]
            reconstructed_image_patches = reconstructed_image.permute(0, 3, 1, 2)
            psnr_component = psnr_loss(reconstructed_image_patches, image_patches)

            if batch_num % 100 == 0:
                print(f"Batch {batch_num+1}/{batches_per_epoch}")

            psnr_component.backward()
            total_patch_loss += psnr_component.item()
            batch_num += 1
            if batch_num >= batches_per_epoch:
                break
        optimizer.step()
        total_patch_loss /= batches_per_epoch
        if -total_patch_loss > dataset.threshold_psnr:
            dataset.update_time()

        if epoch in log_epochs:
            print(f"Experiment {model.experiment_name}, Epoch {epoch}, Patch PSNR Loss: {total_patch_loss:.6f}")
            epoch_dir = f"{base_path}/epoch_{epoch}"
            os.makedirs(epoch_dir, exist_ok=True)
            model.eval()  # Set model to evaluation mode
            with torch.inference_mode():
                debug_frames = 5
                test_frames = 10
                save_images(model, H=256, W=256, n_images=debug_frames, gif_frames=T, base_dir=epoch_dir)

                test_controls = torch.randint(0, dataset.current_T + 1, (test_frames,), device=image_tensor.device)
                base_batch = image_tensor[test_controls]

                print("Computing test batch...")
                pred_batch = [model.full_image(test_control / T, H, W) for test_control in test_controls]
                for i, full_res_image in enumerate(pred_batch):
                    rgb_image = full_res_image.squeeze(0).permute(2, 0, 1)  # [C, H, W]
                    rgb_path = os.path.join(epoch_dir, f"full_res_rgb_{i}.png")
                    save_image(rgb_image, rgb_path)

                pred_batch = torch.stack(pred_batch).to(dtype=torch.float32)
                print(f"Base batch shape: {base_batch.shape}, Pred batch shape: {pred_batch.shape}")

                base_batch_nchw = base_batch.permute(0, 3, 1, 2)
                pred_batch_nchw = pred_batch.permute(0, 3, 1, 2).to(device=base_batch_nchw.device)
                comparison = torch.cat([base_batch_nchw, pred_batch_nchw], dim=3)  # Concatenate along width

                ssim_result = ssim(base_batch_nchw, pred_batch_nchw)
                psnr_result = psnr(base_batch_nchw, pred_batch_nchw)
                writer.add_scalar('Eval/SSIM', ssim_result.item(), epoch)
                writer.add_scalar('Eval/PSNR', psnr_result.item(), epoch)

                for i, t in enumerate(test_controls):
                    writer.add_image(f'Images/Comparison_t{t}', comparison[i], epoch)
                print(f"Num trainable params: {num_params}")
                print(f"SSIM result: {ssim_result.item():.4f}, PSNR result: {psnr_result.item():.4f}")
        global_step += 1


def get_total_grad_norm(model, norm_type=2):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm

