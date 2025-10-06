import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v3 as iio
import numpy as np
from torchvision.utils import save_image
from torchvision.transforms import GaussianBlur
from torch.profiler import profile, record_function, ProfilerActivity

from encoding_utils import (
    sample_fourier_transforms,
    compute_helmholtz_encoding,
    compute_rbf_encoding,
    residual_after_modes,
    simple_energy_map,
    weighted_random_centers,
    compute_dog,
    residual_from_gaussians,
    save_gaussians_as_image
)
from single_pixel import VFXNet, PSNRLoss
from soap import SOAP


def load_image(png_path, device, dtype=torch.float32):
    png_frame = iio.imread(png_path, plugin="pillow")
    frame = np.array(png_frame) / 255.0  # Normalize to [0, 1]
    return torch.tensor(frame, dtype=dtype, device=device).unsqueeze(0)


class RBFLayer(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(d_out, d_in))
        log_sigmas = torch.zeros(d_out)
        self.sigmas = nn.Parameter(torch.exp(log_sigmas) + 1e-6)

    def forward(self, x):
        x2 = (x ** 2).sum(dim=-1, keepdim=True)            # [B, 1]
        c2 = (self.centers ** 2).sum(dim=-1).unsqueeze(0)  # [1, d_out]
        cross = x @ self.centers.T                         # [B, d_out]
        dist_sq = x2 + c2 - 2 * cross                      # [B, d_out]
        dist_sq = dist_sq.clamp_min(0)                     # avoid tiny negatives
        rbf = torch.exp(-dist_sq / (2 * self.sigmas**2))   # [B, d_out]
        return rbf

class HybridHalfSine(nn.Module):
    """
    nn.Linear(d_in, d_out) with half ReLU, half sine activations.
    Same parameter count as Linear.
    """
    def __init__(self, d_in, d_out, omega=30.0, first_layer=False, rotate=False):
        super().__init__()
        self.lin = nn.Linear(d_in, d_out)
        self.omega = omega
        # self.omega = nn.Parameter(torch.tensor(omega))
        self.split = d_out // 2  # first half ReLU, second half sine
        self.rotate = rotate

        # init: standard for ReLU rows, scaled for sine rows
        nn.init.kaiming_uniform_(self.lin.weight, a=0.0)
        nn.init.zeros_(self.lin.bias)
        if first_layer:
            bound = 1.0 / d_in
        else:
            bound = (6.0 / d_in) ** 0.5 / self.omega
        with torch.no_grad():
            self.lin.weight[self.split:].uniform_(-bound, bound)
            self.lin.bias[self.split:].fill_(0.0)

    def forward(self, x):
        z = self.lin(x)
        a = F.relu(z[..., :self.split], inplace=False)
        b = torch.sin(self.omega * z[..., self.split:])
        if self.rotate:
            return torch.cat([b, a], dim=-1)
        else:
            return torch.cat([a, b], dim=-1)


class BasicRecon(nn.Module):
    def __init__(
        self,
        device,
        low_init_freqs,
        # coarse_centers,
        # coarse_sigmas,
        high_init_freqs,
        # fine_centers,
        # fine_sigmas,
        num_harmonics=256,
        hidden_dim=256,
        output_channels=3
    ):
        super().__init__()
        print("Setting default device", device)
        torch.set_default_device(device)
        self.device = device
        self.output_channels = output_channels
        self.pos_encoding_len = num_harmonics
        self.hidden_dim = hidden_dim

        self.linear_enc1 = nn.Linear(2, num_harmonics)
        self.linear_enc2 = nn.Linear(2, num_harmonics)

        layers1 = [
            nn.Linear(num_harmonics * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_channels + 1),
        ]


        layers2 = [
            nn.Linear(num_harmonics * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_channels + 1),
        ]

        self.decoder1 = nn.Sequential(*layers1)
        self.decoder2 = nn.Sequential(*layers2)

        self.low_wavevectors = low_init_freqs
        # self.coarse_centers = coarse_centers
        # self.coarse_sigmas = coarse_sigmas

        self.high_wavevectors = high_init_freqs
        # self.fine_centers = fine_centers
        # self.fine_sigmas = fine_sigmas
        # self.rbf_sigmas = F.softplus(torch.full((num_harmonics,), torch.log(torch.exp(torch.tensor(0.25))-1))) + 0.01
        self.low_wavevectors.detach()
        # self.coarse_centers.detach()
        # self.coarse_sigmas.detach()
        self.high_wavevectors.detach()
        # self.fine_centers.detach()
        # self.fine_sigmas.detach()

    def forward(self, raw_pos):
        # branch_weight = self.switcher(raw_pos).squeeze(-1)

        low_encoded_linear = self.linear_enc1(raw_pos)
        low_encoded_hh = compute_helmholtz_encoding(
            raw_pos,
            self.pos_encoding_len,
            self.low_wavevectors,
        ).detach()
        # coarse_rbfs = compute_rbf_encoding(
        #     raw_pos,
        #     self.coarse_centers,
        #     self.coarse_sigmas
        # ).detach()

        high_encoded_linear = self.linear_enc2(raw_pos)
        high_encoded_hh = compute_helmholtz_encoding(
            raw_pos,
            self.pos_encoding_len,
            self.high_wavevectors,
        ).detach()
        # fine_rbfs = compute_rbf_encoding(
        #     raw_pos,
        #     self.fine_centers,
        #     self.fine_sigmas
        # ).detach()

        encoded_pos1 = torch.cat([low_encoded_linear, low_encoded_hh], dim=-1)
        encoded_pos2 = torch.cat([high_encoded_linear, high_encoded_hh], dim=-1)

        output1 = encoded_pos1
        output2 = encoded_pos2

        for layer in self.decoder1:
            output1 = layer(output1)

        for layer in self.decoder2:
            output2 = layer(output2)

        rgb1, weight1 = output1[..., :self.output_channels], output1[..., -1:]
        rgb2, weight2 = output2[..., :self.output_channels], output2[..., -1:]
        weights = torch.softmax(torch.cat([weight1, weight2], dim=-1), dim=-1)
        w1, w2 = weights[..., 0:1], weights[..., 1:2]
        output = rgb1 * w1 + rgb2 * w2

        return output

    def full_image_inf(self, H=512, W=512):
        with torch.inference_mode():
            safe_batch_size = 512*512
            x_coords = torch.linspace(-1, 1, W, device=self.device)
            y_coords = torch.linspace(-1, 1, H, device=self.device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            raw_pos = torch.stack([grid_x, grid_y], dim=-1)
            flat_pos = raw_pos.view(-1, 2)
            chunks = []
            for i in range(0, flat_pos.shape[0], safe_batch_size):
                chunk_pos = flat_pos[i:i+safe_batch_size]
                chunks.append(self.forward(chunk_pos))
            reconstructed = torch.cat(chunks, dim=0)
            return reconstructed.view(H, W, 3)

    def visualize_switching(self, H=512, W=512):
        with torch.inference_mode():
            x_coords = torch.linspace(-1, 1, W, device=self.device)
            y_coords = torch.linspace(-1, 1, H, device=self.device)
            grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
            raw_pos = torch.stack([grid_x, grid_y], dim=-1)
            flat_pos = raw_pos.view(-1, 2)

            branch_weight = self.switcher(flat_pos).squeeze(-1)
            return branch_weight.view(H, W)


def full_image_train(image, model, opt, loss, H=512, W=512, scaler=None):
    x_coords = torch.linspace(-1, 1, W, device=model.device)
    y_coords = torch.linspace(-1, 1, H, device=model.device)
    grid_y, grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')
    raw_pos = torch.stack([grid_x, grid_y], dim=-1)
    flat_pos = raw_pos.view(-1, 2)
    N = flat_pos.shape[0]

    flat_image = image.view(-1, image.shape[-1])

    safe_batch_size = 4*512*512
    opt.zero_grad(set_to_none=True)
    total_loss = torch.zeros((), device=model.device, dtype=torch.float32)

    # Debug: check if we have the right total size
    actual_processed = 0
    use_autocast = scaler is not None

    for i in range(0, flat_pos.shape[0], safe_batch_size):
        chunk_pos = flat_pos[i:i+safe_batch_size]
        chunk_size = chunk_pos.shape[0]
        chunk_target = flat_image[i:i+safe_batch_size]

        with torch.cuda.amp.autocast(enabled=use_autocast):
            chunk_out = model.forward(chunk_pos)
            # Compute loss for this chunk (should be mean over chunk)
            chunk_loss = loss(chunk_out, chunk_target)

        if scaler is not None:
            scaler.scale(chunk_loss).backward()
        else:
            chunk_loss.backward()

        total_loss += chunk_loss.detach().float()
        actual_processed += chunk_size
    total_loss /= ((N / safe_batch_size) + 0.1)  # Average loss over all chunks

    # Verify we processed all pixels
    assert actual_processed == N, f"Processed {actual_processed} but expected {N}"

    if scaler is not None:
        scaler.step(opt)
        scaler.update()
    else:
        opt.step()
    return total_loss


def single_frame_recon_test(png_path, device, num_harmonics=256, hidden_dim=256):
    gt = load_image(png_path, device)
    B, H, W, C = gt.shape

    low_freq_init, idx = sample_fourier_transforms(gt, num_harmonics, device)
    recon, residual = residual_after_modes(gt, idx, device=device)
    # coarse_centers, coarse_sigmas, _ = compute_dog(residual, sigma_min=16)

    high_freq_init, idx = sample_fourier_transforms(residual, num_harmonics, device)
    recon, residual = residual_after_modes(residual, idx, device=device)
    # fine_centers, fine_sigmas, _ = compute_dog(residual, sigma_min=4)

    model = BasicRecon(
        device,
        low_freq_init,
        #coarse_centers,
        #coarse_sigmas,
        high_freq_init,
        #fine_centers,
        #fine_sigmas,
        num_harmonics,
        hidden_dim,
        output_channels=C
    ).to(device)
    # model = torch.compile(model)

    use_amp = device.startswith("cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    psnr_loss = PSNRLoss()
    optimizer = SOAP(
        model.parameters(),
    )

    target = gt.squeeze(0)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for iter in range(10):
        loss = full_image_train(target, model, optimizer, psnr_loss, H, W, scaler=scaler if use_amp else None)
        # loss = psnr_loss(output, target)
        # loss.backward()
        # optimizer.step()

        if (iter + 1) % 100 == 0:
            with torch.no_grad():
                val_output = model.full_image_inf(H=H, W=W)
                val_loss = psnr_loss(val_output, target)
                residual_image = ((val_output - target) + 1.0) * 0.5
                # normalize residual to [0,1]
                residual_image = (residual_image - residual_image.min()) / (residual_image.max() - residual_image.min() + 1e-6)
                save_image(residual_image.permute(2,0,1), f"ai/test_runs/residual_iter_{iter+1:04d}_{num_params}params.png")
                val_output = val_output.permute(2,0,1)
                save_image(val_output, f"ai/test_runs/recon_iter_{iter+1:04d}_{num_params}params.png")
                # switch_map = model.visualize_switching(H=H, W=W)
                # save_image(switch_map, f"ai/test_runs/switchmap_iter_{iter+1:04d}_{num_params}params.png")



if __name__ == "__main__":
    png_path = "uvg/beauty/beauty_frame_000042.png"
    image = load_image(png_path, "cuda:1")
    # centers, sigmas, vals = compute_dog(image, sigma_min=16)
    # save_gaussians_as_image(vals, centers, sigmas)

    # recon, residual = residual_from_gaussians(image, centers, sigmas)
    # # save images
    # save_image(recon.squeeze(0).permute(2,0,1), "ai/test_runs/02dog_recon.png")
    # save_image(residual.squeeze(0).permute(2,0,1), "ai/test_runs/02dog_residual.png")
    # EncodingSelector(image, pool_size=10000, top_k=256)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
        single_frame_recon_test(png_path, "cuda:1", num_harmonics=256, hidden_dim=256)

    print("CUDA Time Total\n", prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    print("CPU Time Total\n", prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    print("CUDA Memory Usage\n", prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))
    print("CPU Memory Usage\n", prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))

    prof.export_chrome_trace("trace.json")
