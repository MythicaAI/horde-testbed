import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

from functools import partial

def kernel_expand(raw_pos, height, width, kernel_size=3):
    """
    Expand the raw positions to include neighboring pixels based on the kernel size.
    Handles edges by padding with a default value.
    :param raw_pos: Tensor of shape [H*W, 2] containing x and y coordinates.
    :param height: Height of the image.
    :param width: Width of the image.
    :param kernel_size: Size of the kernel for expansion.
    :param padding_value: Value to use for padding at the edges.
    :return: Tensor of shape [H*W, kernel_size^2, 2] containing expanded positions.
    """
    H, W = height, width
    x_coords = raw_pos[:, 0]
    y_coords = raw_pos[:, 1]

    # Create a grid of offsets based on the kernel size
    offsets = torch.arange(-(kernel_size // 2), kernel_size // 2 + 1, device=raw_pos.device)
    x_offsets, y_offsets = torch.meshgrid(offsets, offsets, indexing="ij")

    # Expand the coordinates
    expanded_x = torch.clamp(x_coords.unsqueeze(-1) + x_offsets.flatten(), 0, width - 1)
    expanded_y = torch.clamp(y_coords.unsqueeze(-1) + y_offsets.flatten(), 0, height - 1)

    return torch.stack([expanded_x, expanded_y], dim=-1).view(-1, kernel_size ** 2, 2)


def compute_spiral_encoding(x, target_dim, freqs=None, amps=None):
    out = []
    if freqs is None:
        num_harmonics = (target_dim // 2) + 1
        freqs = torch.arange(1, num_harmonics + 1, device=x.device).repeat_interleave(2)
    if amps is None:
        amps = torch.ones_like(freqs)
    for i in range(freqs.shape[0]):
        if i % 2 == 1:
            out += torch.sin(freqs[i] * x * 2 * np.pi) * amps[i]
        elif i % 2 == 0:
            out += torch.cos(freqs[i] * x * 2 * np.pi) * amps[i]
    return torch.cat(out, dim=-1)


def compute_sinusoidal_encoding(coords, target_dim, freqs=None):
    cycle_length = 2
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = freqs.unsqueeze(0) * reshaped_coords

    sin_chunk, cos_chunk = updated_coords.split(num_harmonics * N, dim=1)

    encodings = [
        torch.sin(sin_chunk * 2 * np.pi),
        torch.cos(cos_chunk * 2 * np.pi)
    ]

    encoded_tensor = torch.cat(encodings, dim=-1)
    return torch.cat(encodings, dim=-1)


def compute_mathy_encoding(coords, target_dim, freqs=None):
    cycle_length = 8
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = torch.clamp(freqs.unsqueeze(0) * reshaped_coords, min=1e-5)  # epsilon to help with stability at 0

    sin_chunk, cos_chunk, exp_chunk, log1p_chunk, sqrt_chunk, inv_chunk, rsqrt_chunk, sq_chunk = updated_coords.split(num_harmonics * N, dim=1)

    encodings = torch.cat([
        torch.sin(sin_chunk * 2 * np.pi),
        torch.cos(cos_chunk * 2 * np.pi),
        torch.exp(exp_chunk),
        torch.log1p(log1p_chunk),
        torch.sqrt(sqrt_chunk),
        1 / (inv_chunk),
        torch.rsqrt(rsqrt_chunk),
        sq_chunk * sq_chunk,
    ], dim=1)

    return encodings


def compute_less_mathy_encoding(coords, target_dim, freqs=None):
    cycle_length = 4
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = torch.clamp(freqs.unsqueeze(0) * reshaped_coords, min=1e-5)  # epsilon to help with stability at 0

    sin_chunk, cos_chunk, exp_chunk, log1p_chunk = updated_coords.split(num_harmonics * N, dim=1)

    encodings = torch.cat([
        torch.sin(sin_chunk * 2 * np.pi),
        torch.cos(cos_chunk * 2 * np.pi),
        torch.exp(exp_chunk),
        torch.log1p(log1p_chunk),
    ], dim=1)

    return encodings


def compute_sinexp_encoding(coords, target_dim, freqs=None):
    cycle_length = 2
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = torch.clamp(freqs.unsqueeze(0) * reshaped_coords, min=1e-5)  # epsilon to help with stability at 0

    sin_chunk, exp_chunk = updated_coords.split(num_harmonics * N, dim=1)

    encodings = torch.cat([
        torch.sin(sin_chunk * 2 * np.pi),
        torch.exp(exp_chunk),
    ], dim=1)

    return encodings


def compute_coslog_encoding(coords, target_dim, freqs=None):
    cycle_length = 2
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = torch.clamp(freqs.unsqueeze(0) * reshaped_coords, min=1e-5)  # epsilon to help with stability at 0

    cos_chunk, log1p_chunk = updated_coords.split(num_harmonics * N, dim=1)

    encodings = torch.cat([
        torch.cos(cos_chunk * 2 * np.pi),
        torch.log1p(log1p_chunk),
    ], dim=1)

    return encodings


def compute_analytic_encoding(coords, target_dim, freqs=None, encoding_cycle=["sin", "cos", "exp", "log1p"]):
    cycle_length = len(encoding_cycle)
    N = int(coords.shape[1])

    num_harmonics = int(math.ceil(target_dim / (cycle_length * N)))
    if freqs is None:
        freqs = torch.arange(1, num_harmonics + 1, device=coords.device).repeat_interleave(N).repeat(cycle_length)
        freqs = freqs[:target_dim]  # Ensure we only take as many frequencies as needed

    reshaped_coords = coords.repeat([1, math.ceil(target_dim / N)])
    updated_coords = freqs.unsqueeze(0) * reshaped_coords

    encodings = []
    chunk_size = num_harmonics * N
    chunks = updated_coords.split(chunk_size, dim=1)

    for i, func in enumerate(encoding_cycle):
        chunk = chunks[i]
        if func == "sin":
            encodings.append(torch.sin(chunk * 2 * np.pi))
        elif func == "cos":
            encodings.append(torch.cos(chunk * 2 * np.pi))
        elif func == "exp":
            encodings.append(torch.exp(chunk))
        elif func == "log1p":
            chunk = torch.nn.functional.softplus(chunk)  # Ensure positivity for stability
            encodings.append(torch.log1p(chunk))
        else:
            raise ValueError(f"Unknown encoding function: {func}")
    return torch.cat(encodings, dim=1)


def compute_helmholtz_encoding(coords, target_dim, wavevectors):
    if wavevectors.dim() == 2:
        return torch.sin((coords * (2*torch.pi)) @ wavevectors.T)
    elif wavevectors.dim() == 3:
        return torch.sin((coords.unsqueeze(1) * (2*torch.pi)) @ wavevectors.transpose(1, 2)).squeeze(1)


def compute_rbf_encoding(coords, centers, sigmas):
    dists = torch.cdist(coords, centers, p=2)  # [B, num_centers]
    rbf = torch.exp(-0.5 * (dists / sigmas) ** 2)
    return rbf


def sample_fourier_transforms(image_tensor, num_harmonics, device):
    T, H, W, C = image_tensor.shape
    power = torch.zeros((H, W), dtype=torch.float32, device=device)
    for t in range(T):
        for c in range(C):
            F = torch.fft.fftshift(torch.fft.fft2(image_tensor[t, :, :, c])).to(device)
            power += (F.real**2 + F.imag**2)

    power /= (T * C)
    p = power.flatten()
    p = p / p.sum()
    
    # Use normalized frequencies that work with [-1,1] coordinate system
    ky = torch.fft.fftshift(torch.fft.fftfreq(H, d=2.0/H)).to(device)  # d=2/H for [-1,1] range
    kx = torch.fft.fftshift(torch.fft.fftfreq(W, d=2.0/W)).to(device)  # d=2/W for [-1,1] range

    # Build grids aligned with power's [H, W] layout
    KY, KX = torch.meshgrid(ky, kx, indexing="ij")  # both [H, W]
    KY = KY.reshape(-1)
    KX = KX.reshape(-1)

    # Sample M indices by spectral power
    g = torch.Generator(device=device)
    idx = torch.multinomial(p, num_samples=num_harmonics, replacement=False, generator=g)

    ks = torch.stack([KX[idx], KY[idx]], dim=-1).float()  # [M, 2] = (kx, ky)
    return ks, idx


def residual_after_modes(img, ks_idx, device=None):
    T,H,W,C = img.shape
    device = device or img.device
    img = img.to(device)

    # Build a symmetric mask that keeps selected bins and their conjugates.
    mask = torch.zeros(H*W, dtype=torch.bool, device=device)
    mask[ks_idx] = True

    # Also ensure we keep conjugate positions (because real signals need symmetric spectrum)
    # In shifted indexing, conjugate index for (i,j) is (H-1-i, W-1-j).
    idx_2d = torch.stack([ks_idx // W, ks_idx % W], dim=-1)  # [M, 2] -> (iy, ix)
    conj_iy = (H - 1 - idx_2d[:, 0]) % H
    conj_ix = (W - 1 - idx_2d[:, 1]) % W
    conj_idx = (conj_iy * W + conj_ix)
    mask[conj_idx] = True

    mask2d = mask.view(H, W)

    recon = torch.empty_like(img)
    for t in range(T):
        for c in range(C):
            F = torch.fft.fftshift(torch.fft.fft2(img[t, :, :, c]))
            F_keep = torch.zeros_like(F)
            F_keep[mask2d] = F[mask2d]
            # Back to spatial:
            rec = torch.fft.ifft2(torch.fft.ifftshift(F_keep)).real
            recon[t, :, :, c] = rec

    residual = img - recon
    return recon, residual


def simple_energy_map(residual, blur=5):
    E = residual.pow(2).mean(dim=(0,3))
    if blur > 1:
        E = F.avg_pool2d(E[None,None], kernel_size=blur, stride=1, padding=blur//2)[0,0]
    return E.clamp_min(0)


def _ks_for_sigma(s):
    k = int(6*s + 1);  return k if k % 2 else k+1


def compute_dog(image, sigma_min=8, octaves=3, num_harmonics=256, device=None, r_factor=2):
    B, H, W, C = image.shape
    x = image[0].permute(2,0,1).contiguous()        # [C,H,W]

    # geometric scales: 3 per octave (+1 endpoint)
    s_exp = torch.linspace(0, octaves, octaves*3 + 1, device=x.device)
    sigmas = sigma_min * (2.0 ** s_exp)             # [S]

    # pre-blur
    blurs = [gaussian_blur(x, _ks_for_sigma(float(s)), float(s)) for s in sigmas]

    scores = []   # list of [1,H,W]
    for i in range(len(blurs)-1):
        k = sigmas[i+1] / sigmas[i]
        dog = (blurs[i+1] - blurs[i]).abs().mean(dim=0, keepdim=True)  # [1,H,W]
        scores.append(dog / (k - 1))

    # --- spatial NMS per scale (radius ≈ r_factor * sigma)
    masks2d = []
    for i, sc in enumerate(scores):
        R = max(3, int(round(r_factor * float(sigmas[i]))))
        if R % 2 == 0: R += 1
        pooled = F.max_pool2d(sc, kernel_size=R, stride=1, padding=R//2)
        masks2d.append(sc.eq(pooled))  # [1,H,W] True at local maxima

    # --- cross-scale NMS (keep if ≥ neighbors in scale)
    masks = []
    for i, sc in enumerate(scores):
        if i == 0:
            m_scale = sc >= scores[i+1]
        elif i == len(scores)-1:
            m_scale = sc >= scores[i-1]
        else:
            m_scale = (sc >= scores[i-1]) & (sc >= scores[i+1])
        masks.append(masks2d[i] & m_scale)

    # gather candidates
    all_vals, all_xy, all_sig = [], [], []
    for i, (sc, m) in enumerate(zip(scores, masks)):
        yy, xx = (m[0].nonzero(as_tuple=True))
        if yy.numel() == 0: 
            continue
        v = sc[0, yy, xx]
        all_vals.append(v)
        all_xy.append(torch.stack([xx.float(), yy.float()], dim=-1))
        all_sig.append(torch.full_like(v, float(sigmas[i])))

    if not all_vals:
        return (torch.empty(0,2,device=x.device),
                torch.empty(0,device=x.device),
                torch.empty(0,device=x.device))

    vals = torch.cat(all_vals)      # [N]
    xy   = torch.cat(all_xy)        # [N,2]  (x,y)
    sgm  = torch.cat(all_sig)       # [N]

    # global top-K after NMS (already spatially de-clustered)
    K = min(num_harmonics, vals.numel())
    keep = torch.topk(vals, K).indices
    return xy[keep], sgm[keep], vals[keep]


def residual_from_gaussians(image_bhwc, centers_xy, sigmas):
    """
    image_bhwc : [B,H,W,C] float in [0,1]
    centers_xy : [K,2] (x,y) float pixel coords
    sigmas     : [K]   float pixel sigmas

    Returns: recon_bhwc, residual_bhwc
    """
    x = image_bhwc[0].permute(2,0,1).contiguous()      # [C,H,W]
    C, H, W = x.shape
    device = x.device
    K = centers_xy.shape[0]

    # ----- b = Φ^T y  (via blur-sampling at each σ)
    # group by unique σ to reuse blurs
    cx = centers_xy[:,0].clamp(0, W-1)
    cy = centers_xy[:,1].clamp(0, H-1)
    us, inv = torch.unique(sigmas, sorted=True, return_inverse=True)
    blurs = [gaussian_blur(x, _ks_for_sigma(float(s)), float(s)) for s in us.tolist()]  # each [C,H,W]

    # bilinear sample per group
    b = torch.empty(K, C, device=device)
    for s_idx, u in enumerate(us):
        mask = (inv == s_idx)
        if not mask.any(): 
            continue
        cx_m = cx[mask]; cy_m = cy[mask]
        # grid_sample expects normalized coords in [-1,1]
        gx = (cx_m / (W-1)) * 2 - 1
        gy = (cy_m / (H-1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)  # [1,N,1,2]
        samp = F.grid_sample(blurs[s_idx].unsqueeze(0), grid, align_corners=True)  # [1,C,N,1]
        b[mask] = samp.squeeze(0).squeeze(-1).T  # [N,C]

    # ----- A = Φ^T Φ  (closed form Gaussian overlap in 2D)
    sx2 = sigmas.view(-1,1).to(device).float().pow(2)
    sy2 = sigmas.view(1,-1).to(device).float().pow(2)
    sigma_sum = sx2 + sy2                                             # [K,K]
    dx = centers_xy[:,0].view(-1,1) - centers_xy[:,0].view(1,-1)
    dy = centers_xy[:,1].view(-1,1) - centers_xy[:,1].view(1,-1)
    d2 = dx*dx + dy*dy
    A = (1.0 / (2*math.pi*sigma_sum)) * torch.exp(-0.5 * d2 / sigma_sum)  # [K,K]

    # tiny numerical stabilizer (fixed scale, not a "dial")
    lam = 1e-6 * A.diag().mean()
    A = A + lam * torch.eye(K, device=device)

    # ----- solve for weights per channel
    # a: [K,C]
    a = torch.linalg.solve(A, b)  # stable SPD solve

    # ----- reconstruct: ŷ(x) = Σ_i a_i φ_i(x)
    recon = torch.zeros_like(x)
    for i in range(K):
        s = float(sigmas[i])
        cx_i = cx[i]; cy_i = cy[i]
        R = max(1, int(round(4*s)))
        y0 = max(0, int(cy_i.item()) - R); y1 = min(H, int(cy_i.item()) + R + 1)
        x0 = max(0, int(cx_i.item()) - R); x1 = min(W, int(cx_i.item()) + R + 1)
        yy = torch.arange(y0, y1, device=device) - cy_i
        xx = torch.arange(x0, x1, device=device) - cx_i
        # continuous, L1-normalized 2D Gaussian density
        g = torch.exp(-0.5 * ((yy[:,None]**2 + xx[None,:]**2) / (s*s + 1e-12))) / (2*math.pi*s*s)
        recon[:, y0:y1, x0:x1] += a[i][:, None, None] * g  # broadcast [C,1,1] * [h,w]

    residual = x - recon
    return recon.permute(1,2,0).unsqueeze(0), residual.permute(1,2,0).unsqueeze(0)


def save_gaussians_as_image(scores, idxs, sigmas, path="gaussy.png"):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    
    # Hard-code image dimensions
    W, H = 1920, 1080
    
    # Create matplotlib visualization showing circles at actual locations
    fig, ax = plt.subplots(1, 1, figsize=(16, 9))  # Match aspect ratio
    
    # Set up the plot area
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Match image coordinate system
    
    # Draw circles for each DoG feature
    for i in range(len(idxs)):
        x0, y0 = idxs[i].cpu()
        sigma = sigmas[i].cpu().item()
        
        # Create circle with radius = sigma, alpha based on relative sigma size
        circle = Circle((x0, y0), radius=sigma * 2, 
                       fill=False, edgecolor='red', 
                       linewidth=1, alpha=0.7)
        ax.add_patch(circle)
        
        # Also add a small dot at the center
        ax.plot(x0, y0, 'r.', markersize=2, alpha=0.8)
    
    ax.set_title(f'DoG Features: {len(idxs)} circles at detected locations\n(Circle radius = sigma)')
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    
    plt.savefig(path.replace('.png', '_circles.png'), bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"DoG circles visualization saved to {path.replace('.png', '_circles.png')}")
    print(f"Found {len(idxs)} features with sigma range: [{sigmas.min():.1f}, {sigmas.max():.1f}]")


def weighted_random_centers(E, M, bsigma=0.1, beta=0.5, eps=1e-12):
    H, W = E.shape
    w = E.flatten()
    w = w / (w.sum() + eps)

    # weighted-without-replacement via Gumbel-top-k (fast + one pass)
    g = -torch.log(-torch.log(torch.rand_like(w)))
    scores = torch.log(w + eps) + g
    idx = scores.topk(M).indices
    y = (idx // W).float(); x = (idx % W).float()
    # norm to [-1, 1] - ensure consistent with coordinate system
    y = (y / (H - 1)) * 2 - 1
    x = (x / (W - 1)) * 2 - 1
    xy_centers = torch.stack([x, y], dim=-1)

    energies = w[idx] + eps
    med = energies.median()
    # Scale sigmas appropriately for [-1,1] coordinate system
    sigmas = ((energies / med).pow(-beta) * bsigma).clamp_min(1e-6)

    return xy_centers, sigmas


def compute_full_helmholtz_encoding(coords, target_dim, wavevectors):
    transform = (coords * 2 * np.pi) @ wavevectors.T
    sin_term = torch.sin(transform).unsqueeze(-1)
    cos_term = torch.cos(transform)

    k_norm = torch.linalg.norm(wavevectors, dim=1, keepdim=True).clamp(min=1e-5)
    k_rot  = torch.stack([-wavevectors[:, 1], wavevectors[:, 0]], dim=1) / k_norm
    v_k    = (cos_term[:, :, None] * wavevectors) / k_norm
    v_krot = cos_term[:, :, None] * k_rot
    return torch.cat([sin_term, v_k, v_krot], dim=-1)

def compute_linear_encoding(x, target_dim):
    return x.repeat(1, target_dim // x.shape[-1])[:, :target_dim]


def compute_polynomial_encoding(x, max_degree):
    out = []
    for i in range(1, max_degree + 1):
        out.append(x ** i)
    return torch.cat(out, dim=-1)


def compute_gaussian_encoding(x, target_dim, std=10.0, seed=42):
    generator = torch.Generator(device=x.device).manual_seed(seed)
    B = torch.randn(x.shape[1], target_dim // 2, generator=generator, device=x.device) * std
    x_proj = 2 * torch.pi * x @ B  # [B, F]
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def compute_targeted_encodings(x, target_dim, scheme="spiral", include_raw=False, freqs=None, seed=42, encoding_cycle=None):
    _, N = x.shape
    encodings = []

    if include_raw:
        if scheme == "full_helmholtz":
            target = N * 2 + 1
            encodings.append(x.repeat(target))
        encodings.append(x)
        target_dim -= N
    target_dim = int(target_dim)

    if target_dim > 0:
        if scheme in ["spiral", "sinusoidal", "mathy", "less_mathy", "sinexp", "coslog", "analytic", "helmholtz", "full_helmholtz"]:
            encoding_fn = {
                "spiral": compute_spiral_encoding,
                "sinusoidal": compute_sinusoidal_encoding,
                "mathy": compute_mathy_encoding,
                "less_mathy": compute_less_mathy_encoding,
                "sinexp": compute_sinexp_encoding,
                "coslog": compute_coslog_encoding,
                "analytic": partial(compute_analytic_encoding, encoding_cycle=encoding_cycle),
                "helmholtz": compute_helmholtz_encoding,
                "full_helmholtz": compute_full_helmholtz_encoding,
            }[scheme]
            encodings.append(encoding_fn(x, target_dim, freqs))
        elif scheme == "gaussian":
            encodings.append(compute_gaussian_encoding(x, target_dim, seed=seed))
        elif scheme == "linear":
            encodings.append(compute_linear_encoding(x, target_dim))
        elif scheme == "polynomial":
            deg = target_dim // x.shape[-1]
            encodings.append(compute_polynomial_encoding(x, deg))
        elif scheme is None:
            encodings.append(torch.zeros(x.shape[0], target_dim, device=x.device))
        else:
            raise ValueError(f"Unknown encoding scheme: {scheme}")
    return torch.cat(encodings, dim=-1)


class SineLayer(nn.Module):
    """Siren activation function."""
    def __init__(self, in_features, out_features, is_first=False, omega=30.0):
        super().__init__()
        self.omega = omega
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.linear.in_features) / self.omega,
                    np.sqrt(6 / self.linear.in_features) / self.omega
                )

    def forward(self, x):
        return torch.sin(self.omega * self.linear(x))


class Tanh01(nn.Module):
    def forward(self, x):
        return 0.5 * (torch.tanh(x) + 1.0)
