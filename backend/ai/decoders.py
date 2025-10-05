import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from encoding_utils import SineLayer, Tanh01, kernel_expand, compute_targeted_encodings, compute_helmholtz_encoding, compute_analytic_encoding

class VFXSpiralNetDecoder(nn.Module):
    def __init__(self, device, **kwargs):
        defaults = {
            "latent_dim": 128,
            "trunk_pos_channels": 0,
            "trunk_pos_scheme": "sinusoidal",
            "trunk_pos_include_raw": True,
            "trunk_time_channels": 0,
            "trunk_time_scheme": "sinusoidal",
            "trunk_time_include_raw": True,
            "film_time_channels": 8,
            "film_time_scheme": "spiral",
            "film_time_include_raw": True,
            "film_pos_channels": 16,
            "film_pos_scheme": "spiral",
            "film_pos_include_raw": True,
            "output_channels": 4,
            "hidden_dim": 64,
            "prefilm_dims": 32,
            "apply_film": [1],
            "num_layers": 4,
            "learned_encodings": False,
            "siren_film": False,
            "siren_trunk": False,
            "encoding_cycle": None,
            "frequency_initialization": "linear",
            "target_resolution": 1024,
            "device": device,
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)
        torch.set_default_device(self.device)
        super().__init__()
        input_dim = self.latent_dim + self.trunk_pos_channels + self.trunk_time_channels

        if self.latent_dim > 0:
            self.latent = nn.Parameter(torch.randn(self.latent_dim, 1), requires_grad=True)

        if self.prefilm_dims > 0:
            if self.film_time_channels > 0:
                if self.siren_film:
                    self.time_embed = SineLayer(self.film_time_channels, self.prefilm_dims, is_first=True)
                else:
                    self.time_embed = nn.Sequential(
                        nn.Linear(self.film_time_channels, self.prefilm_dims),
                        nn.ReLU(),
                    )

            if self.film_pos_channels > 0:
                if self.siren_film:
                    self.pos_embed = SineLayer(self.film_pos_channels, self.prefilm_dims, is_first=True)
                else:
                    self.pos_embed = nn.Sequential(
                        nn.Linear(self.film_pos_channels, self.prefilm_dims),
                        nn.ReLU(),
                    )

            if self.film_time_channels > 0 or self.film_pos_channels > 0:
                prefilm_input_dim = (self.film_time_channels > 0) * self.prefilm_dims + (self.film_pos_channels > 0) * self.prefilm_dims
                self.film = nn.Linear(prefilm_input_dim, self.hidden_dim * 2)
        else:
            if self.film_time_channels > 0 or self.film_pos_channels > 0:
                prefilm_input_dim = self.film_time_channels + self.film_pos_channels
                self.film = nn.Linear(prefilm_input_dim, self.hidden_dim * 2)

        self.layers = nn.ModuleList()
        if self.siren_trunk:
            self.layers.append(SineLayer(input_dim, self.hidden_dim, is_first=True))
        else:
            self.layers.append(nn.Linear(input_dim, self.hidden_dim))
        for i in range(1, self.num_layers - 1):
            if self.siren_trunk:
                self.layers.append(SineLayer(self.hidden_dim, self.hidden_dim))
            else:
                self.layers.append(nn.GELU())
                self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        if not self.siren_trunk:
            self.layers.append(nn.GELU())
        self.layers.append(nn.Linear(self.hidden_dim, self.output_channels))
        self.layers.append(nn.Sigmoid())

        self.max_frequency = self.target_resolution / 2.0

        if self.learned_encodings:
            encoding_len = len(self.encoding_cycle) if self.encoding_cycle is not None else 1
            target_pos_dim = self.film_pos_channels - (2 if self.film_pos_include_raw else 0)
            num_pos_harmonics = int(math.ceil(target_pos_dim / (encoding_len * 2)))
            if self.frequency_initialization == "linear":
                base_pos_freqs = torch.linspace(0, self.max_frequency, steps=num_pos_harmonics)
            elif self.frequency_initialization == "exponential":
                base_pos_freqs = 2 ** linspace(0, math.log2(self.max_frequency), steps=num_pos_harmonics)
            elif self.frequency_initialization == "inverse":
                base_pos_freqs = torch.rand(1, num_pos_harmonics).clamp(min=1/self.max_frequency)
            else:
                raise ValueError(f"Unknown frequency initialization: {self.frequency_initialization}")
            if self.film_pos_scheme == "helmholtz" or self.film_pos_scheme == "full_helmholtz":
                base_pos_freqs = torch.exp(
                    torch.empty(target_pos_dim).uniform_(0.0, math.log(self.max_frequency))
                )
                directions = torch.empty(target_pos_dim).uniform_(0.0, 2 * math.pi)
                unit_vectors = torch.stack([torch.cos(directions), torch.sin(directions)], dim=1)
                full_pos_freqs = (base_pos_freqs.unsqueeze(1) * unit_vectors)
                if self.film_pos_scheme == "full_helmholtz":
                    self.pos_head = ResonantHead(target_pos_dim, n_dim=2)
            else:
                full_pos_freqs = base_pos_freqs.repeat_interleave(2).repeat(encoding_len)
            self.pos_freqs = nn.Parameter(full_pos_freqs[:target_pos_dim].float(), requires_grad=True)

            target_time_dim = self.film_time_channels - (1 if self.film_time_include_raw else 0)
            num_time_harmonics = int(math.ceil(target_time_dim / encoding_len))
            if self.frequency_initialization == "linear":
                base_time_freqs = torch.arange(1, num_time_harmonics + 1)
            elif self.frequency_initialization == "exponential":
                base_time_freqs = 2 ** torch.arange(1, num_time_harmonics + 1)
            elif self.frequency_initialization == "inverse":
                initial_freqs = torch.abs(torch.randn(1, num_time_harmonics + 1)).clamp(min=1e-3)
                base_time_freqs = 1.0 / initial_freqs
            else:
                raise ValueError(f"Unknown frequency initialization: {self.frequency_initialization}")
            full_time_freqs = base_time_freqs.repeat_interleave(1).repeat(encoding_len)
            self.time_freqs = nn.Parameter(full_time_freqs[:target_time_dim].float(), requires_grad=True)
        else:
            self.pos_freqs = None
            self.time_freqs = None

    def forward(self, raw_pos, time, latent=None, return_hidden_layer=None):
        B, _ = raw_pos.shape
        trunk_input = []
        if self.latent_dim > 0:
            trunk_input.append(self.latent.expand(-1, B).T)
        
        if self.trunk_pos_channels > 0:
            pos_enc = compute_targeted_encodings(
                raw_pos,
                self.trunk_pos_channels,
                scheme=self.trunk_pos_scheme,
                include_raw=self.trunk_pos_include_raw,
            )
            trunk_input.append(pos_enc)

        if self.trunk_time_channels > 0:
            trunk_time = compute_targeted_encodings(
                time,
                self.trunk_time_channels,
                scheme=self.trunk_time_scheme,
                include_raw=self.trunk_time_include_raw,
            )
            trunk_input.append(trunk_time)

        trunk_input = torch.cat(trunk_input, dim=-1)

        film_input = []
        if self.film_pos_channels > 0:
            if self.frequency_initialization == "inverse":
                pos_freqs = 1.0 / (self.pos_freqs)
            else:
                pos_freqs = self.pos_freqs
            film_pos = compute_targeted_encodings(
                raw_pos,
                self.film_pos_channels,
                scheme=self.film_pos_scheme,
                include_raw=self.film_pos_include_raw,
                freqs=pos_freqs,
                encoding_cycle=self.encoding_cycle,
            )
            if self.film_pos_scheme == "full_helmholtz":
                film_pos = self.pos_head(film_pos)
                print("film pos stats:", film_pos.mean().item(), film_pos.std().item())
                print("more stats:", film_pos.min().item(), film_pos.max().item())
            if self.prefilm_dims > 0:
                film_pos = self.pos_embed(film_pos)
            film_input.append(film_pos)

        if self.film_time_channels > 0:
            if self.frequency_initialization == "inverse":
                time_freqs = 1.0 / (self.time_freqs)
            else:
                time_freqs = self.time_freqs
            film_time = compute_targeted_encodings(
                time,
                self.film_time_channels,
                scheme=self.film_time_scheme,
                include_raw=self.film_time_include_raw,
                freqs=time_freqs,
                encoding_cycle=self.encoding_cycle,
            )
            if self.prefilm_dims > 0:
                film_time = self.time_embed(film_time)
            film_input.append(film_time)

        if film_input:
            film_input = torch.cat(film_input, dim=-1)
        else:
            film_input = None

        outputs = trunk_input
        for i, layer in enumerate(self.layers):
            if i in self.apply_film and film_input is not None:
                gamma, beta = self.film(film_input).chunk(2, dim=-1)
                outputs = layer((gamma * outputs) + beta)
            else:
                outputs = layer(outputs)
            if return_hidden_layer is not None and i == return_hidden_layer:
                return outputs
        return outputs


class ResonantHead(torch.nn.Module):
    """
    Used to collapse the helmholtz encodings into a single channel.
    """
    def __init__(self, L, n_dim:int=2):
        super().__init__()
        self.W = torch.nn.Parameter(torch.randn(L, n_dim * 2 + 1))
        self.b = torch.nn.Parameter(torch.zeros(L))

    def forward(self, V):  # V: [N, B, 5]
        # s[n,l] = V[n,l,:] · W[l,:] + b[l]
        return torch.einsum('nlc,lc->nl', V, self.W) + self.b


class SpecificDecoder(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        torch.set_default_device(device)
        self.trunk_interface = 64
        self.trunk_head = nn.Sequential(
            nn.Linear(3, self.trunk_interface),
            nn.GELU(),
            nn.LayerNorm(self.trunk_interface),
        )

        self.output_channels = 3
        self.trunk_base = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, self.output_channels),
            nn.Sigmoid()
        )
        
        self.pos_embeddings = 256
        self.prefilm_dim = 128
        self.pos_embed = nn.Sequential(
            nn.Linear(self.pos_embeddings, self.prefilm_dim),
            nn.GELU(),
            nn.LayerNorm(self.prefilm_dim),
        )

        self.time_embeddings = 64
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embeddings, self.prefilm_dim),
            nn.GELU(),
            nn.LayerNorm(self.prefilm_dim),
        )

        self.film = nn.Linear(self.prefilm_dim * 2, self.trunk_interface * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

        pos_res = 2048
        time_res = 1024
        self.pos_encoding_len = self.pos_embeddings - 2
        self.base_pos_freqs = torch.exp(
            torch.empty(self.pos_encoding_len).uniform_(0.0, math.log(pos_res / 2))
        )
        directions = torch.empty(self.pos_encoding_len).uniform_(0.0, 2 * math.pi)
        unit_vectors = torch.stack([torch.cos(directions), torch.sin(directions)], dim=1)
        self.wavevectors = nn.Parameter(self.base_pos_freqs.unsqueeze(1) * unit_vectors)

        self.time_encoding_len = self.time_embeddings - 1
        self.base_time_freqs = nn.Parameter(torch.exp(
            torch.empty(self.time_encoding_len).uniform_(0.0, math.log(time_res / 2))
        ))
    
    def forward(self, raw_pos, time):
        trunk_input = torch.cat([raw_pos, time], dim=-1)
        trunk_out = self.trunk_head(trunk_input)

        pos_enc = compute_helmholtz_encoding(
            raw_pos,
            self.pos_encoding_len,
            self.wavevectors,
        )

        time_enc = compute_analytic_encoding(
            time,
            self.time_encoding_len,
            freqs=self.base_time_freqs,
            encoding_cycle=["sin"],
        )

        pos_input = self.pos_embed(torch.cat([raw_pos, pos_enc], dim=-1))
        time_input = self.time_embed(torch.cat([time, time_enc], dim=-1))

        film_input = torch.cat([pos_input, time_input], dim=-1)
        film_gamma, film_beta = self.film(film_input).chunk(2, dim=-1)
        modulated = ((1 + film_gamma) * trunk_out) + film_beta
        output = self.trunk_base(modulated)
        return output

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


class DualDecoder(nn.Module):
    def __init__(self, device, low_freqs, high_freqs):
        super().__init__()
        torch.set_default_device(device)
        self.num_harmonics = 256
        self.low_freqs = low_freqs
        self.high_freqs = high_freqs
        self.output_channels = 3

        self.low_linear_enc = nn.Linear(2, self.num_harmonics)
        self.low_linear_first = nn.Sequential(
            nn.Linear(self.num_harmonics * 2, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
        )

        self.high_linear_enc = nn.Linear(2, self.num_harmonics)
        self.high_linear_first = nn.Sequential(
            nn.Linear(self.num_harmonics * 2, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
        )

        self.time_linear_enc = nn.Linear(1, self.num_harmonics)

        self.base_time_freqs = torch.exp(torch.empty(self.num_harmonics).uniform_(0.0, math.log(500)))

        self.time_film = nn.Sequential(
            nn.Linear(self.num_harmonics * 2, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics * 2),
        )
        # Initialize FiLM to identity: gamma=0 (so 1+gamma=1), beta=0
        nn.init.zeros_(self.time_film[-1].weight)
        nn.init.zeros_(self.time_film[-1].bias)

        self.low_trunk = nn.Sequential(
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.output_channels + 1),
            nn.Sigmoid()
        )

        self.high_trunk = nn.Sequential(
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.num_harmonics),
            nn.ReLU(),
            nn.Linear(self.num_harmonics, self.output_channels + 1),
            nn.Sigmoid()
        )
        
        # Call initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Skip the FiLM layer since we already initialized it
                if m is not self.time_film[-1]:
                    nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                    nn.init.zeros_(m.bias)

    def forward(self, raw_pos, time):
        low_enc_hh = compute_helmholtz_encoding(
            raw_pos,
            self.low_freqs.shape[0],
            self.low_freqs,
        )
        low_enc_lin = self.low_linear_enc(raw_pos)
        low_enc = torch.cat([low_enc_hh, low_enc_lin], dim=-1)
        low_branch_start = self.low_linear_first(low_enc)

        high_enc_hh = compute_helmholtz_encoding(
            raw_pos,
            self.high_freqs.shape[0],
            self.high_freqs,
        )
        high_enc_lin = self.high_linear_enc(raw_pos)
        high_enc = torch.cat([high_enc_hh, high_enc_lin], dim=-1)
        high_branch_start = self.high_linear_first(high_enc)

        time_enc_lin = self.time_linear_enc(time)
        time_enc_sin = compute_analytic_encoding(
            time,
            self.num_harmonics,
            freqs=self.base_time_freqs,
            encoding_cycle=["sin"],
        )
        time_enc = torch.cat([time_enc_sin, time_enc_lin], dim=-1)
        time_gamma, time_beta = self.time_film(time_enc).chunk(2, dim=-1)

        low_modulated = ((1 + time_gamma) * low_branch_start) + time_beta
        high_modulated = ((1 + time_gamma) * high_branch_start) + time_beta

        low_out = self.low_trunk(low_modulated)
        high_out = self.high_trunk(high_modulated)

        rgb1, weight1 = low_out[..., :self.output_channels], low_out[..., -1:]
        rgb2, weight2 = high_out[..., :self.output_channels], high_out[..., -1:]
        weights = torch.softmax(torch.cat([weight1, weight2], dim=-1), dim=-1)
        w1, w2 = weights[..., 0:1], weights[..., 1:2]
        output = rgb1 * w1 + rgb2 * w2
        return output


class TestDecoder(nn.Module):  #WeirdRef
    def __init__(self, device, freq_init=None, **kwargs):
        defaults = {
            "embedding_dim": 256,
            "time_encoding_len": 128,
            "pos_encoding_len": 256,
            "output_channels": 3,
            "time_res": 1024,
        }
        defaults.update(kwargs)

        for key, value in defaults.items():
            setattr(self, key, value)
        super().__init__()
        torch.set_default_device(device)

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_encoding_len, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
        )

        self.time_pos_modulate = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),  # -> [δγ | β]
        )
        # Identity FiLM init: δγ=0, β=0  ⇒  γ=1, β=0
        nn.init.zeros_(self.time_pos_modulate[-1].weight)
        nn.init.zeros_(self.time_pos_modulate[-1].bias)

        self.pos_embed = nn.Sequential(
            nn.Linear(self.pos_encoding_len, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
        )

        self.wavevectors = freq_init

        self.base_time_freqs = nn.Parameter(torch.exp(
            torch.empty(self.time_encoding_len).uniform_(0.0, math.log(self.time_res / 2))
        ))

        self.trunk_base = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim * 2, self.output_channels),
            nn.Sigmoid()
        )

        for m in self.trunk_base:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
                nn.init.zeros_(m.bias)

    def forward(self, raw_pos, time):
        time_enc = compute_analytic_encoding(
            time,
            self.time_encoding_len,
            freqs=self.base_time_freqs,
            encoding_cycle=["sin"],
        )

        encoded_pos = compute_helmholtz_encoding(
            raw_pos,
            self.pos_encoding_len,
            self.wavevectors,
        )

        time_embedding = self.time_embed(time_enc)
        gamma, beta = self.time_pos_modulate(time_embedding).chunk(2, dim=-1)

        pos_embedding = self.pos_embed(encoded_pos)
        pos_embedding = (( 1 + gamma) * pos_embedding) + beta
        output = self.trunk_base(torch.cat([pos_embedding, time_embedding], dim=-1))
        return output


class TLearnedWaveVectorDecoder(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        torch.set_default_device(device)
        self.embedding_dim = 256
        self.time_encoding_len = 128
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_encoding_len, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
        )

        self.time_pos_transform = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.GELU(),
            nn.Linear(64, 2),
            nn.Tanh(),
        )

        self.pos_encoding_len = 256

        self.time_wavevector_transform = nn.Sequential(
            nn.Linear(self.embedding_dim, self.pos_encoding_len * 2),
            nn.GELU(),
            nn.Linear(self.pos_encoding_len * 2, self.pos_encoding_len * 2),
            nn.Tanh(),
            nn.LayerNorm(self.pos_encoding_len * 2),
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(self.pos_encoding_len, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
        )

        pos_res = 2048
        time_res = 1024
        self.base_pos_freqs = torch.exp(
            torch.empty(self.pos_encoding_len).uniform_(0.0, math.log(pos_res / 2))
        )
        directions = torch.empty(self.pos_encoding_len).uniform_(0.0, 2 * math.pi)
        unit_vectors = torch.stack([torch.cos(directions), torch.sin(directions)], dim=1)
        self.wavevectors = (self.base_pos_freqs.unsqueeze(1) * unit_vectors)

        self.base_time_freqs = nn.Parameter(torch.exp(
            torch.empty(self.time_encoding_len).uniform_(0.0, math.log(time_res / 2))
        ))

        self.output_channels = 3
        self.trunk_base = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim // 2, self.output_channels),
            nn.Sigmoid()
        )

    def safe_logit(self, p, eps=1e-5):
        p = p.clamp(eps, 1 - eps)
        return torch.log(p) - torch.log1p(-p)
    
    def forward(self, raw_pos, time):
        time_enc = compute_analytic_encoding(
            time,
            self.time_encoding_len,
            freqs=self.base_time_freqs,
            encoding_cycle=["sin"],
        )

        time_embedding = self.time_embed(time_enc)
        transform = self.time_pos_transform(time_embedding)

        logit_pos = self.safe_logit(raw_pos)
        transformed_pos = torch.sigmoid(logit_pos + transform)

        wavevector_transform = self.time_wavevector_transform(time_embedding).reshape(-1, self.pos_encoding_len, 2)
        wavevectors = self.wavevectors[None, :, :] + wavevector_transform

        encoded_pos = compute_helmholtz_encoding(
            transformed_pos,
            self.pos_encoding_len,
            wavevectors,
        )

        pos_embedding = self.pos_embed(encoded_pos)
        output = self.trunk_base(torch.cat([pos_embedding, time_embedding], dim=-1))
        return output


class TModulatedDecoder(nn.Module):
    def __init__(self, device, freq_init=None, **kwargs):
        defaults = {
            "embedding_dim": 256,
            "time_encoding_len": 128,
            "pos_encoding_len": 256,
            "output_channels": 3,
            "pos_res": 2048,
            "time_res": 1024,
        }
        defaults.update(kwargs)
        for key, value in defaults.items():
            setattr(self, key, value)
        super().__init__()
        torch.set_default_device(device)

        self.time_embed = nn.Sequential(
            nn.Linear(self.time_encoding_len, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
        )

        self.time_pos_modulate = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim * 2),
            nn.Tanh(),
            nn.LayerNorm(self.embedding_dim * 2),
        )

        self.pos_embed = nn.Sequential(
            nn.Linear(self.pos_encoding_len, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.LayerNorm(self.embedding_dim),
        )

        if freq_init is None:
            self.base_pos_freqs = torch.exp(
                torch.empty(self.pos_encoding_len).uniform_(0.0, math.log(pos_res / 2))
            )
            directions = torch.empty(self.pos_encoding_len).uniform_(0.0, 2 * math.pi)
            unit_vectors = torch.stack([torch.cos(directions), torch.sin(directions)], dim=1)
            self.wavevectors = nn.Parameter(self.base_pos_freqs.unsqueeze(1) * unit_vectors)
        else:
            assert freq_init.shape == (self.pos_encoding_len, 2)
            self.wavevectors = nn.Parameter(freq_init)

        self.base_time_freqs = nn.Parameter(torch.exp(
            torch.empty(self.time_encoding_len).uniform_(0.0, math.log(self.time_res / 2))
        ))

        self.trunk_base = nn.Sequential(
            nn.Linear(self.embedding_dim * 2, self.embedding_dim * 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 2, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.GELU(),
            nn.Linear(self.embedding_dim // 2, self.output_channels),
            nn.Sigmoid()
        )

    def forward(self, raw_pos, time):
        time_enc = compute_analytic_encoding(
            time,
            self.time_encoding_len,
            freqs=self.base_time_freqs,
            encoding_cycle=["sin"],
        )

        encoded_pos = compute_helmholtz_encoding(
            raw_pos,
            self.pos_encoding_len,
            self.wavevectors,
        )

        time_embedding = self.time_embed(time_enc)
        gamma, beta = self.time_pos_modulate(time_embedding).chunk(2, dim=-1)

        pos_embedding = self.pos_embed(encoded_pos)
        pos_embedding = (gamma * pos_embedding) + beta
        output = self.trunk_base(torch.cat([pos_embedding, time_embedding], dim=-1))
        return output


class BigDecoder(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        torch.set_default_device(device)
        self.trunk_head = nn.Sequential(
            nn.Linear(3, 512),
            nn.ReLU(),
        )

        self.output_channels = 3
        self.trunk_base = nn.Sequential(
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.output_channels),
            nn.Sigmoid()
        )
        
        self.pos_embeddings = 512
        self.pos_embed = nn.Sequential(
            nn.Linear(self.pos_embeddings, 1024),
            nn.ReLU(),
        )

        self.time_embeddings = 128
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embeddings, 1024),
            nn.ReLU(),
        )

        self.film = nn.Linear(2048, 1024)

        pos_res = 1024
        time_res = 500
        self.pos_encoding_len = self.pos_embeddings - 2
        self.base_pos_freqs = torch.exp(
            torch.empty(self.pos_encoding_len).uniform_(0.0, math.log(pos_res / 2))
        )
        directions = torch.empty(self.pos_encoding_len).uniform_(0.0, 2 * math.pi)
        unit_vectors = torch.stack([torch.cos(directions), torch.sin(directions)], dim=1)
        self.wavevectors = (self.base_pos_freqs.unsqueeze(1) * unit_vectors)

        self.time_encoding_len = self.time_embeddings - 1
        self.base_time_freqs = torch.exp(
            torch.empty(self.time_encoding_len).uniform_(0.0, math.log(time_res / 2))
        )
    
    def forward(self, raw_pos, time):
        trunk_input = torch.cat([raw_pos, time], dim=-1)
        trunk_out = self.trunk_head(trunk_input)

        pos_enc = compute_helmholtz_encoding(
            raw_pos,
            self.pos_encoding_len,
            self.wavevectors,
        )

        time_enc = compute_analytic_encoding(
            time,
            self.time_encoding_len,
            freqs=self.base_time_freqs,
            encoding_cycle=["sin"],
        )

        pos_input = self.pos_embed(torch.cat([raw_pos, pos_enc], dim=-1))
        time_input = self.time_embed(torch.cat([time, time_enc], dim=-1))

        film_input = torch.cat([pos_input, time_input], dim=-1)
        film_gamma, film_beta = self.film(film_input).chunk(2, dim=-1)
        modulated = (film_gamma * trunk_out) + film_beta
        output = self.trunk_base(modulated)
        return output


class CoordFlowish(nn.Module):
    def __init__(self, device, **kwargs):
        super().__init__()
        torch.set_default_device(device)
        self.embedding_dim = 192
        self.encoding_dim = 180
        self.output_channels = 3
        self.time_embedding = nn.Sequential(
            nn.Linear(self.encoding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
        )

        self.time_transform = nn.Sequential(
            nn.Linear(self.embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
        
        self.x_embedding = nn.Sequential(
            nn.Linear(self.encoding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
        )

        self.y_embedding = nn.Sequential(
            nn.Linear(self.encoding_dim, self.embedding_dim),
            nn.ReLU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.ReLU(),
        )

        self.trunk_base = nn.Sequential(
            nn.Linear(self.embedding_dim * 3, self.embedding_dim * 3),
            nn.GELU(),
            nn.Linear(self.embedding_dim * 3, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, self.output_channels),
            nn.Sigmoid()
        )

        pos_res = 1024
        time_res = 500
        self.x_freqs = torch.exp(
            torch.empty(self.encoding_dim).uniform_(0.0, math.log(pos_res / 2))
        )
        self.y_freqs = torch.exp(
            torch.empty(self.encoding_dim).uniform_(0.0, math.log(pos_res / 2))
        )
        self.time_freqs = torch.exp(
            torch.empty(self.encoding_dim).uniform_(0.0, math.log(time_res / 2))
        )
    
    def forward(self, raw_pos, time):
        raw_pos = raw_pos * 2.0 - 1.0  # scale to [-1, 1]
        time_enc = compute_analytic_encoding(
            time,
            self.encoding_dim,
            freqs=self.time_freqs,
            encoding_cycle=["sin", "cos"],
        )

        time_embedding = self.time_embedding(time_enc)

        transform = self.time_transform(time_embedding)
        s, theta, dx, dy = transform.chunk(4, dim=-1)
        s = torch.exp(0.1 * s).clamp(0.7, 1.3)
        theta = math.pi * torch.tanh(theta)  # rotate between -pi and pi
        dx = 0.1 * torch.tanh(dx)
        dy = 0.1 * torch.tanh(dy)
        rot1 = torch.stack([s * torch.cos(theta), -s * torch.sin(theta)], dim=-1)
        rot2 = torch.stack([s * torch.sin(theta),  s * torch.cos(theta)], dim=-1)
        translate = torch.stack([dx, dy], dim=-1)
        transform_mat = torch.cat([rot1, rot2, translate], dim=-2)
        padded_pos = torch.cat([raw_pos, torch.ones(raw_pos.shape[0], 1, device=raw_pos.device)], dim=-1)

        # print(f"Raw pos stats: {raw_pos.mean().item():.4f} ± {raw_pos.std().item():.4f}")
        transformed_pos = (padded_pos.unsqueeze(1) @ transform_mat).squeeze(1)
        recentered_pos = (transformed_pos + 1.0) / 2.0
        transformed_pos = torch.sigmoid(recentered_pos)
        # print(f"Transformed pos stats: {transformed_pos.mean().item():.4f} ± {transformed_pos.std().item():.4f}")

        x_enc = compute_analytic_encoding(
            transformed_pos[:, :1],
            self.encoding_dim,
            freqs=self.x_freqs,
            encoding_cycle=["sin", "cos"],
        )
        x_embedding = self.x_embedding(x_enc)

        y_enc = compute_analytic_encoding(
            transformed_pos[:, 1:2],
            self.encoding_dim,
            freqs=self.y_freqs,
            encoding_cycle=["sin", "cos"],
        )
        y_embedding = self.y_embedding(y_enc)

        trunk_input = torch.cat([x_embedding, y_embedding, time_embedding], dim=-1)
        output = self.trunk_base(trunk_input).clamp(0.0, 1.0)
        return output
