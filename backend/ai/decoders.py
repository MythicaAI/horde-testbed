import math

import torch
import torch.nn as nn

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
        self.trunk_head = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
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
        self.pos_embed = nn.Sequential(
            nn.Linear(self.pos_embeddings, 128),
            nn.ReLU(),
        )

        self.time_embeddings = 64
        self.time_embed = nn.Sequential(
            nn.Linear(self.time_embeddings, 128),
            nn.ReLU(),
        )

        self.film = nn.Linear(256, 128)

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
