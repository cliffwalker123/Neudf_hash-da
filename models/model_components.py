import math
from typing import Any, Dict, Union
import torch
import torch.nn as nn
import tinycudann as tcnn

class VanillaFrequency(nn.Module):
    def __init__(self, in_channels: int, config: Dict[str, Any]):
        super().__init__()
        self.N_freqs = config['n_frequencies']
        self.in_channels, self.n_input_dims = in_channels, in_channels
        self.funcs = [torch.sin, torch.cos]
        self.freq_bands = 2**torch.linspace(0, self.N_freqs-1, self.N_freqs)
        self.n_output_dims = self.in_channels * (len(self.funcs) * self.N_freqs)
        self.n_masking_step = config.get('n_masking_step', 0)
        self.update_step(None, None) # mask should be updated at the beginning each step

    def forward(self, x):
        out = []
        for freq, mask in zip(self.freq_bands, self.mask):
            for func in self.funcs:
                out += [func(freq*x) * mask]
        return torch.cat(out, -1)

    def update_step(self, epoch: int, global_step: int):
        if self.n_masking_step <= 0 or global_step is None:
            self.mask = torch.ones(self.N_freqs, dtype=torch.float32)
            print("Initializing VanillaFrequency mask: ", self.mask)
        else:
            self.mask = (1. - torch.cos(math.pi * (global_step / self.n_masking_step * self.N_freqs - torch.arange(0, self.N_freqs)).clamp(0, 1))) / 2.
            print(f'Update mask: {global_step}/{self.n_masking_step} {self.mask}')

class CompositeEncoding(nn.Module):
    def __init__(
        self,
        encoding,
        include_xyz: bool = False,
        xyz_scale: float = 1.,
        xyz_offset: float = 0.,
        n_frequencies: int = 0,
        device: str = 'cuda',
    ):
        super(CompositeEncoding, self).__init__()
        self.encoding = encoding
        self.include_xyz, self.xyz_scale, self.xyz_offset = include_xyz, xyz_scale, xyz_offset
        self.n_output_dims = self.encoding.n_output_dims
        self.n_frequencies = n_frequencies
        self.device = device
        if self.include_xyz:
            self.n_output_dims += self.encoding.n_input_dims
        if n_frequencies > 0:
            config_encoding_x = {
                "otype": "VanillaFrequency",
                "n_frequencies": self.n_frequencies,
                "is_composite": False,
            }
            self.encoding_freq = get_encoding(self.encoding.n_input_dims, config_encoding_x, self.device,)
            self.n_output_dims += self.encoding_freq.n_output_dims

    def forward(self, x, *args):
        enc = self.encoding(x, *args)
        if self.n_frequencies > 0:
            enc = torch.cat([self.encoding_freq(x, *args), enc], dim=-1)
        if self.include_xyz:
            enc = torch.cat([self.xyz_scale * x + self.xyz_offset, enc], dim=-1)
        return enc


def get_encoding(n_input_dims: int, config: Dict[str, Any], device: Union[str, int]):
    # print("get get")
    # input is assumed to be in range [0, 1]
    if config.get("otype") == 'VanillaFrequency':
        encoding = VanillaFrequency(n_input_dims, config)
    else:
        with torch.cuda.device(device):
            encoding = tcnn.Encoding(n_input_dims, config)
    if config.get("is_composite", True):
        encoding = CompositeEncoding(
            encoding,
            include_xyz=config.get('include_xyz', False),
            xyz_scale=2.,
            xyz_offset=-1.,
            n_frequencies=config.get('n_frequencies', 0),
            device=device,
        )
    return encoding