import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

act_fn_dict = {
    'softplus': torch.nn.Softplus(),
    'relu': torch.nn.ReLU(),
    'gelu': torch.nn.GELU(),
    'tanh': torch.nn.Tanh(),
}


class AppearanceDecoder(torch.nn.Module):
    def __init__(self, n_features, hidden_dim=64, act='gelu', fixed_opacity=False):
        super().__init__()
        self.hidden_dim = hidden_dim
            
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        
        if not fixed_opacity:
            self.opacity = nn.Linear(self.hidden_dim, 1)
            self.opacity_act = nn.Sigmoid()
            self.opacity_offset = 0 # used to reset
        
        self.fixed_opacity = fixed_opacity
        self.shs = nn.Linear(hidden_dim, 16*3)
    
    def reset_opacity(self, x): # reset floor to 0.5
        x = self.net(x)
        x = self.opacity(x)
        self.opacity_offset = torch.where(x > 0, 0, -x)
        
    def forward(self, x):
        x = self.net(x)
        shs = self.shs(x).reshape(-1, 16, 3)
        if not self.fixed_opacity:
            x = self.opacity(x)
            opacity = self.opacity_act(x + self.opacity_offset)
        else:
            opacity = torch.ones((x.shape[0], 1), device=x.device)
        return {'shs': shs, 'opacity': opacity}


class GeometryDecoder(torch.nn.Module):
    def __init__(self, n_features, isotropic=True, hidden_dim=128, act='gelu'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.isotropic = isotropic
        
        self.net = torch.nn.Sequential(
            nn.Linear(n_features, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
        )
        self.xyz_offsets = nn.Linear(self.hidden_dim, 3)
        if not isotropic:
            self.rotations = nn.Sequential(nn.Linear(self.hidden_dim, 6))

        # self.scales = nn.Sequential(nn.Linear(self.hidden_dim, 1 if isotropic else 3)) ~ not enough
        self.scales = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            act_fn_dict[act],
            nn.Linear(self.hidden_dim, 1 if isotropic else 3),
        )
        
    def forward(self, x):
        x = self.net(x)
        
        xyz_offsets = self.xyz_offsets(x)
        
        rotations = self.rotations(x) if not self.isotropic else None

        scales_aux = self.scales(x)
        scales = torch.log(torch.exp(scales_aux) + 1)  # torch.nn.Softplus()

        if scales_aux.shape[-1] == 1:
            scales_aux = scales_aux.repeat(1, 3)
            scales = scales.repeat(1, 3)
        
        return {
            'xyz_offsets': xyz_offsets,
            'rotations': rotations,
            'scales': scales,
            'scales_aux': scales_aux,
        }

