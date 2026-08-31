"""Direct additive 45-to-11 KAN used only for Scheme-D teacher dynamics."""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class RobustFeatureScaler(nn.Module):
    def __init__(self,feature_dim:int,clip:float=4.):
        super().__init__(); self.feature_dim=feature_dim; self.clip=float(clip)
        self.register_buffer("center",torch.zeros(feature_dim)); self.register_buffer("scale",torch.ones(feature_dim)); self.register_buffer("fitted",torch.tensor(False))
    @torch.no_grad()
    def fit(self,x):
        q25,median,q75=torch.quantile(x.float(),torch.tensor((.25,.5,.75),device=x.device),dim=0); self.center.copy_(median); self.scale.copy_(((q75-q25)/1.349).clamp_min(1e-6)); self.fitted.fill_(True); return self
    def forward(self,x):
        if not bool(self.fitted): raise RuntimeError("scaler is not fitted")
        return ((x-self.center)/self.scale).clamp(-self.clip,self.clip)
    def inverse(self,z): return z*self.scale+self.center


class AdditiveDynamicsKAN(nn.Module):
    def __init__(self,input_dim:int,output_dim:int=11,*,grid_size:int=8,spline_order:int=3,grid_range=(-4.,4.)):
        super().__init__(); self.input_dim=input_dim; self.output_dim=output_dim; self.grid_size=grid_size; self.spline_order=spline_order
        step=(grid_range[1]-grid_range[0])/grid_size; knots=torch.arange(-spline_order,grid_size+spline_order+1).float()*step+grid_range[0]
        self.register_buffer("grid",knots.expand(input_dim,-1).contiguous()); self.base_weight=nn.Parameter(torch.empty(output_dim,input_dim)); self.spline_weight=nn.Parameter(torch.empty(output_dim,input_dim,grid_size+spline_order)); self.spline_scale=nn.Parameter(torch.ones(output_dim,input_dim)); self.output_bias=nn.Parameter(torch.zeros(output_dim))
        nn.init.kaiming_uniform_(self.base_weight,a=math.sqrt(5)); nn.init.normal_(self.spline_weight,0,.02/grid_size)
    def b_splines(self,x):
        expanded=x.unsqueeze(-1); basis=((expanded>=self.grid[:,:-1])&(expanded<self.grid[:,1:])).to(x.dtype)
        for order in range(1,self.spline_order+1):
            left_den=self.grid[:,order:-1]-self.grid[:,:-(order+1)]; right_den=self.grid[:,order+1:]-self.grid[:,1:-order]
            basis=(expanded-self.grid[:,:-(order+1)])/left_den*basis[...,:-1]+(self.grid[:,order+1:]-expanded)/right_den*basis[...,1:]
        return basis.contiguous()
    @property
    def scaled_spline_weight(self): return self.spline_weight*self.spline_scale.unsqueeze(-1)
    def edge_contributions(self,x):
        spline=torch.einsum("...ik,oik->...oi",self.b_splines(x),self.scaled_spline_weight); base=torch.nn.functional.silu(x).unsqueeze(-2)*self.base_weight; return base+spline
    def forward(self,x): return self.edge_contributions(x).sum(-1)+self.output_bias
    def curvature_loss(self):
        grid=torch.linspace(-4,4,129,device=self.grid.device); x=grid[:,None].expand(-1,self.input_dim); curves=self.edge_contributions(x); return (curves[2:]-2*curves[1:-1]+curves[:-2]).square().mean()
