"""Standalone frozen DreamerV3 world model for Scheme-D offline replay."""

from __future__ import annotations
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x): return torch.sign(x)*torch.log1p(torch.abs(x))
def symexp(x): return torch.sign(x)*torch.expm1(torch.abs(x))


def make_mlp(in_dim,hidden_dim,out_dim,depth=3):
    layers=[]; width=in_dim
    for _ in range(depth): layers.extend((nn.Linear(width,hidden_dim),nn.ELU(),nn.LayerNorm(hidden_dim))); width=hidden_dim
    layers.append(nn.Linear(width,out_dim)); return nn.Sequential(*layers)


class CategoricalHead(nn.Module):
    def __init__(self,in_dim,stoch_dim,classes,unimix):
        super().__init__(); self.out=nn.Linear(in_dim,stoch_dim*classes); self.stoch_dim=stoch_dim; self.classes=classes; self.unimix=float(unimix)
    def forward(self,x): return self.out(x).reshape(*x.shape[:-1],self.stoch_dim,self.classes)
    def mixed_logits(self,logits):
        probs=torch.softmax(logits,-1); probs=(1-self.unimix)*probs+self.unimix/self.classes; return torch.log(probs.clamp_min(1e-8))


class TwoHotHead(nn.Module):
    def __init__(self,in_dim,hidden_dim,bins,low,high):
        super().__init__(); self.net=make_mlp(in_dim,hidden_dim,bins); self.register_buffer("support",torch.linspace(low,high,bins))
    def forward(self,state):
        probs=torch.softmax(self.net(state),-1); value=(probs*self.support).sum(-1,keepdim=True); return symexp(value)


class FrozenWorldModel(nn.Module):
    def __init__(self,*,obs_dim=38,action_dim=6,hidden_dim=256,deter_dim=256,stoch_dim=32,classes=32,unimix=.01,scalar_bins=255,scalar_min=-8.,scalar_max=8.):
        super().__init__(); self.obs_dim=obs_dim; self.action_dim=action_dim; self.deter_dim=deter_dim; self.stoch_dim=stoch_dim; self.classes=classes
        self.encoder=make_mlp(obs_dim,hidden_dim,hidden_dim); self.gru=nn.GRUCell(stoch_dim*classes+action_dim,deter_dim); self.prior=CategoricalHead(deter_dim,stoch_dim,classes,unimix); self.posterior=CategoricalHead(deter_dim+hidden_dim,stoch_dim,classes,unimix)
        state_dim=deter_dim+stoch_dim*classes; self.decoder=make_mlp(state_dim,hidden_dim,obs_dim); self.reward=TwoHotHead(state_dim,hidden_dim,scalar_bins,scalar_min,scalar_max); self.continue_head=make_mlp(state_dim,hidden_dim,1)
    @classmethod
    def from_checkpoint(cls,path:Path,algo_cfg,device):
        model=cls(hidden_dim=int(algo_cfg.hidden_dim),deter_dim=int(algo_cfg.deter_dim),stoch_dim=int(algo_cfg.stoch_dim),classes=int(algo_cfg.classes),unimix=float(algo_cfg.unimix),scalar_bins=int(algo_cfg.scalar_bins),scalar_min=float(algo_cfg.scalar_min),scalar_max=float(algo_cfg.scalar_max))
        state=torch.load(path,map_location="cpu",weights_only=True); prefix="world_model."; filtered={k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}; missing,unexpected=model.load_state_dict(filtered,strict=True)
        if missing or unexpected: raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
        model.to(device).eval();
        for parameter in model.parameters(): parameter.requires_grad_(False)
        return model
    def initial(self,batch,device,dtype):
        return torch.zeros(batch,self.deter_dim,device=device,dtype=dtype),torch.zeros(batch,self.stoch_dim,self.classes,device=device,dtype=dtype)
    @staticmethod
    def sample(logits,uniform,head):
        mixed=head.mixed_logits(logits); gumbel=-torch.log(-torch.log(uniform.clamp(1e-7,1-1e-7))); index=torch.argmax(mixed+gumbel,-1); return F.one_hot(index,head.classes).to(logits.dtype)
    def filter_step(self,obs,h,z,previous_action,uniform):
        h=self.gru(torch.cat((z.reshape(z.shape[0],-1),previous_action),-1),h); embed=self.encoder(symlog(obs)); logits=self.posterior(torch.cat((h,embed),-1)); z=self.sample(logits,uniform,self.posterior); return h,z,logits
    @torch.no_grad()
    def expected_imagine(self,h,z,action,uniforms):
        """Return MC prior predictions; uniforms is [B,M,S,C]."""
        batch,samples=uniforms.shape[:2]; h_next=self.gru(torch.cat((z.reshape(batch,-1),action),-1),h); prior_logits=self.prior(h_next); expanded_logits=prior_logits[:,None].expand(-1,samples,-1,-1); z_next=self.sample(expanded_logits,uniforms,self.prior); expanded_h=h_next[:,None].expand(-1,samples,-1); state=torch.cat((expanded_h,z_next.reshape(batch,samples,-1)),-1); flat=state.reshape(batch*samples,-1)
        observation=symexp(self.decoder(flat)).reshape(batch,samples,self.obs_dim); reward=self.reward(flat).reshape(batch,samples,1); continuation=torch.sigmoid(self.continue_head(flat)).reshape(batch,samples,1)
        return observation,reward,continuation,prior_logits
    def categorical_kl(self,posterior_logits,prior_logits):
        logp=self.posterior.mixed_logits(posterior_logits); logq=self.prior.mixed_logits(prior_logits); p=logp.exp(); return (p*(logp-logq)).sum(-1).mean(-1)
