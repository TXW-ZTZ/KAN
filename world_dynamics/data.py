"""Episode-safe Scheme-D dynamics dataset loader."""

from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import torch
from .features import DYNAMICS_FEATURE_NAMES


OUTPUT_NAMES=(
    "delta_position_error_body_x_m","delta_position_error_body_y_m","delta_position_error_body_z_m",
    "delta_velocity_error_body_x_mps","delta_velocity_error_body_y_mps","delta_velocity_error_body_z_mps",
    "delta_angular_velocity_body_x_radps","delta_angular_velocity_body_y_radps","delta_angular_velocity_body_z_radps",
    "reward","termination_risk",
)


@dataclass(frozen=True)
class EpisodeSplit:
    train:tuple[int,...]; validation:tuple[int,...]; test:tuple[int,...]
    def as_dict(self): return {"train":list(self.train),"validation":list(self.validation),"test":list(self.test)}


def split_episode_indices(count:int,seed:int=20260827):
    g=torch.Generator().manual_seed(seed); order=torch.randperm(count,generator=g).tolist(); ntrain=round(.70*count); nval=round(.15*count); return EpisodeSplit(tuple(order[:ntrain]),tuple(order[ntrain:ntrain+nval]),tuple(order[ntrain+nval:]))


class DynamicsDataset:
    def __init__(self,root):
        self.root=Path(root).resolve(); self.manifest=json.loads((self.root/"manifest.json").read_text())
        if tuple(self.manifest["feature_names"])!=DYNAMICS_FEATURE_NAMES or tuple(self.manifest["output_names"])!=OUTPUT_NAMES: raise ValueError("Scheme-D schema mismatch")
        self.episodes=[torch.load(self.root/item["path"],map_location="cpu",weights_only=True) for item in self.manifest["episode_files"]]
    def concatenate(self,indices):
        keys=("features","dreamer_target","dreamer_variance","actual_target","prior_posterior_kl","step")
        result={k:torch.cat([self.episodes[i][k] for i in indices],0).float() for k in keys}; result["episode_id"]=torch.cat([torch.full((len(self.episodes[i]["step"]),1),i,dtype=torch.float32) for i in indices]); return result
    def lengths(self,indices): return [len(self.episodes[i]["step"]) for i in indices]
