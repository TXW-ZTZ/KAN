"""Causal, physically named inputs for the Scheme-D dynamics explainer."""

from __future__ import annotations

import math
import torch


AXES=("x","y","z")
DYNAMICS_FEATURE_NAMES=(
    *(f"state_position_error_body_{a}_m" for a in AXES),
    *(f"state_velocity_error_body_{a}_mps" for a in AXES),
    *(f"state_reference_accel_body_{a}_mps2" for a in AXES),
    *(f"state_angular_velocity_body_{a}_radps" for a in AXES),
    "state_gravity_direction_body_x","state_gravity_direction_body_y","state_episode_phase",
    *(f"action_command_{i}" for i in range(6)),
    *(f"action_delta_from_previous_throttle_{i}" for i in range(6)),
    *(f"memory_past_ema_0.25s_position_error_{a}_m" for a in AXES),
    *(f"memory_persistent_position_error_{a}_m" for a in AXES),
    *(f"memory_velocity_innovation_{a}_mps" for a in AXES),
    *(f"memory_past_ema_0.25s_angular_velocity_{a}_radps" for a in AXES),
    *(f"memory_past_ema_0.25s_action_command_{i}" for i in range(6)),
)


def _past_ema(values:torch.Tensor,dt:float,tau:float)->torch.Tensor:
    decay=math.exp(-dt/tau); state=torch.zeros_like(values[0]); rows=[]
    for current in values:
        rows.append(state); state=decay*state+(1-decay)*current
    return torch.stack(rows)


def build_dynamics_features(semantics:torch.Tensor,actions:torch.Tensor,*,dt:float=.016)->torch.Tensor:
    if semantics.ndim!=2 or semantics.shape[1]!=21: raise ValueError("semantics must be [T,21]")
    if actions.shape!=(semantics.shape[0],6): raise ValueError("actions must be [T,6]")
    position=semantics[:,0:3]; velocity=semantics[:,3:6]; angular=semantics[:,9:12]
    pos_short=_past_ema(position,dt,.25); pos_long=_past_ema(position,dt,1.0)
    vel_short=_past_ema(velocity,dt,.25); ang_short=_past_ema(angular,dt,.25); action_short=_past_ema(actions,dt,.25)
    current=torch.cat((semantics[:,0:14],semantics[:,20:21]),-1)
    action_delta=actions-semantics[:,14:20]
    result=torch.cat((current,actions,action_delta,pos_short,pos_long-position,velocity-vel_short,ang_short,action_short),-1)
    if result.shape[1]!=len(DYNAMICS_FEATURE_NAMES): raise RuntimeError(f"feature schema mismatch: {result.shape}")
    return result
