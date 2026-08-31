"""Standalone Track/BlueROV observation-to-semantics transform."""

from __future__ import annotations
import torch


SEMANTIC_NAMES = (
    "position_error_body_x_m",
    "position_error_body_y_m",
    "position_error_body_z_m",
    "velocity_error_body_x_mps",
    "velocity_error_body_y_mps",
    "velocity_error_body_z_mps",
    "reference_accel_body_x_mps2",
    "reference_accel_body_y_mps2",
    "reference_accel_body_z_mps2",
    "angular_velocity_body_x_radps",
    "angular_velocity_body_y_radps",
    "angular_velocity_body_z_radps",
    "gravity_direction_body_x",
    "gravity_direction_body_y",
    "previous_throttle_0",
    "previous_throttle_1",
    "previous_throttle_2",
    "previous_throttle_3",
    "previous_throttle_4",
    "previous_throttle_5",
    "episode_phase",
)


def _quat_rotate_inverse(q:torch.Tensor,v:torch.Tensor)->torch.Tensor:
    q=q/q.norm(dim=-1,keepdim=True).clamp_min(1e-8); qw=q[...,:1]; qv=q[...,1:]
    return v*(2*qw.square()-1)-2*qw*torch.cross(qv,v,dim=-1)+2*qv*(qv*v).sum(-1,keepdim=True)


def semanticize(observation:torch.Tensor,*,dt:float=.016,future_step_stride:int=5)->torch.Tensor:
    if observation.shape[-1]!=38: raise ValueError("Track/BlueROV observation must have 38 channels")
    targets_world=observation[...,0:12].reshape(*observation.shape[:-1],4,3); q=observation[...,12:16]
    expanded_q=q.unsqueeze(-2).expand(*targets_world.shape[:-1],4); targets=_quat_rotate_inverse(expanded_q,targets_world)
    velocity=observation[...,16:22]; linear=_quat_rotate_inverse(q,velocity[...,:3]); angular=_quat_rotate_inverse(q,velocity[...,3:])
    h=float(dt*future_step_stride); p0,p1,p2,p3=targets.unbind(-2)
    reference_velocity=(-11*p0+18*p1-9*p2+2*p3)/(6*h); reference_accel=(2*p0-5*p1+4*p2-p3)/(h*h)
    up=torch.zeros_like(linear); up[...,2]=1; gravity=_quat_rotate_inverse(q,up)
    previous_throttle=(observation[...,28:34]+1)*.5; phase=observation[...,34:35]
    return torch.cat((p0,reference_velocity-linear,reference_accel,angular,gravity[...,:2],previous_throttle,phase),-1)
