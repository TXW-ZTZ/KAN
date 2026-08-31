"""Extract Scheme-D Dreamer one-step dynamics targets from recorded episodes."""

from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path
import torch
from omegaconf import OmegaConf

from world_dynamics.data import OUTPUT_NAMES
from world_dynamics.features import DYNAMICS_FEATURE_NAMES,build_dynamics_features
from world_dynamics.frozen_world_model import FrozenWorldModel
from world_dynamics.semantics import semanticize


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b""): digest.update(chunk)
    return digest.hexdigest()


def assemble_outputs(next_semantic,current_semantic,reward,risk):
    delta=next_semantic-current_semantic
    return torch.cat((delta[...,0:3],delta[...,3:6],delta[...,9:12],reward,risk),-1)


@torch.no_grad()
def extract_episode(model,payload,*,episode_index,mc_samples,prior_seed,chunk_size,device):
    obs=payload["raw_observation"].float().to(device); semantics=payload["semantics"].float(); actions=payload["teacher_action_operational"].float(); stored_uniform=payload["posterior_uniform"].float().to(device); steps=len(obs)
    h,z=model.initial(1,device,obs.dtype); previous=torch.zeros(1,6,device=device,dtype=obs.dtype); hs=[]; zs=[]; posterior_logits=[]
    for t in range(steps):
        h,z,logits=model.filter_step(obs[t:t+1],h,z,previous,stored_uniform[t:t+1]); hs.append(h.squeeze(0).cpu()); zs.append(z.squeeze(0).cpu()); posterior_logits.append(logits.squeeze(0).cpu()); previous=actions[t:t+1].to(device)
    hs=torch.stack(hs); zs=torch.stack(zs); posterior_logits=torch.stack(posterior_logits)
    generator=torch.Generator(device="cpu").manual_seed(prior_seed+episode_index); predicted=[]; variances=[]; kls=[]
    for start in range(0,steps-1,chunk_size):
        stop=min(steps-1,start+chunk_size); count=stop-start; uniforms=torch.rand((count,mc_samples,model.stoch_dim,model.classes),generator=generator).to(device)
        observation,reward,continuation,prior_logits=model.expected_imagine(hs[start:stop].to(device),zs[start:stop].to(device),actions[start:stop].to(device),uniforms)
        predicted_semantic=semanticize(observation); current=semantics[start:stop].to(device)[:,None,:].expand(-1,mc_samples,-1); outputs=assemble_outputs(predicted_semantic,current,reward,1-continuation)
        predicted.append(outputs.mean(1).cpu()); variances.append(outputs.var(1,unbiased=False).cpu()); kls.append(model.categorical_kl(posterior_logits[start+1:stop+1].to(device),prior_logits).cpu().unsqueeze(-1))
    actual_next=semantics[1:]; actual=assemble_outputs(actual_next,semantics[:-1],payload["reward"][:-1].float(),payload["terminated"][:-1].float())
    features=build_dynamics_features(semantics,actions)[:-1]
    return {"features":features,"dreamer_target":torch.cat(predicted),"dreamer_variance":torch.cat(variances),"actual_target":actual,"prior_posterior_kl":torch.cat(kls),"step":payload["step"][:-1].int()}


def main():
    p=argparse.ArgumentParser(); p.add_argument("--source-dir",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--teacher-config",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--mc-samples",type=int,default=16); p.add_argument("--prior-seed",type=int,default=2026082703); p.add_argument("--chunk-size",type=int,default=128); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); args=p.parse_args()
    source=args.source_dir.resolve(); source_manifest=json.loads((source/"manifest.json").read_text()); checkpoint_hash=sha256(args.checkpoint)
    if source_manifest["teacher_checkpoint_sha256"]!=checkpoint_hash: raise RuntimeError("source/checkpoint SHA256 mismatch")
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device); cfg=OmegaConf.load(args.teacher_config); model=FrozenWorldModel.from_checkpoint(args.checkpoint,cfg.algo,device)
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False); manifest={"schema_version":1,"protocol":"Scheme-D frozen Dreamer prior MC one-step dynamics versus recorded Isaac outcome","created_utc":datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),"scope":{"task":"Track","robot":"BlueROV","trajectory":"lemniscate"},"source_dataset":str(source),"source_manifest_sha256":sha256(source/"manifest.json"),"teacher_checkpoint":str(args.checkpoint.resolve()),"teacher_checkpoint_sha256":checkpoint_hash,"teacher_config_sha256":sha256(args.teacher_config),"mc_samples":args.mc_samples,"prior_seed":args.prior_seed,"feature_names":list(DYNAMICS_FEATURE_NAMES),"output_names":list(OUTPUT_NAMES),"environment_executed":False,"target_definition":"MC expectation under Dreamer prior after posterior state s_t and recorded action a_t","episode_files":[]}
    for episode_index,item in enumerate(source_manifest["episode_files"]):
        payload=torch.load(source/item["path"],map_location="cpu",weights_only=True); result=extract_episode(model,payload,episode_index=episode_index,mc_samples=args.mc_samples,prior_seed=args.prior_seed,chunk_size=args.chunk_size,device=device); path=output/f"episode_{episode_index:04d}.pt"; torch.save(result,path)
        teacher_error=(result["dreamer_target"]-result["actual_target"]); manifest["episode_files"].append({"episode":episode_index,"path":path.name,"steps":len(result["step"]),"sha256":sha256(path),"dreamer_vs_isaac_rmse":float(teacher_error.square().mean().sqrt()),"mean_prior_posterior_kl":float(result["prior_posterior_kl"].mean())}); print(f"episode={episode_index+1}/{len(source_manifest['episode_files'])} steps={len(result['step'])} model_rmse={float(teacher_error.square().mean().sqrt()):.4f}",flush=True)
    manifest["episodes"]=len(manifest["episode_files"]); manifest["samples"]=sum(x["steps"] for x in manifest["episode_files"]); (output/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n"); print(output)


if __name__=="__main__": main()
