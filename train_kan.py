"""Train the Scheme-D direct additive KAN against frozen Dreamer targets."""

from __future__ import annotations
import argparse,json,math,random
from copy import deepcopy
from pathlib import Path
import torch
from world_dynamics.data import DynamicsDataset,OUTPUT_NAMES,split_episode_indices
from world_dynamics.metrics import regression_report
from world_dynamics.model import AdditiveDynamicsKAN,RobustFeatureScaler


def seed_all(seed):
    random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def pair_indices(lengths,device):
    before=[]; after=[]; offset=0
    for length in lengths:
        if length>1: before.append(torch.arange(offset,offset+length-1,device=device)); after.append(torch.arange(offset+1,offset+length,device=device))
        offset+=length
    return torch.cat(before),torch.cat(after)


@torch.no_grad()
def predict(model,scaler,data,center,scale,device): return (model(scaler(data["features"].to(device)))*scale+center).cpu()


def train(args):
    seed_all(args.seed); dataset=DynamicsDataset(args.dataset_dir); split=split_episode_indices(len(dataset.episodes),args.split_seed); train_data=dataset.concatenate(split.train); val=dataset.concatenate(split.validation); test=dataset.concatenate(split.test); device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device)
    scaler=RobustFeatureScaler(train_data["features"].shape[1],args.scaler_clip).to(device).fit(train_data["features"].to(device)); center=train_data["dreamer_target"].mean(0).to(device); scale=train_data["dreamer_target"].std(0).clamp_min(1e-3).to(device); model=AdditiveDynamicsKAN(45,11,grid_size=args.grid_size,spline_order=args.spline_order,grid_range=(-args.scaler_clip,args.scaler_clip)).to(device)
    x=scaler(train_data["features"].to(device)); y=(train_data["dreamer_target"].to(device)-center)/scale; before,after=pair_indices(dataset.lengths(split.train),device); optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=0); scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode="max",factor=.5,patience=8,min_lr=1e-6)
    best=deepcopy(model.state_dict()); best_score=-math.inf; best_epoch=0; stale=0; history=[]
    for epoch in range(1,args.epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True); p=model(x); mse=torch.nn.functional.mse_loss(p,y); temporal=torch.nn.functional.mse_loss(p[after]-p[before],y[after]-y[before]); loss=mse+args.temporal_weight*temporal
        if args.curvature_weight: loss=loss+args.curvature_weight*model.curvature_loss()
        loss.backward(); optimizer.step()
        if epoch==1 or epoch%args.validation_interval==0:
            val_prediction=predict(model,scaler,val,center,scale,device); val_report=regression_report(val_prediction,val["dreamer_target"],OUTPUT_NAMES); score=val_report["median_r2"]; scheduler.step(score); history.append({"epoch":epoch,"loss":float(loss),"mse":float(mse),"temporal_mse":float(temporal),"val_median_r2":score}); print(f"epoch={epoch:04d} loss={float(loss):.6f} val_median_r2={score:.5f}",flush=True)
            if score>best_score+args.min_delta: best,best_score,best_epoch,stale=deepcopy(model.state_dict()),score,epoch,0
            else: stale+=args.validation_interval
            if stale>=args.patience: break
    model.load_state_dict(best); test_prediction=predict(model,scaler,test,center,scale,device); fidelity=regression_report(test_prediction,test["dreamer_target"],OUTPUT_NAMES); model_error=regression_report(test["dreamer_target"],test["actual_target"],OUTPUT_NAMES); mc_se=(test["dreamer_variance"]/dataset.manifest["mc_samples"]).mean(0).sqrt()
    report={"role":"offline additive Scheme-D frozen-world-model explainer","dataset":str(dataset.root),"split":split.as_dict(),"sample_counts":{"train":len(train_data["features"]),"validation":len(val["features"]),"test":len(test["features"])},"architecture":{"topology":[45,11],"feature_names":dataset.manifest["feature_names"],"output_names":list(OUTPUT_NAMES),"grid_size":args.grid_size,"spline_order":args.spline_order},"target_scaling":{"center":center.cpu().tolist(),"scale":scale.cpu().tolist()},"optimization":{"epochs_run":epoch,"best_epoch":best_epoch,"best_validation_median_r2":best_score},"kan_to_dreamer_fidelity":fidelity,"dreamer_to_isaac_model_error":model_error,"dreamer_mc_standard_error_rms":mc_se.tolist(),"mean_test_prior_posterior_kl":float(test["prior_posterior_kl"].mean()),"history":history,"claim_boundary":"KAN fidelity explains the frozen Dreamer model. Dreamer-to-Isaac error separately limits any physical-dynamics interpretation."}
    args.output_dir.mkdir(parents=True,exist_ok=False); torch.save({"kan_state_dict":model.state_dict(),"scaler_state_dict":scaler.state_dict(),"target_center":center.cpu(),"target_scale":scale.cpu(),"report":report},args.output_dir/"best_kan.pt"); (args.output_dir/"report.json").write_text(json.dumps(report,indent=2)+"\n"); print("test_r2",[round(r["r2"],4) for r in fidelity["per_output"]]); return report


def main():
    p=argparse.ArgumentParser(); p.add_argument("dataset_dir",type=Path); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); p.add_argument("--seed",type=int,default=20260827); p.add_argument("--split-seed",type=int,default=20260827); p.add_argument("--grid-size",type=int,default=8); p.add_argument("--spline-order",type=int,default=3); p.add_argument("--scaler-clip",type=float,default=4); p.add_argument("--learning-rate",type=float,default=.003); p.add_argument("--epochs",type=int,default=1800); p.add_argument("--patience",type=int,default=320); p.add_argument("--validation-interval",type=int,default=20); p.add_argument("--min-delta",type=float,default=1e-4); p.add_argument("--temporal-weight",type=float,default=.08); p.add_argument("--curvature-weight",type=float,default=0); train(p.parse_args())
if __name__=="__main__": main()
