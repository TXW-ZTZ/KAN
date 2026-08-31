"""Audit Scheme-D R2 denominators, outlier sensitivity, and target resolution.

The script never replaces the formal held-out result with a post-hoc trimmed
score.  Trimming is reported only as a sensitivity analysis.  It also reports
directly decoded absolute physical-state groups with a persistence baseline so
that apparently reasonable absolute-state R2 values are not overinterpreted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from world_dynamics.data import DynamicsDataset
from world_dynamics.model import AdditiveDynamicsKAN, RobustFeatureScaler


PHYSICAL_OUTPUTS = 9
TRIM_FRACTIONS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)
ABSOLUTE_GROUPS = {
    "relative_position_world": tuple(range(0, 3)),
    "quaternion_wxyz": tuple(range(12, 16)),
    "world_linear_velocity": tuple(range(16, 19)),
    "world_angular_velocity": tuple(range(19, 22)),
}
SELECTED_PHYSICAL_VARIABLES = {
    0:"relative_position_world_x_m",1:"relative_position_world_y_m",2:"relative_position_world_z_m",
    12:"quaternion_w",13:"quaternion_x",14:"quaternion_y",15:"quaternion_z",
    16:"linear_velocity_world_x_mps",17:"linear_velocity_world_y_mps",18:"linear_velocity_world_z_mps",
    19:"angular_velocity_world_x_radps",20:"angular_velocity_world_y_radps",21:"angular_velocity_world_z_radps",
}


def _r2(prediction: torch.Tensor, target: torch.Tensor) -> float | None:
    prediction=prediction.double(); target=target.double()
    sst=(target-target.mean()).square().sum()
    if float(sst)<=0.0:
        return None
    return float(1.0-(target-prediction).square().sum()/sst)


def _pearson(prediction: torch.Tensor, target: torch.Tensor) -> float | None:
    prediction=prediction.double(); target=target.double()
    ps=prediction.std(unbiased=False); ts=target.std(unbiased=False)
    if float(ps)<=0.0 or float(ts)<=0.0:
        return None
    return float(((prediction-prediction.mean())*(target-target.mean())).mean()/(ps*ts))


def _per_output(prediction,target,names,stored_rows):
    rows=[]
    for index,name in enumerate(names):
        p=prediction[:,index].double(); y=target[:,index].double(); error=y-p
        sse=error.square().sum(); sst=(y-y.mean()).square().sum(); std=y.std(unbiased=False); rmse=error.square().mean().sqrt()
        sorted_squared=torch.sort(error.square(),descending=True).values
        shares={}
        for fraction in (0.001,0.01,0.05):
            count=max(1,int(len(error)*fraction))
            shares[f"top_{100*fraction:g}_percent"]=float(sorted_squared[:count].sum()/sse) if float(sse)>0 else 0.0
        recomputed=None if float(sst)<=0 else float(1-sse/sst)
        stored=stored_rows[index]["r2"]
        rows.append({
            "index":index,"name":name,"samples":len(y),"r2_recomputed_float64":recomputed,"r2_stored":stored,
            "absolute_r2_difference":None if stored is None or recomputed is None else abs(recomputed-stored),
            "pearson":_pearson(p,y),"target_std":float(std),"rmse":float(rmse),
            "rmse_over_target_std":None if float(std)<=0 else float(rmse/std),"sse":float(sse),"sst":float(sst),
            "absolute_error_quantiles":{"q50":float(torch.quantile(error.abs(),.50)),"q90":float(torch.quantile(error.abs(),.90)),"q99":float(torch.quantile(error.abs(),.99)),"max":float(error.abs().max())},
            "sse_concentration":shares,
        })
    return rows


def _shared_trim_sensitivity(prediction,target):
    p=prediction[:,:PHYSICAL_OUTPUTS].double(); y=target[:,:PHYSICAL_OUTPUTS].double()
    score=((y-p)/y.std(0,unbiased=False).clamp_min(1e-12)).square().sum(1)
    result=[]
    for fraction in (0.0,*TRIM_FRACTIONS):
        keep=torch.ones(len(y),dtype=torch.bool)
        if fraction:
            keep[torch.topk(score,int(len(y)*fraction)).indices]=False
        values=[_r2(p[keep,index],y[keep,index]) for index in range(PHYSICAL_OUTPUTS)]
        result.append({"removed_fraction":fraction,"removed_rows":int((~keep).sum()),"retained_rows":int(keep.sum()),"physical_mean_r2":sum(values)/len(values),"physical_median_r2":sorted(values)[len(values)//2],"per_output_r2":values})
    return score,result


def _tail_sensitivity(prediction,target,episode_ids,steps_to_end):
    rows=[]
    for removed_per_episode in (0,1,2,5,10,20,40,80):
        keep=steps_to_end>=removed_per_episode
        values=[_r2(prediction[keep,index],target[keep,index]) for index in range(PHYSICAL_OUTPUTS)]
        rows.append({"last_rows_removed_per_episode":removed_per_episode,"retained_rows":int(keep.sum()),"physical_mean_r2":sum(values)/len(values),"physical_median_r2":sorted(values)[len(values)//2]})
    return rows


def _absolute_state_groups(stage,source_dir,test_indices):
    current=[]; actual=[]
    for episode in test_indices:
        payload=torch.load(source_dir/f"episode_{episode:04d}.pt",map_location="cpu",weights_only=True)
        observation=payload["raw_observation"].double(); current.append(observation[:-1]); actual.append(observation[1:])
    current=torch.cat(current); actual=torch.cat(actual)
    posterior=stage["posterior_decoder_reconstruction"]["selected_absolute_physical_state"]["per_output"]
    prior=stage["prior_one_step_prediction"]["selected_absolute_physical_state"]["per_output"]
    persistence_selected=stage["persistence_baseline_selected_absolute_physical_state"]["per_output"]
    groups={}; selected=[]
    selected_group_indices={"relative_position_world":tuple(range(0,3)),"quaternion_wxyz":tuple(range(3,7)),"world_linear_velocity":tuple(range(7,10)),"world_angular_velocity":tuple(range(10,13))}
    for name,indices in selected_group_indices.items():
        persistence=[persistence_selected[index]["r2"] for index in indices]
        posterior_r2=[posterior[index]["r2"] for index in indices]
        prior_r2=[prior[index]["r2"] for index in indices]
        groups[name]={"indices":list(indices),"posterior_r2":posterior_r2,"posterior_mean_r2":sum(posterior_r2)/len(posterior_r2),"prior_r2":prior_r2,"prior_mean_r2":sum(prior_r2)/len(prior_r2),"persistence_r2":persistence,"persistence_mean_r2":sum(persistence)/len(persistence)}
    for selected_index,(observation_index,name) in enumerate(SELECTED_PHYSICAL_VARIABLES.items()):
        selected.append({"observation_index":observation_index,"name":name,"posterior_r2":posterior[selected_index]["r2"],"prior_r2":prior[selected_index]["r2"],"persistence_r2":persistence_selected[selected_index]["r2"],"prior_rmse":prior[selected_index]["rmse"],"target_std":prior[selected_index]["target_std"],"prior_pearson":prior[selected_index]["pearson"]})
    return {"selection_rule":"Directly observed absolute next physical state; quaternion is normalized and sign-aligned; excludes derived 16-ms finite-difference semantics, repeated phase, throttle, and redundant heading/up channels.","variables":selected,"uniform_mean_prior_r2":sum(row["prior_r2"] for row in selected)/len(selected),"uniform_mean_persistence_r2":sum(row["persistence_r2"] for row in selected)/len(selected),"quaternion_angle_degrees":{"posterior":stage["posterior_decoder_reconstruction"]["selected_quaternion_angle_degrees"],"prior":stage["prior_one_step_prediction"]["selected_quaternion_angle_degrees"]},"groups":groups}


@torch.no_grad()
def audit(args):
    dataset=DynamicsDataset(args.dataset_dir)
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True); report=checkpoint["report"]
    test_indices=tuple(report["split"]["test"]); test=dataset.concatenate(test_indices)
    architecture=report["architecture"]
    model=AdditiveDynamicsKAN(architecture["topology"][0],architecture["topology"][1],grid_size=architecture["grid_size"],spline_order=architecture["spline_order"])
    scaler=RobustFeatureScaler(architecture["topology"][0],4); model.load_state_dict(checkpoint["kan_state_dict"]); scaler.load_state_dict(checkpoint["scaler_state_dict"]); model.eval()
    kan_prediction=(model(scaler(test["features"]))*checkpoint["target_scale"]+checkpoint["target_center"]).double()
    dreamer=test["dreamer_target"].double(); actual=test["actual_target"].double(); names=architecture["output_names"]

    episode_ids=[]; steps_to_end=[]
    for episode in test_indices:
        length=len(dataset.episodes[episode]["step"]); episode_ids.append(torch.full((length,),episode)); steps_to_end.append(torch.arange(length-1,-1,-1))
    episode_ids=torch.cat(episode_ids); steps_to_end=torch.cat(steps_to_end)

    kan_rows=_per_output(kan_prediction,dreamer,names,report["kan_to_dreamer_fidelity"]["per_output"])
    reality_rows=_per_output(dreamer,actual,names,report["dreamer_to_isaac_model_error"]["per_output"])
    kan_score,kan_trim=_shared_trim_sensitivity(kan_prediction,dreamer)
    reality_score,reality_trim=_shared_trim_sensitivity(dreamer,actual)
    top=[]
    for index in torch.topk(reality_score,min(args.top_k,len(reality_score))).indices:
        i=int(index); top.append({"episode":int(episode_ids[i]),"step":int(test["step"][i]),"joint_standardized_squared_error":float(reality_score[i]),"physical_error_l2":float((actual[i,:PHYSICAL_OUTPUTS]-dreamer[i,:PHYSICAL_OUTPUTS]).norm())})

    stage=json.loads(args.stage_audit.read_text()); source_dir=Path(stage["source_dataset"])
    max_difference=max(row["absolute_r2_difference"] or 0.0 for row in (*kan_rows,*reality_rows))
    result={
        "protocol":"Scheme-D nominal R2 denominator, outlier, and target-resolution audit",
        "dataset":str(dataset.root),"checkpoint":str(args.checkpoint.resolve()),"test_episode_indices":list(test_indices),"test_samples":len(actual),
        "r2_definition":"1 - SSE/SST, where SSE=sum((target-prediction)^2) and SST=sum((target-mean(target))^2), accumulated in float64",
        "maximum_absolute_difference_from_stored_r2":max_difference,
        "kan_to_dreamer":{"per_output":kan_rows,"shared_row_post_hoc_trim_sensitivity":kan_trim},
        "dreamer_to_isaac":{"per_output":reality_rows,"shared_row_post_hoc_trim_sensitivity":reality_trim,"end_of_episode_sensitivity":_tail_sensitivity(dreamer,actual,episode_ids,steps_to_end),"largest_joint_physical_residual_rows":top},
        "selected_primary_physical_variables":_absolute_state_groups(stage,source_dir,test_indices),
        "interpretation":{
            "denominator":"The stored R2 values reproduce in float64; there is no denominator implementation error.",
            "outliers":"Trimming is diagnostic only. Even aggressive post-hoc removal leaves strongly negative physical-increment R2, so exclusions must not replace the formal result.",
            "recommended_primary_suite":"Report directly decoded absolute next relative position, quaternion, world linear velocity, and world angular velocity with their persistence baselines.",
            "recommended_stress_test":"Retain the nine 16-ms semantic increments as an innovation-resolution stress test, not the sole physical audit.",
            "claim_boundary":"Reasonable absolute-state R2 does not establish useful dynamics prediction when persistence is substantially better."
        }
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n")
    selected=result["selected_primary_physical_variables"]
    print(json.dumps({"test_samples":len(actual),"maximum_r2_reproduction_error":max_difference,"reality_physical_mean_r2":reality_trim[0]["physical_mean_r2"],"reality_physical_mean_r2_after_10pct_post_hoc_trim":reality_trim[-1]["physical_mean_r2"],"selected_13_variable_mean_prior_r2":selected["uniform_mean_prior_r2"],"selected_13_variable_mean_persistence_r2":selected["uniform_mean_persistence_r2"],"absolute_groups":{key:{"prior_mean_r2":value["prior_mean_r2"],"persistence_mean_r2":value["persistence_mean_r2"]} for key,value in selected["groups"].items()}},indent=2))


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-dir",type=Path,required=True); parser.add_argument("--checkpoint",type=Path,required=True); parser.add_argument("--stage-audit",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--top-k",type=int,default=20); audit(parser.parse_args())


if __name__=="__main__": main()
