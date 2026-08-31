"""Fidelity and world-model error metrics for Scheme D."""

from __future__ import annotations
import torch


def regression_report(prediction,target,names):
    rows=[]
    for i,name in enumerate(names):
        # Accumulate SSE/SST in float64.  Scheme D contains very small 16-ms
        # increments, so exposing the denominator and the RMSE/std identity is
        # useful for distinguishing a genuine resolution failure from a
        # numerical or denominator bug.
        p=prediction[:,i].double(); y=target[:,i].double(); error=y-p; target_std=y.std(unbiased=False)
        if float(target_std)<1e-8: r2=None; corr=None
        else:
            total=(y-y.mean()).square().sum(); r2=float(1-error.square().sum()/total); cov=((y-y.mean())*(p-p.mean())).mean(); corr=float(cov/(target_std*p.std(unbiased=False)).clamp_min(1e-12))
        sse=float(error.square().sum()); sst=float((y-y.mean()).square().sum()); rmse=float(error.square().mean().sqrt())
        rows.append({"index":i,"name":name,"r2":r2,"mae":float(error.abs().mean()),"rmse":rmse,"pearson":corr,"target_std":float(target_std),"sse":sse,"sst":sst,"rmse_over_target_std":None if float(target_std)<1e-8 else rmse/float(target_std),"constant_target":bool(float(target_std)<1e-8)})
    valid=[r for r in rows if r["r2"] is not None]; values=torch.tensor([r["r2"] for r in valid])
    return {"per_output":rows,"mean_r2":float(values.mean()),"median_r2":float(values.median()),"worst_output":min(valid,key=lambda r:r["r2"]),"constant_target_outputs":[r["name"] for r in rows if r["constant_target"]]}
