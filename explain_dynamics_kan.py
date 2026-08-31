"""Create a figure4paper-style Scheme-D dynamics atlas."""

from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches,path as mpath
from matplotlib.colors import Normalize
import numpy as np
import torch

from world_dynamics.data import DynamicsDataset,OUTPUT_NAMES,split_episode_indices
from world_dynamics.metrics import regression_report
from world_dynamics.model import AdditiveDynamicsKAN,RobustFeatureScaler


def as_numpy(value):
    """Convert tensors without Tensor.numpy(), which is unavailable with NumPy 2 here."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return np.asarray(value)


BG="#FFFFFF"; PANEL="#F4F7FA"; GRID="#D6DCE3"; TEXT="#20252B"; MUTED="#6B7280"
CYAN="#2A7F9E"; BLUE="#0F4D92"; GREEN="#55A868"; YELLOW="#C89422"; ORANGE="#D18436"; RED="#B64342"; PURPLE="#8064A2"
NEON=(BLUE,"#3775BA",GREEN,YELLOW,ORANGE,RED,PURPLE,"#45A1C9","#B279A2","#2A7F9E","#8C6D31")
GROUPS=(("position state",range(0,3)),("velocity state",range(3,6)),("reference accel",range(6,9)),("angular state",range(9,12)),("attitude + phase",range(12,15)),("current action",range(15,21)),("action delta",range(21,27)),("position memory",range(27,33)),("velocity memory",range(33,36)),("angular memory",range(36,39)),("action memory",range(39,45)))
SHORT_OUTPUT=("Δpₓ","Δpᵧ","Δp_z","Δvₓ","Δvᵧ","Δv_z","Δωₓ","Δωᵧ","Δω_z","reward","risk")


def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"figure.facecolor":BG,"axes.facecolor":BG,"savefig.facecolor":BG,"text.color":TEXT,"axes.labelcolor":TEXT,"axes.edgecolor":"#4B5563","axes.linewidth":1.2,"axes.spines.top":False,"axes.spines.right":False,"xtick.color":TEXT,"ytick.color":TEXT,"grid.color":GRID,"grid.linewidth":.7,"legend.frameon":False,"pdf.fonttype":42,"ps.fonttype":42,"svg.fonttype":"none"})


def save(fig,folder,stem):
    fig.savefig(folder/f"{stem}.pdf",dpi=300,bbox_inches="tight",facecolor=BG); fig.savefig(folder/f"{stem}.png",dpi=220,bbox_inches="tight",facecolor=BG); plt.close(fig)


def short_feature(name):
    return name.replace("state_","").replace("memory_","").replace("position_error_body_","pos err ").replace("velocity_error_body_","vel err ").replace("angular_velocity_body_","ang vel ").replace("past_ema_0.25s_","EMA ").replace("action_command_","act ").replace("action_delta_from_previous_throttle_","Δact ").replace("persistent_","persist ").replace("reference_accel_body_","ref acc ").replace("_mps2"," [m/s²]").replace("_radps"," [rad/s]").replace("_mps"," [m/s]").replace("_m"," [m]").replace("_", " ")


def load_model(path,device):
    payload=torch.load(path,map_location="cpu",weights_only=True); report=payload["report"]; model=AdditiveDynamicsKAN(45,11,grid_size=report["architecture"]["grid_size"],spline_order=report["architecture"]["spline_order"]).to(device); scaler=RobustFeatureScaler(45,4).to(device); model.load_state_dict(payload["kan_state_dict"]); scaler.load_state_dict(payload["scaler_state_dict"]); model.eval(); return payload,model,scaler,payload["target_center"].to(device),payload["target_scale"].to(device)


@torch.no_grad()
def bundle(model,scaler,center,scale,data,device):
    z=scaler(data["features"].to(device)); edges=(model.edge_contributions(z)*scale[None,:,None]).cpu(); pred=(model(z)*scale+center).cpu(); return z.cpu(),edges,pred


@torch.no_grad()
def curves(model,scaler,scale,train_x,train_edges,importance,names,device):
    rows=[[None]*45 for _ in range(11)]; center=train_edges.mean(0)
    for o in range(11):
        for f in range(45):
            lo,hi=torch.quantile(train_x[:,f],torch.tensor((.01,.99))); grid=torch.linspace(lo,hi,161); z=torch.zeros(len(grid),45,device=device); z[:,f]=((grid.to(device)-scaler.center[f])/scaler.scale[f]).clamp(-4,4); curve=(model.edge_contributions(z)[:,o,f]*scale[o]).cpu()-center[o,f]; rows[o][f]={"grid":as_numpy(grid),"curve":as_numpy(curve),"importance":float(importance[o,f]),"feature":f,"feature_name":names[f],"q01":float(lo),"q99":float(hi)}
    return rows,center


def neon_pipeline(folder):
    fig,ax=plt.subplots(figsize=(16,7)); ax.set_xlim(0,16); ax.set_ylim(0,7); ax.axis("off")
    def box(x,y,w,h,title,body,color):
        ax.add_patch(patches.FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.06",facecolor=PANEL,edgecolor=color,lw=1.6))
        ax.text(x+.18,y+h-.35,title,color=color,weight="bold",fontsize=11); ax.text(x+.18,y+h-.75,body,va="top",fontsize=8.2,linespacing=1.35,color=TEXT)
    box(.35,3.9,2.35,1.75,"POSTERIOR STATE","sₜ = [hₜ,zₜ]\nrecorded action aₜ\nfrozen checkpoint",CYAN); box(3.35,3.9,2.35,1.75,"DREAMER PRIOR","p(zₜ₊₁|hₜ₊₁)\nMC16 prior samples\ndecoder · reward · continue",PURPLE); box(6.35,3.9,2.35,1.75,"TEACHER TARGET","E[Δp,Δv,Δω]\nE[reward, risk]\n+ predictive variance",YELLOW); box(9.35,3.9,2.35,1.75,"45-D EVIDENCE","state · action · Δaction\nEMA · persistence\ninnovation",GREEN); box(12.35,3.9,2.7,1.75,"DYNAMICS-KAN","11 × 45 splines\nDreamer fidelity only\nnever controls BlueROV",ORANGE)
    for x in (2.7,5.7,8.7,11.7): ax.annotate("",xy=(x+.65,4.78),xytext=(x,4.78),arrowprops=dict(arrowstyle="-|>",lw=1.7,color=MUTED))
    ax.plot([.7,15.1],[2.7,2.7],color=RED,ls="--",lw=1.2); ax.text(7.9,2.35,"REALITY FIREWALL",ha="center",color=RED,weight="bold",fontsize=10)
    ax.text(7.9,1.82,"KAN error = KAN − Dreamer     ·     world-model error = Dreamer − Isaac",ha="center",fontsize=12,color=TEXT)
    ax.text(7.9,1.25,"Only the first measures explainer fidelity. Only a small second error would permit a physical-dynamics reading.",ha="center",fontsize=8.8,color=MUTED)
    ax.text(8,6.55,"SCHEME D // FROZEN WORLD-MODEL DYNAMICS EXPLAINER",ha="center",fontsize=16,weight="bold",color=CYAN)
    save(fig,folder,"D01_neon_world_model_pipeline")


def error_triangle(folder,fidelity,model_error,mc_se,target_std):
    fig,ax=plt.subplots(figsize=(10,8)); vertices=np.array([[0,0],[1,0],[.5,np.sqrt(3)/2]]); ax.plot(*np.vstack((vertices,vertices[0])).T,color=MUTED,lw=1.4)
    points=[]
    for i in range(11):
        f=np.clip((fidelity[i]["r2"]+0.2)/1.2,0,1); r=model_error[i]["r2"]; reality=0 if r is None else np.clip((r+1)/2,0,1); certainty=np.clip(1-mc_se[i]/max(target_std[i],1e-8),0,1); w=np.array([f,reality,certainty])+.03; w=w/w.sum(); point=w[0]*vertices[0]+w[1]*vertices[1]+w[2]*vertices[2]; points.append(point); ax.scatter(*point,s=180,color=NEON[i],edgecolor="white",lw=.8); ax.text(*point,str(i),fontsize=6.5,color="white",weight="bold",ha="center",va="center")
    ax.text(-.04,-.04,"KAN fidelity",color=CYAN,weight="bold",ha="right"); ax.text(1.04,-.04,"Isaac validity",color=GREEN,weight="bold"); ax.text(.5,.91,"posterior certainty",color=PURPLE,weight="bold",ha="center")
    for i,label in enumerate(SHORT_OUTPUT): ax.text(1.03,.78-i*.052,f"[{i:02d}]  {label}",transform=ax.transAxes,color=NEON[i],fontsize=8,weight="bold")
    ax.set_xlim(-.18,1.18); ax.set_ylim(-.12,1); ax.axis("off"); ax.set_title("Three-way evidence simplex: explanation, reality and stochastic certainty",fontsize=15,weight="bold",color=TEXT); save(fig,folder,"D02_three_way_evidence_simplex")
    return np.array(points)


def output_constellation(folder,fidelity,model_error,mc_se,target_std):
    theta=np.linspace(0,2*np.pi,11,endpoint=False); width=2*np.pi/13; fig=plt.figure(figsize=(11,9)); ax=fig.add_subplot(111,polar=True); r2=np.array([max(-.2,r["r2"]) for r in fidelity]); reality=np.array([0 if r["r2"] is None else max(-1,r["r2"]) for r in model_error]); uncertainty=1-np.array(mc_se)/np.maximum(target_std,1e-8)
    ax.bar(theta,np.clip((r2+.2)/1.2,0,1),width=width,color=NEON,alpha=.75,label="KAN→Dreamer R²"); ax.scatter(theta,np.clip((reality+1)/2,0,1),s=70,color="white",marker="x",label="Dreamer→Isaac validity"); ax.scatter(theta,np.clip(uncertainty,0,1),s=55,color=YELLOW,edgecolor=BG,label="posterior certainty"); ax.set_xticks(theta,SHORT_OUTPUT); ax.set_ylim(0,1.05); ax.grid(alpha=.35); ax.set_title("Dynamics-output constellation",pad=24,fontsize=15,weight="bold"); ax.legend(loc="upper right",bbox_to_anchor=(1.30,1.15)); save(fig,folder,"D03_output_constellation")


def group_glyphs(folder,importance):
    matrix=torch.stack([importance[:,list(idx)].sum(-1) for _,idx in GROUPS],1); matrix=matrix/matrix.sum(1,keepdim=True).clamp_min(1e-12); fig,ax=plt.subplots(figsize=(14,7));
    for o in range(11):
        for g in range(len(GROUPS)): ax.scatter(g,o,s=40+1400*float(matrix[o,g]),color=NEON[o],alpha=.25+.7*float(matrix[o,g]),edgecolor="white",lw=.3)
    ax.set_xticks(range(len(GROUPS)),[g[0] for g in GROUPS],rotation=35,ha="right"); ax.set_yticks(range(11),SHORT_OUTPUT); ax.invert_yaxis(); ax.grid(alpha=.25); ax.set_title("Influence glyph matrix · bubble area = standardized spline effect",fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,"D04_influence_glyph_matrix"); return matrix


def reality_gap(folder,fidelity,model_error):
    fig,ax=plt.subplots(figsize=(13,6.5)); y=np.arange(11); k=np.array([r["rmse"] for r in fidelity]); m=np.array([r["rmse"] for r in model_error]); ax.hlines(y,k,m,color=GRID,lw=3); ax.scatter(k,y,s=90,color=CYAN,label="KAN − Dreamer RMSE"); ax.scatter(m,y,s=90,color=RED,marker="D",label="Dreamer − Isaac RMSE");
    for i in range(11): ax.text(max(k[i],m[i])+max(m)*.025,i,f"×{m[i]/max(k[i],1e-9):.1f}",va="center",fontsize=7,color=MUTED)
    ax.set_yticks(y,SHORT_OUTPUT); ax.invert_yaxis(); ax.set_xscale("log"); ax.set_xlabel("RMSE (native output units, logarithmic)"); ax.legend(); ax.grid(axis="x",alpha=.3); ax.set_title("Reality gap slopegraph · explaining Dreamer is not validating Dreamer",fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,"D05_world_model_reality_gap")


def circular_chord(folder,matrix):
    fig,ax=plt.subplots(figsize=(11,11)); ax.set_aspect("equal"); ax.axis("off"); ng=len(GROUPS); no=11; left_angles=np.linspace(np.pi*.62,np.pi*1.38,ng); right_angles=np.linspace(-np.pi*.38,np.pi*.38,no); left=np.c_[np.cos(left_angles),np.sin(left_angles)]; right=np.c_[np.cos(right_angles),np.sin(right_angles)]; vmax=float(matrix.max())
    Path=mpath.Path
    top=[]
    for o in range(no):
        for g in range(ng): top.append((float(matrix[o,g]),o,g))
    for value,o,g in sorted(top,reverse=True)[:65]:
        p0=left[g]; p3=right[o]; verts=[p0,p0*.55,p3*.55,p3]; path=Path(verts,[Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4]); ax.add_patch(patches.PathPatch(path,facecolor="none",edgecolor=NEON[o],lw=.3+5*value/vmax,alpha=.10+.65*value/vmax))
    for g,p in enumerate(left): ax.scatter(*p,s=130,color=GREEN,edgecolor="white"); ax.text(*(p*1.13),GROUPS[g][0],ha="right",va="center",fontsize=7.5,color=GREEN)
    for o,p in enumerate(right): ax.scatter(*p,s=150,color=NEON[o],edgecolor="white"); ax.text(*(p*1.13),SHORT_OUTPUT[o],ha="left",va="center",fontsize=8,color=NEON[o],weight="bold")
    ax.text(0,1.19,"45-D evidence families",ha="center",color=GREEN,weight="bold"); ax.text(0,-1.18,"line width = normalized additive effect",ha="center",color=MUTED,fontsize=8); ax.set_title("Circular influence chord · state/action/history → Dreamer outputs",fontsize=15,weight="bold",pad=20); save(fig,folder,"D06_circular_influence_chord")


def family_orbits(folder,matrix):
    fig=plt.figure(figsize=(14,8)); axes=[fig.add_subplot(2,3,i+1,polar=True) for i in range(6)]; families=(("position",range(0,3)),("velocity",range(3,6)),("angular",range(6,9)),("reward",[9]),("risk",[10]),("all outputs",range(11)))
    angles=np.linspace(0,2*np.pi,len(GROUPS),endpoint=False); width=2*np.pi/(len(GROUPS)+2)
    for ax,(label,outs) in zip(axes,families):
        values=as_numpy(matrix[list(outs)].mean(0)); ax.bar(angles,values,width=width,color=NEON[:len(GROUPS)],alpha=.8); ax.set_xticks(angles,[g[0] for g in GROUPS],fontsize=6); ax.set_yticks([]); ax.set_title(label,color=TEXT,weight="bold")
    fig.suptitle("Input-family influence orbits",fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,"D07_input_family_orbits")


def spline_gallery(folder,rows,importance,names,outputs,stem,title):
    selected=[]
    for o in outputs:
        for f in torch.argsort(importance[o],descending=True)[:4].tolist(): selected.append((o,f))
    n=len(selected); cols=4; rowsn=math.ceil(n/cols); fig,axes=plt.subplots(rowsn,cols,figsize=(15,3.2*rowsn)); axes=np.atleast_1d(axes).ravel()
    for ax,(o,f) in zip(axes,selected):
        row=rows[o][f]; ax.plot(row["grid"],row["curve"],color=NEON[o],lw=2); ax.fill_between(row["grid"],row["curve"],0,color=NEON[o],alpha=.13); ax.axhline(0,color=MUTED,ls="--",lw=.6); ax.set_title(f"{SHORT_OUTPUT[o]} ← {short_feature(names[f])}",fontsize=7.5,color=NEON[o]); ax.tick_params(labelsize=6)
    for ax in axes[n:]: ax.axis("off")
    fig.suptitle(title,fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,stem)


def fit_symbolic(x,y):
    candidates=[]
    for degree,name in ((1,"linear"),(2,"quadratic"),(3,"cubic")):
        c=np.polyfit(x,y,degree); fit=np.polyval(c,x); total=max(np.square(y-y.mean()).sum(),1e-12); r2=1-np.square(y-fit).sum()/total; candidates.append((r2,degree+1,name,c,fit))
    for direction in ("above","below"):
        for theta in np.linspace(np.quantile(x,.15),np.quantile(x,.85),61):
            h=np.maximum(0,x-theta) if direction=="above" else np.maximum(0,theta-x); A=np.c_[np.ones_like(x),h]; c=np.linalg.lstsq(A,y,rcond=None)[0]; fit=A@c; r2=1-np.square(y-fit).sum()/max(np.square(y-y.mean()).sum(),1e-12); candidates.append((r2,3,"hinge_"+direction,np.array([c[0],c[1],theta]),fit))
    best=max(candidates,key=lambda x:x[0]); chosen=min([x for x in candidates if x[0]>=best[0]-.006],key=lambda x:(x[1],-x[0])); r2,_,kind,c,fit=chosen
    if kind=="linear": formula=f"{c[0]:+.3g}x {c[1]:+.3g}"
    elif kind=="quadratic": formula=f"{c[0]:+.3g}x² {c[1]:+.3g}x {c[2]:+.3g}"
    elif kind=="cubic": formula=f"{c[0]:+.3g}x³ {c[1]:+.3g}x² {c[2]:+.3g}x {c[3]:+.3g}"
    else: formula=f"{c[0]:+.3g} {c[1]:+.3g}·hinge(x; θ={c[2]:+.3g}, {kind[6:]})"
    return {"family":kind,"parameters":c.tolist(),"formula":formula,"r2":float(r2)}


def symbolic_outputs(folder,output,rows,importance,names,baseline,fidelity):
    fits=[[None]*45 for _ in range(11)]
    for o in range(11):
        for f in range(45): fits[o][f]=fit_symbolic(np.asarray(rows[o][f]["grid"]),np.asarray(rows[o][f]["curve"]))
    matrix=np.array([[fits[o][f]["r2"] for f in range(45)] for o in range(11)]); fig,ax=plt.subplots(figsize=(15,7)); im=ax.imshow(matrix,aspect="auto",cmap="plasma",vmin=0,vmax=1); ax.set_xticks(range(45),[short_feature(n) for n in names],rotation=90,fontsize=5); ax.set_yticks(range(11),SHORT_OUTPUT); fig.colorbar(im,ax=ax,pad=.01,label="compact symbolic edge R²"); ax.set_title("Symbolic compressibility scan · 495 exact splines",fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,"D13_symbolic_compressibility_scan")
    cards=[]; fig,axes=plt.subplots(4,3,figsize=(16,15)); axes=axes.ravel()
    for o,ax in enumerate(axes):
        if o>=11: ax.axis("off"); continue
        ax.axis("off"); ax.add_patch(patches.FancyBboxPatch((.01,.02),.98,.96,boxstyle="round,pad=.02",facecolor=PANEL,edgecolor=NEON[o],lw=1.5)); ax.text(.05,.88,f"[{o:02d}] {SHORT_OUTPUT[o]}",color=NEON[o],weight="bold",fontsize=14,transform=ax.transAxes); ax.text(.05,.79,f"bias {float(baseline[o]):+.4g} · KAN R² {fidelity[o]['r2']:.3f}",fontsize=7.5,color=MUTED,transform=ax.transAxes)
        order=torch.argsort(importance[o],descending=True).tolist(); selected=sorted(order[:12],key=lambda f:(-fits[o][f]["r2"],-float(importance[o,f])))[:3]; terms=[]
        for rank,f in enumerate(selected): fit=fits[o][f]; y=.62-rank*.20; ax.text(.05,y,short_feature(names[f]),fontsize=7.5,weight="bold",transform=ax.transAxes); ax.text(.05,y-.08,"ψ ≈ "+fit["formula"],fontsize=6.5,family="monospace",transform=ax.transAxes); ax.text(.80,y-.08,f"R² {fit['r2']:.3f}",fontsize=6.5,color=YELLOW,transform=ax.transAxes); terms.append({"feature":f,"feature_name":names[f],**fit})
        ax.text(.05,.04,"+ exact remainder from 42 edges",color=RED,fontsize=6.8,transform=ax.transAxes); cards.append({"output":o,"output_name":OUTPUT_NAMES[o],"terms":terms})
    fig.suptitle("SYMBOLIC LAW TERMINAL // support-limited approximations",fontsize=15,weight="bold",color=CYAN); fig.tight_layout(); save(fig,folder,"D14_symbolic_law_terminal")
    lines=["Scheme D symbolic laws","="*78,"Exact KAN: y_hat_j = b_j + sum_i psi_j,i(x_i)","These formulas approximate individual edges only inside empirical support.",""]
    for card in cards:
        lines.append(f"{card['output_name']}")
        for t in card["terms"]: lines.append(f"  {t['feature_name']}: psi(x) ~= {t['formula']}; edge_R2={t['r2']:.6f}")
        lines.append("  + exact remainder from 42 edges\n")
    lines.append("Boundary: these are laws of the frozen Dreamer surrogate, not certified hydrodynamic equations.")
    (output/"symbolic_dreamer_dynamics_laws.txt").write_text("\n".join(lines)+"\n")
    return fits,cards


def rules_wheel(folder,output,rows,importance,names):
    rules=[]
    for o in range(11):
        best=None
        for f in torch.argsort(importance[o],descending=True)[:12].tolist():
            x=np.asarray(rows[o][f]["grid"]); y=np.asarray(rows[o][f]["curve"])
            for direction in ("above","below"):
                for theta in np.linspace(np.quantile(x,.15),np.quantile(x,.85),81):
                    h=np.maximum(0,x-theta) if direction=="above" else np.maximum(0,theta-x); A=np.c_[np.ones_like(x),h]; c=np.linalg.lstsq(A,y,rcond=None)[0]; fit=A@c; r2=1-np.square(y-fit).sum()/max(np.square(y-y.mean()).sum(),1e-12); candidate={"output":o,"feature":f,"feature_name":names[f],"threshold":float(theta),"direction":direction,"intercept":float(c[0]),"slope":float(c[1]),"r2":float(r2),"importance":float(importance[o,f])}
                    if best is None or (candidate["r2"]+.05*candidate["importance"])>(best["r2"]+.05*best["importance"]): best=candidate
        rules.append(best)
    theta=np.linspace(0,2*np.pi,11,endpoint=False); fig=plt.figure(figsize=(11,9)); ax=fig.add_subplot(111,polar=True); radii=np.array([r["r2"] for r in rules]); sizes=200+1600*np.array([r["importance"] for r in rules]); colors=[CYAN if r["slope"]>=0 else RED for r in rules]; ax.scatter(theta,radii,s=sizes,c=colors,alpha=.75,edgecolor="white"); ax.set_xticks(theta,[f"{SHORT_OUTPUT[r['output']]}\n{short_feature(r['feature_name'])}" for r in rules],fontsize=6.5); ax.set_ylim(0,1.05); ax.set_title("Rule rune wheel · radius=hinge R² · size=edge effect · color=direction",pad=25,fontsize=14,weight="bold"); save(fig,folder,"D15_rule_rune_wheel")
    text=["Scheme D one-edge IF-THEN rules","="*78]
    for i,r in enumerate(rules,1): comp=">" if r["direction"]=="above" else "<"; change="increases" if r["slope"]>0 else "decreases"; text.extend([f"R{i:02d} {OUTPUT_NAMES[r['output']]} <- {r['feature_name']}",f"  IF {r['feature_name']} {comp} {r['threshold']:.9g}",f"  THEN the Dreamer-KAN edge contribution {change}; hinge slope={r['slope']:+.9g}",f"  edge_R2={r['r2']:.6f}; importance={r['importance']:.6f}",""])
    text.append("Not a physical law: every rule approximates one KAN edge explaining Dreamer, not Isaac Sim."); (output/"if_then_dreamer_dynamics_rules.txt").write_text("\n".join(text)+"\n"); return rules


def internal_lawbook(folder,rows,importance,names,fidelity):
    choices=((3,3,"surge-error damping"),(4,4,"sway-error damping"),(5,5,"heave-error damping"),(6,12,"roll restoring cue"),(7,13,"pitch restoring cue"),(9,15,"reward vs thruster-0"))
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,(o,f,label) in zip(axes.flat,choices): row=rows[o][f]; ax.plot(row["grid"],row["curve"],color=NEON[o],lw=2.5); ax.fill_between(row["grid"],row["curve"],0,color=NEON[o],alpha=.15); ax.axhline(0,color=MUTED,ls="--",lw=.7); ax.set_xlabel(short_feature(names[f])); ax.set_ylabel(SHORT_OUTPUT[o]+" contribution"); ax.set_title(f"{label}\nKAN fidelity R²={fidelity[o]['r2']:.3f}",fontsize=9,weight="bold")
    fig.suptitle("Dreamer-internal dynamics lawbook · diagnostic, not hydrodynamic certification",fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,"D16_dreamer_internal_lawbook")


def response_surfaces(folder,rows,importance,names):
    outs=(0,3,4,5,6,7); fig,axes=plt.subplots(2,3,figsize=(15,8.5)); surfaces=[]
    for o in outs:
        fs=torch.argsort(importance[o],descending=True)[:2].tolist(); rx,ry=rows[o][fs[0]],rows[o][fs[1]]; x,y=np.asarray(rx["grid"]),np.asarray(ry["grid"]); surface=np.asarray(ry["curve"])[:,None]+np.asarray(rx["curve"])[None,:]; surfaces.append((o,fs,x,y,surface))
    vmax=max(np.max(np.abs(s[-1])) for s in surfaces)
    for ax,(o,fs,x,y,surface) in zip(axes.flat,surfaces): mesh=ax.pcolormesh(x,y,surface,cmap="coolwarm",vmin=-vmax,vmax=vmax,shading="auto"); ax.contour(x,y,surface,colors=TEXT,alpha=.35,linewidths=.4,levels=9); ax.set_xlabel(short_feature(names[fs[0]]),fontsize=7); ax.set_ylabel(short_feature(names[fs[1]]),fontsize=7); ax.set_title(SHORT_OUTPUT[o],color=NEON[o],weight="bold")
    fig.subplots_adjust(right=.88,hspace=.32,wspace=.28); cax=fig.add_axes((.91,.18,.015,.62)); fig.colorbar(mesh,cax=cax,label="two-edge Dreamer response"); fig.suptitle("Additive state-action response manifolds",fontsize=15,weight="bold"); save(fig,folder,"D17_state_action_response_manifolds")


def diagnostics(folder,test,pred,names):
    kan_error=(pred-test["dreamer_target"]).square().mean(1).sqrt(); model_error=(test["dreamer_target"]-test["actual_target"]).square().mean(1).sqrt(); uncertainty=(test["dreamer_variance"].mean(1)/16).sqrt(); kl=test["prior_posterior_kl"].squeeze(-1)
    fig,axes=plt.subplots(1,2,figsize=(13,5.2)); hb=axes[0].hexbin(as_numpy(kl),as_numpy(kan_error),gridsize=45,cmap="viridis",bins="log",mincnt=1); fig.colorbar(hb,ax=axes[0],label="log count"); axes[0].set_xlabel("prior–posterior KL"); axes[0].set_ylabel("KAN–Dreamer per-step RMSE"); axes[0].set_title("Innovation versus explainer error",weight="bold"); hb=axes[1].hexbin(as_numpy(kl),as_numpy(model_error),gridsize=45,cmap="magma",bins="log",mincnt=1); fig.colorbar(hb,ax=axes[1],label="log count"); axes[1].set_xlabel("prior–posterior KL"); axes[1].set_ylabel("Dreamer–Isaac per-step RMSE"); axes[1].set_title("Innovation versus reality gap",weight="bold"); fig.tight_layout(); save(fig,folder,"D18_innovation_error_hexbin")
    fig,ax=plt.subplots(figsize=(10,6)); hb=ax.hexbin(as_numpy(uncertainty),as_numpy(model_error),C=as_numpy(kl),reduce_C_function=np.mean,gridsize=48,cmap="plasma",mincnt=1); fig.colorbar(hb,ax=ax,label="mean prior–posterior KL"); ax.set_xlabel("Dreamer MC standard error (all outputs)"); ax.set_ylabel("Dreamer–Isaac per-step RMSE"); ax.set_title("Does posterior uncertainty warn about the world-model reality gap?",fontsize=14,weight="bold"); save(fig,folder,"D19_uncertainty_reality_gap")
    return kan_error,model_error,uncertainty,kl


def episode_map(folder,dataset,test_indices,model,scaler,center,scale,device):
    rows=[]
    for epi in test_indices:
        ep=dataset.episodes[epi]
        with torch.no_grad(): pred=(model(scaler(ep["features"].to(device)))*scale+center).cpu()
        rows.append((epi,float((pred-ep["dreamer_target"]).square().mean().sqrt()),float((ep["dreamer_target"]-ep["actual_target"]).square().mean().sqrt()),float(ep["prior_posterior_kl"].mean()),len(ep["step"])))
    fig,ax=plt.subplots(figsize=(10,6)); sc=ax.scatter([r[1] for r in rows],[r[2] for r in rows],s=[35+r[4]*.35 for r in rows],c=[r[3] for r in rows],cmap="plasma",edgecolor="white",lw=.5); fig.colorbar(sc,ax=ax,label="mean prior–posterior KL");
    for r in rows: ax.text(r[1],r[2],str(r[0]),fontsize=6,color=TEXT)
    ax.set_xlabel("episode KAN–Dreamer RMSE"); ax.set_ylabel("episode Dreamer–Isaac RMSE"); ax.set_title("Held-out episode dual-error map · bubble size = trajectory length",fontsize=14,weight="bold"); save(fig,folder,"D20_episode_dual_error_map"); return rows


def timeline_and_spectrum(folder,dataset,episode_rows,model,scaler,center,scale,device):
    epi=sorted(episode_rows,key=lambda r:r[2])[len(episode_rows)//2][0]; ep=dataset.episodes[epi]; steps=as_numpy(ep["step"].squeeze(-1));
    with torch.no_grad(): pred=(model(scaler(ep["features"].to(device)))*scale+center).cpu()
    outputs=(2,5,9); fig,axes=plt.subplots(3,1,figsize=(15,9),sharex=True)
    for ax,o in zip(axes,outputs): ax.plot(steps,as_numpy(ep["actual_target"][:,o]),color=GREEN,lw=1,label="Isaac actual"); ax.plot(steps,as_numpy(ep["dreamer_target"][:,o]),color=PURPLE,lw=1.3,label="Dreamer MC16"); ax.plot(steps,as_numpy(pred[:,o]),color=CYAN,lw=1.5,label="Dynamics-KAN"); ax.fill_between(steps,as_numpy(ep["dreamer_target"][:,o]-2*(ep["dreamer_variance"][:,o]/16).sqrt()),as_numpy(ep["dreamer_target"][:,o]+2*(ep["dreamer_variance"][:,o]/16).sqrt()),color=PURPLE,alpha=.12); ax.set_ylabel(SHORT_OUTPUT[o]); ax.legend(ncol=3,fontsize=7)
    axes[-1].set_xlabel("recorded step"); fig.suptitle(f"Triple-model oscilloscope // held-out episode {epi}",fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,"D21_triple_model_oscilloscope")
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    for ax,o in zip(axes,outputs):
        e1=as_numpy(pred[:,o]-ep["dreamer_target"][:,o]); e2=as_numpy(ep["dreamer_target"][:,o]-ep["actual_target"][:,o]); freq=np.fft.rfftfreq(len(e1),d=.016); p1=np.abs(np.fft.rfft(e1-e1.mean()))**2; p2=np.abs(np.fft.rfft(e2-e2.mean()))**2; mask=freq<=8; ax.loglog(freq[mask][1:],p1[mask][1:]+1e-12,color=CYAN,label="KAN−Dreamer"); ax.loglog(freq[mask][1:],p2[mask][1:]+1e-12,color=RED,label="Dreamer−Isaac"); ax.set_xlabel("frequency [Hz]"); ax.set_ylabel("error power"); ax.set_title(SHORT_OUTPUT[o]); ax.legend(fontsize=7)
    fig.suptitle("Error-spectrum observatory",fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,"D22_error_spectrum_observatory"); return epi


def embedding(folder,test,kan_error,model_error,kl):
    x=test["features"]; z=(x-x.mean(0))/x.std(0).clamp_min(1e-6); _,_,v=torch.pca_lowrank(z,q=2,center=False); pc=as_numpy(z@v[:,:2]); rng=np.random.default_rng(20260827); take=rng.choice(len(pc),min(5500,len(pc)),replace=False); fig,axes=plt.subplots(1,3,figsize=(16,5))
    for ax,value,title,cmap in zip(axes,(kl,kan_error,model_error),("prior–posterior KL","KAN–Dreamer error","Dreamer–Isaac error"),("plasma","viridis","magma")):
        sc=ax.scatter(pc[take,0],pc[take,1],c=as_numpy(value)[take],s=7,cmap=cmap,alpha=.55,rasterized=True); fig.colorbar(sc,ax=ax,pad=.01); ax.set_title(title,weight="bold"); ax.set_xlabel("dynamics PC1"); ax.set_ylabel("dynamics PC2")
    fig.suptitle("Dynamics-state observatory · the same held-out manifold under three diagnostics",fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,"D23_dynamics_state_observatory")


def local_sunburst(folder,test,pred,edges,names):
    centered=edges-edges.mean(0); error=(pred-test["dreamer_target"]).square().mean(1); sample=int(torch.argsort(error)[len(error)//2]); output=int(torch.argmax(centered[sample].abs().sum(-1))); local=centered[sample,output]; theta=np.linspace(0,2*np.pi,45,endpoint=False); values=as_numpy(local.abs()); colors=[]
    for f in range(45): colors.append(NEON[next(g for g,(_,idx) in enumerate(GROUPS) if f in idx)])
    fig=plt.figure(figsize=(13,9)); ax=fig.add_axes((.04,.08,.68,.80),polar=True); ax.bar(theta,values,width=2*np.pi/49,bottom=.18,color=colors,alpha=.82,edgecolor="white",lw=.25); ax.scatter(theta,[.13]*45,c=[CYAN if float(v)>=0 else RED for v in local],s=24); top=torch.argsort(local.abs(),descending=True)[:8]
    for rank,f in enumerate(top,1): ax.text(theta[int(f)],.22+values[int(f)],str(rank),fontsize=6.5,color=TEXT,weight="bold",ha="center",va="center")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=.45); ax.set_title(f"Local contribution sunburst · {SHORT_OUTPUT[output]}\nouter length=|edge| · inner color sign",pad=20,fontsize=14,weight="bold")
    card=fig.add_axes((.75,.16,.23,.63)); card.axis("off"); card.add_patch(patches.FancyBboxPatch((0,0),1,1,boxstyle="round,pad=.025",facecolor=PANEL,edgecolor=GRID,lw=1.2)); card.text(.08,.93,"Top local spline edges",weight="bold",fontsize=10)
    for rank,f in enumerate(top,1):
        value=float(local[int(f)]); sign="+" if value>=0 else "−"; card.text(.08,.84-(rank-1)*.10,f"[{rank}] {short_feature(names[int(f)])}",fontsize=7.5,weight="bold",color=colors[int(f)]); card.text(.12,.795-(rank-1)*.10,f"{sign} |ψ| = {abs(value):.3g}",fontsize=7,color=TEXT)
    save(fig,folder,"D24_local_contribution_sunburst")


def ridgelines_and_hexbin(folder,test,pred,fidelity):
    errors=pred-test["dreamer_target"]; fig,ax=plt.subplots(figsize=(12,8)); xgrid=np.linspace(-4,4,260)
    for o in range(11):
        values=as_numpy(errors[:,o]/test["dreamer_target"][:,o].std().clamp_min(1e-8)); hist,bins=np.histogram(np.clip(values,-4,4),bins=100,range=(-4,4),density=True); centers=(bins[:-1]+bins[1:])/2; hist=hist/max(hist.max(),1e-9)*.75; ax.fill_between(centers,o,o+hist,color=NEON[o],alpha=.55); ax.plot(centers,o+hist,color=NEON[o],lw=1)
    ax.set_yticks(range(11),SHORT_OUTPUT); ax.set_xlabel("standardized KAN−Dreamer error"); ax.set_title("Explainer-error ridgelines",fontsize=15,weight="bold"); save(fig,folder,"D25_explainer_error_ridgelines")
    fig,axes=plt.subplots(4,3,figsize=(13,14)); axes=axes.ravel()
    for o,ax in enumerate(axes):
        if o>=11: ax.axis("off"); continue
        hb=ax.hexbin(as_numpy(test["dreamer_target"][:,o]),as_numpy(pred[:,o]),gridsize=38,cmap="viridis",bins="log",mincnt=1); lo=min(float(test["dreamer_target"][:,o].min()),float(pred[:,o].min())); hi=max(float(test["dreamer_target"][:,o].max()),float(pred[:,o].max())); ax.plot([lo,hi],[lo,hi],color=RED,ls="--",lw=.8); ax.set_title(f"{SHORT_OUTPUT[o]} · R² {fidelity[o]['r2']:.3f}",fontsize=8,color=NEON[o]); ax.tick_params(labelsize=6)
    fig.suptitle("KAN→Dreamer fidelity density atlas",fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,"D26_fidelity_density_atlas")


def action_ribbons(folder,rows,names):
    fig,axes=plt.subplots(3,2,figsize=(14,11)); outputs=(0,3,4,5,9,10)
    for ax,o in zip(axes.flat,outputs):
        for cmd in range(6): f=15+cmd; row=rows[o][f]; curve=np.asarray(row["curve"]); curve=curve-curve[len(curve)//2]; ax.plot(row["grid"],curve,color=NEON[cmd],lw=1.8,label=f"a{cmd}"); ax.fill_between(row["grid"],curve,0,color=NEON[cmd],alpha=.035)
        ax.axhline(0,color=MUTED,ls="--",lw=.6); ax.set_xlabel("action command"); ax.set_ylabel("counterfactual edge Δ"); ax.set_title(SHORT_OUTPUT[o],color=NEON[o],weight="bold")
    axes[0,0].legend(ncol=3,fontsize=7); fig.suptitle("Thruster-action response ribbons inside the Dreamer surrogate",fontsize=15,weight="bold"); fig.tight_layout(); save(fig,folder,"D27_thruster_response_ribbons")


def budget_and_dashboard(folder,report,fidelity,model_error,mc_se,target_std):
    fig,ax=plt.subplots(figsize=(14,7)); x=np.arange(11); kan=np.array([r["rmse"] for r in fidelity])/np.maximum(target_std,1e-8); mc=np.array(mc_se)/np.maximum(target_std,1e-8); real=np.array([r["rmse"] for r in model_error])/np.maximum(target_std,1e-8); ax.bar(x,kan,color=CYAN,label="KAN→Dreamer normalized RMSE"); ax.bar(x,mc,bottom=kan,color=PURPLE,label="Dreamer MC mean SE"); ax.bar(x,real,bottom=kan+mc,color=RED,alpha=.75,label="Dreamer→Isaac normalized RMSE"); ax.set_yscale("log"); ax.set_xticks(x,SHORT_OUTPUT); ax.set_ylabel("stacked diagnostic scale / Dreamer target std"); ax.legend(ncol=3); ax.set_title("Three-layer error budget · logarithmic scale",fontsize=15,weight="bold"); save(fig,folder,"D28_three_layer_error_budget")
    fig=plt.figure(figsize=(16,9)); gs=fig.add_gridspec(2,4,hspace=.38,wspace=.35); cards=(("33,032","one-step transitions",CYAN),("MC16","Dreamer prior paths",PURPLE),(f"{np.mean([r['r2'] for r in fidelity[:9]]):.3f}","mean physical KAN R²",YELLOW),(f"{fidelity[9]['r2']:.3f}","reward KAN R²",GREEN))
    for i,(value,label,color) in enumerate(cards): ax=fig.add_subplot(gs[0,i]); ax.axis("off"); ax.add_patch(patches.FancyBboxPatch((.02,.12),.96,.76,boxstyle="round,pad=.04",facecolor=PANEL,edgecolor=color,lw=2)); ax.text(.5,.60,value,ha="center",fontsize=24,weight="bold",color=color); ax.text(.5,.33,label,ha="center",fontsize=8.5,color=TEXT)
    ax=fig.add_subplot(gs[1,:2]); r2=[r["r2"] for r in fidelity]; ax.bar(range(11),r2,color=NEON); ax.axhline(0,color=MUTED,lw=.8); ax.set_xticks(range(11),SHORT_OUTPUT); ax.set_ylabel("KAN→Dreamer R²"); ax.set_title("Explanation fidelity",weight="bold")
    ax=fig.add_subplot(gs[1,2:]); physical=np.array([r["rmse"] for r in model_error[:9]]); ax.bar(range(9),physical,color=RED,alpha=.8); ax.set_xticks(range(9),SHORT_OUTPUT[:9]); ax.set_ylabel("Dreamer→Isaac RMSE"); ax.set_title("Physical reality gap",weight="bold")
    fig.suptitle("SCHEME D // WORLD-MODEL AUDIT DASHBOARD",fontsize=17,weight="bold",color=CYAN); fig.text(.5,.02,"Reward is modeled well; physical one-step semantics show a substantial reality gap. KAN curves therefore explain Dreamer, not certified water physics.",ha="center",fontsize=9,color=MUTED); save(fig,folder,"D29_world_model_audit_dashboard")


def build(args):
    style(); device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available() else "cpu" if args.device=="auto" else args.device); dataset=DynamicsDataset(args.dataset_dir); split=split_episode_indices(len(dataset.episodes)); train=dataset.concatenate(split.train); test=dataset.concatenate(split.test); payload,model,scaler,center,scale=load_model(args.checkpoint,device); z,train_edges,train_pred=bundle(model,scaler,center,scale,train,device); test_z,test_edges,test_pred=bundle(model,scaler,center,scale,test,device); names=dataset.manifest["feature_names"]; fidelity=payload["report"]["kan_to_dreamer_fidelity"]["per_output"]; model_error=payload["report"]["dreamer_to_isaac_model_error"]["per_output"]; mc_se=np.asarray(payload["report"]["dreamer_mc_standard_error_rms"]); target_std=np.asarray([r["target_std"] for r in fidelity]); centered=train_edges-train_edges.mean(0); importance=centered.square().mean(0).sqrt()/train["dreamer_target"].std(0).clamp_min(1e-8)[:,None]; rows,edge_center=curves(model,scaler,scale,train["features"],train_edges,importance,names,device); baseline=center.cpu()+model.output_bias.detach().cpu()*scale.cpu()+edge_center.sum(-1)
    output=args.output_dir.resolve(); folder=output/"figures"; folder.mkdir(parents=True,exist_ok=True); neon_pipeline(folder); simplex=error_triangle(folder,fidelity,model_error,mc_se,target_std); output_constellation(folder,fidelity,model_error,mc_se,target_std); matrix=group_glyphs(folder,importance); reality_gap(folder,fidelity,model_error); circular_chord(folder,matrix); family_orbits(folder,matrix); spline_gallery(folder,rows,importance,names,(0,1,2),"D08_position_spline_gallery","Dreamer position-increment spline gallery"); spline_gallery(folder,rows,importance,names,(3,4,5),"D09_velocity_spline_gallery","Dreamer velocity-increment spline gallery"); spline_gallery(folder,rows,importance,names,(6,7,8),"D10_angular_spline_gallery","Dreamer angular-increment spline gallery"); spline_gallery(folder,rows,importance,names,(9,),"D11_reward_spline_gallery","Dreamer reward spline gallery"); spline_gallery(folder,rows,importance,names,(10,),"D12_risk_spline_gallery","Dreamer termination-risk spline gallery"); fits,cards=symbolic_outputs(folder,output,rows,importance,names,baseline,fidelity); rules=rules_wheel(folder,output,rows,importance,names); internal_lawbook(folder,rows,importance,names,fidelity); response_surfaces(folder,rows,importance,names); kan_error,reality,uncertainty,kl=diagnostics(folder,test,test_pred,names); episode_rows=episode_map(folder,dataset,split.test,model,scaler,center,scale,device); representative=timeline_and_spectrum(folder,dataset,episode_rows,model,scaler,center,scale,device); embedding(folder,test,kan_error,reality,kl); local_sunburst(folder,test,test_pred,test_edges,names); ridgelines_and_hexbin(folder,test,test_pred,fidelity); action_ribbons(folder,rows,names); budget_and_dashboard(folder,payload["report"],fidelity,model_error,mc_se,target_std)
    pdfs=sorted(folder.glob("*.pdf")); pngs=sorted(folder.glob("*.png")); expected=29
    with torch.no_grad(): exact=(test_pred-(center.cpu()+model.output_bias.cpu()*scale.cpu()+test_edges.sum(-1))).abs().max().item()
    if len(pdfs)!=expected or len(pngs)!=expected or exact>1e-5: raise RuntimeError(f"atlas audit failed pdf={len(pdfs)} png={len(pngs)} exact={exact}")
    report={"protocol":"Scheme D frozen Dreamer dynamics KAN audit","figure_style":"figure4paper publication style","figures":len(pdfs),"pdfs":[p.name for p in pdfs],"dataset":str(dataset.root),"checkpoint":str(args.checkpoint.resolve()),"split":split.as_dict(),"simplex_coordinates":simplex.tolist(),"group_importance_fraction":matrix.tolist(),"symbolic_cards":cards,"if_then_rules":rules,"representative_episode":representative,"max_exact_edge_reconstruction_error":exact,"claim_boundary":"All KAN laws explain frozen Dreamer predictions. Dreamer-to-Isaac errors are reported separately and prohibit unqualified physical-law claims."}; (output/"dynamics_atlas.json").write_text(json.dumps(report,indent=2)+"\n"); (output/"figure_index.txt").write_text("Scheme D figure4paper dynamics atlas\n"+"="*78+"\n\n"+"\n".join(p.name for p in pdfs)+"\n\nText artifacts:\n  symbolic_dreamer_dynamics_laws.txt\n  if_then_dreamer_dynamics_rules.txt\n")
    with (output/"output_metrics.csv").open("w",newline="") as handle:
        writer=csv.writer(handle); writer.writerow(("output","kan_to_dreamer_r2","kan_to_dreamer_rmse","dreamer_to_isaac_r2","dreamer_to_isaac_rmse","dreamer_mc_se_rms"));
        for i in range(11): writer.writerow((OUTPUT_NAMES[i],fidelity[i]["r2"],fidelity[i]["rmse"],model_error[i]["r2"],model_error[i]["rmse"],mc_se[i]))
    print(f"output={output}"); print(f"pdf_png_pairs={len(pdfs)}"); print(f"max_exact_edge_reconstruction_error={exact:.3e}")


def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset-dir",type=Path,required=True); p.add_argument("--checkpoint",type=Path,required=True); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--device",choices=("auto","cpu","cuda"),default="auto"); build(p.parse_args())
if __name__=="__main__": main()
