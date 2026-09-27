#!/usr/bin/env python3
"""Rebuild presentation figures and CSVs from the published W10 v11 aggregate.

Reads only the four committed closeout/aggregate manifests. No model is loaded.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, PercentFormatter

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "results/learned/w10"
GRID = [-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,9,11,13,15,18]
COL = {"learned":"#0072B2", "learned_snr_randomised":"#009E73",
       "learned_papr_constrained":"#56B4E9", "classical_adaptive":"#D55E00",
       "er9_digital":"#CC79A7", "classical_fixed_mcs":"#7F3C8D",
       "classical_fixed_mod":"#A6761D", "classical_jpeg_secondary":"#E69F00",
       "label_transmission_bound":"#555555", "semantic_recon_ablation":"#648FFF"}
NAME = {"learned":"Learned DJSCC", "learned_snr_randomised":"SNR-randomized DJSCC",
        "learned_papr_constrained":"PAPR-constrained DJSCC", "classical_adaptive":"Adaptive classical",
        "er9_digital":"Task-aware digital (ER-9)", "classical_fixed_mcs":"Fixed MCS",
        "classical_fixed_mod":"Fixed modulation", "classical_jpeg_secondary":"JPEG secondary",
        "label_transmission_bound":"Predicted-label control", "semantic_recon_ablation":"Reconstruction ablation"}

def read(name):
    return json.loads((SRC/name).read_bytes())

def sha(name):
    return hashlib.sha256((SRC/name).read_bytes()).hexdigest()

def verify():
    close=read("w10_continuation_closeout_v11.json")
    pub=read("w10_continuation_units_v11.json")
    um=read("w10_continuation_unit_manifest_v11.json")
    pm=read("w10_continuation_per_image_manifest_v11.json")
    assert close["status"]=="COMPLETE" and close["test"]=="SEALED" and close["test_access"]==0
    assert close["unit_count"]==pub["unit_count"]==um["unit_count"]==252
    assert close["stream_count"]==pm["stream_count"]==357
    assert close["unit_manifest_sha256"]==sha("w10_continuation_unit_manifest_v11.json")
    assert close["per_image_manifest_sha256"]==sha("w10_continuation_per_image_manifest_v11.json")
    units=pub["units"]
    assert len(units)==252 and len(pm["streams"])==357
    by_arm=defaultdict(list)
    for i,u in enumerate(units):
        assert u["ordinal"]==i==um["units"][i]["ordinal"]
        assert u["unit_id"]==um["units"][i]["unit_id"]
        assert u["split"]=="val" and u["test_access"]==0 and u["n_total"]==u["denominator"]==1000
        assert u["n_correct"]==u["scorer_variants"][0]["n_correct"]
        assert u["coverage_rate"]==(1000-u["decode_failure_count"]-u["infeasible_count"])/1000
        assert u["per_image_sha256"]==um["units"][i]["per_image_sha256"]
        for v in u["scorer_variants"]:
            assert v["n_total"]==1000
        by_arm[(u["system"],u["bw_ratio"])].append(u)
    assert len(by_arm)==12
    for v in by_arm.values(): assert [x["snr_db"] for x in v]==GRID
    return close,units,by_arm

def write_csvs(units):
    d=OUT/"data"; d.mkdir(exist_ok=True)
    cols=["ordinal","system","role","bw_ratio","snr_db","primary_scorer","n_total","n_correct",
          "accuracy_end_to_end","n_delivered","coverage_rate","decode_failure_count","infeasible_count",
          "acc_given_delivery","mean_papr_db","max_papr_db","papr_measured_count","papr_cap_db",
          "papr_cap_compliant","papr_domain","unit_id","source_epoch"]
    with (d/"w10_primary_252.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,cols);w.writeheader()
        for u in units:
            w.writerow({"ordinal":u["ordinal"],"system":u["system"],"role":u["role"],"bw_ratio":u["bw_ratio"],
               "snr_db":u["snr_db"],"primary_scorer":u["scorer_variants"][0]["classifier_variant"],
               "n_total":u["n_total"],"n_correct":u["n_correct"],"accuracy_end_to_end":u["n_correct"]/u["n_total"],
               "n_delivered":u["n_total"]-u["decode_failure_count"]-u["infeasible_count"],
               "coverage_rate":u["coverage_rate"],"decode_failure_count":u["decode_failure_count"],
               "infeasible_count":u["infeasible_count"],"acc_given_delivery":u["acc_given_delivery"],
               "mean_papr_db":u["mean_papr_db"],"max_papr_db":u["max_papr_db"],
               "papr_measured_count":u["papr_measured_count"],"papr_cap_db":u["papr_cap_db"],
               "papr_cap_compliant":u["papr_cap_compliant"],"papr_domain":u["papr_domain"],
               "unit_id":u["unit_id"],"source_epoch":("v8" if u["ordinal"]<u_epoch[0] else "v9" if u["ordinal"]<u_epoch[1] else "v11")})
    cols2=["ordinal","system","bw_ratio","snr_db","classifier_variant","n_correct","n_total","accuracy_end_to_end","per_image_sha256"]
    with (d/"w10_scorers_357.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,cols2);w.writeheader()
        for u in units:
            for v in u["scorer_variants"]:
                w.writerow({"ordinal":u["ordinal"],"system":u["system"],"bw_ratio":u["bw_ratio"],
                  "snr_db":u["snr_db"],"classifier_variant":v["classifier_variant"],"n_correct":v["n_correct"],
                  "n_total":v["n_total"],"accuracy_end_to_end":v["n_correct"]/v["n_total"],
                  "per_image_sha256":v["sha256"]})

def style():
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10.5,"axes.titlesize":13,
      "axes.titleweight":"bold","axes.labelsize":11,"axes.spines.top":False,"axes.spines.right":False,
      "axes.edgecolor":"#55616A","axes.labelcolor":"#27313A","text.color":"#27313A",
      "xtick.color":"#4B5660","ytick.color":"#4B5660","grid.color":"#DDE4E9",
      "grid.linewidth":0.7,"legend.frameon":False,"savefig.transparent":False,
      "pdf.fonttype":42,"svg.fonttype":"none"})

def ax_style(ax, xlim=(-8,18), ylim=(0,1), ylabel=True, ticks=None):
    ax.set_xlim(*xlim);ax.set_ylim(*ylim)
    ax.set_xlabel("Channel SNR (dB)")
    if ylabel:ax.set_ylabel("End-to-end accuracy")
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.set_xticks(ticks if ticks is not None else [-8,-4,0,4,8,12,18])
    ax.grid(axis="y");ax.set_axisbelow(True)

def curve(ax,b,system,ratio="r_1_6",variant=None,label=None,ls="-",mark="o",lw=2.35,ms=4.2):
    vals=b[(system,ratio)]
    y=[(next(v["n_correct"] for v in u["scorer_variants"] if v["classifier_variant"]==variant) if variant else u["n_correct"])/1000 for u in vals]
    ax.plot(GRID,y,label=label or NAME[system],color=COL[system],lw=lw,ls=ls,marker=mark,ms=ms,
      markeredgecolor="white",markeredgewidth=.55)

def save(fig,name):
    for ext in ["png","pdf","svg"]:
        p=OUT/"figures"/ext;p.mkdir(parents=True,exist_ok=True)
        fig.savefig(p/(name+"."+ext),dpi=300,bbox_inches="tight",facecolor="white")
    plt.close(fig)

def make(b):
    style()
    fig,ax=plt.subplots(figsize=(10.4,5.5),layout="constrained")
    curve(ax,b,"learned");curve(ax,b,"classical_adaptive")
    ax_style(ax,ylim=(0,1));ax.set_title("01  Learned and adaptive classical transmission  |  1/6 bandwidth")
    ax.annotate("Digital delivery transition",xy=(-4,.834),xytext=(-1.9,.52),
      arrowprops={"arrowstyle":"-","color":"#5B6570"},fontsize=10,color="#5B6570")
    ax.legend(loc="lower right",ncol=2)
    save(fig,"01_headline_full_snr")

    fig,ax=plt.subplots(figsize=(9.4,5.5),layout="constrained")
    for s in ["learned","learned_snr_randomised","classical_adaptive"]:curve(ax,b,s)
    ax_style(ax,xlim=(-8,2),ylim=(0,1),ticks=[-8,-6,-4,-2,0,2])
    ax.set_title("02  Low-SNR behavior  |  1/6 bandwidth")
    ax.legend(loc="lower right")
    save(fig,"02_low_snr_closeup")

    fig,axs=plt.subplots(1,2,figsize=(12.5,5.0),sharey=True,layout="constrained")
    for ax,r,title in zip(axs,["r_1_24","r_1_6"],["1/24 bandwidth · 3,200 symbols","1/6 bandwidth · 12,800 symbols"]):
        curve(ax,b,"learned",r);curve(ax,b,"classical_adaptive",r)
        ax_style(ax,ylim=(0,1),ylabel=ax is axs[0]);ax.set_title(title)
    axs[1].legend(loc="lower right")
    save(fig,"03_bandwidth_efficiency")

    fig,ax=plt.subplots(figsize=(10.4,5.2),layout="constrained")
    curve(ax,b,"learned",label="Fixed-SNR training");curve(ax,b,"learned_snr_randomised")
    ax_style(ax,ylim=(.45,.9));ax.set_title("04  Training robustness  |  1/6 bandwidth")
    ax.legend(loc="lower right")
    save(fig,"04_training_robustness")

    fig,ax=plt.subplots(figsize=(10.4,5.2),layout="constrained")
    curve(ax,b,"learned");curve(ax,b,"er9_digital")
    ax_style(ax,ylim=(0,1));ax.set_title("05  Task-aware digital control  |  matched 1/6 channel-use budget")
    ax.legend(loc="lower right")
    save(fig,"05_task_aware_digital")

    fig,axs=plt.subplots(1,2,figsize=(12.5,5.0),layout="constrained",gridspec_kw={"width_ratios":[2.1,1]})
    curve(axs[0],b,"learned");curve(axs[0],b,"learned_papr_constrained")
    ax_style(axs[0],ylim=(.65,.9));axs[0].set_title("Accuracy across SNR · 1/6 bandwidth")
    axs[0].legend(loc="lower right")
    vals=[b[(s,"r_1_6")][0] for s in ["learned","learned_papr_constrained"]]
    for i,(s,u) in enumerate(zip(["learned","learned_papr_constrained"],vals)):
        y=1-i
        axs[1].plot([u["mean_papr_db"],u["max_papr_db"]],[y,y],color=COL[s],lw=5,solid_capstyle="round")
        axs[1].scatter([u["mean_papr_db"],u["max_papr_db"]],[y,y],color=COL[s],s=[35,65],edgecolor="white",zorder=4)
        axs[1].text(u["max_papr_db"]+.55,y,f'{u["max_papr_db"]:.2f}',va="center",fontsize=10)
    axs[1].axvline(3,color="#555555",ls="--",lw=1.4,label="3 dB cap")
    axs[1].set_yticks([1,0],["Standard","Constrained"]);axs[1].set_xlim(0,24)
    axs[1].set_xlabel("Symbol-domain PAPR (dB)");axs[1].set_title("Mean ● to observed maximum ●")
    axs[1].grid(axis="x");axs[1].legend(loc="lower right")
    save(fig,"06_papr_tradeoff")

    fig,axs=plt.subplots(2,2,figsize=(12.4,9.0),layout="constrained")
    specs=[("Adaptation controls",[("classical_adaptive",None),("classical_fixed_mcs",None),("classical_fixed_mod",None)]),
      ("Image codec control",[("classical_adaptive",None),("classical_jpeg_secondary",None)]),
      ("Predicted-label control",[("er9_digital",None),("label_transmission_bound",None)]),
      ("Reconstruction ablation · different scorer",[("learned",None),("semantic_recon_ablation",None)])]
    for ax,(title,series) in zip(axs.flat,specs):
        for j,(s,v) in enumerate(series):curve(ax,b,s,variant=v,ls="-" if j==0 else "--",mark="o" if j==0 else "s",lw=2,ms=3.4)
        ax_style(ax,ylim=(0,1),ylabel=ax in axs[:,0]);ax.set_title(title);ax.legend(fontsize=8.5,loc="lower right")
    save(fig,"07_secondary_controls")

    fig,axs=plt.subplots(2,1,figsize=(10.4,8.0),sharex=True,layout="constrained")
    for s in ["learned","classical_adaptive","er9_digital"]:curve(axs[0],b,s)
    ax_style(axs[0],ylim=(0,1));axs[0].set_xlabel("");axs[0].set_title("End-to-end accuracy · 1/6 bandwidth")
    axs[0].legend(loc="lower right",ncol=3,fontsize=9)
    for s in ["classical_adaptive","er9_digital"]:
        axs[1].plot(GRID,[u["coverage_rate"] for u in b[(s,"r_1_6")]],color=COL[s],lw=2.35,marker="o",ms=4.2,label=NAME[s])
    axs[1].plot(GRID,[u["coverage_rate"] for u in b[("classical_adaptive","r_1_24")]],color=COL["classical_adaptive"],lw=1.8,ls="--",marker="s",ms=3.8,label="Adaptive classical · 1/24")
    axs[1].set_xlim(-8,18);axs[1].set_ylim(0,1.06);axs[1].set_xticks([-8,-4,0,4,8,12,18]);axs[1].set_xlabel("Channel SNR (dB)")
    axs[1].set_ylabel("Delivery coverage");axs[1].yaxis.set_major_formatter(PercentFormatter(1,decimals=0));axs[1].grid(axis="y")
    axs[1].legend(loc="lower right",ncol=2,fontsize=9)
    save(fig,"08_delivery_coverage")

if __name__=="__main__":
    close,units,arms=verify()
    # Ordinal boundaries are recorded in the published closeout; no data are regenerated.
    provenance=close["ordinal_provenance"]
    ids=[x["authority_id"] for x in provenance]
    u_epoch=[ids.index(close["v9_authority_id"]),ids.index(close["v11_authority_id"])]
    write_csvs(units);make(arms)
    print(f'PASS: {len(units)} units, {sum(len(u["scorer_variants"]) for u in units)} scorer streams, 12 arms × 21 SNR points; epochs {u_epoch}')
