#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import matplotlib.pyplot as plt

FNAME = "success_vs_K_keep_multiS_first10.pkl"

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------
with open(FNAME, "rb") as f:
    data = pickle.load(f)

acc_records = data["acc_records"]
K_sweep = np.array(data["K_sweep"], dtype=int)
methods = data["methods"]        # e.g. ["DI", "sSPADE", "oSPADE"]
S_list = data["S_list"]          # sample numbers list

# ---- Restrict K_sweep to <= 15 only ----
mask = K_sweep <= 15
K = K_sweep[mask]

# ---- Helper ----
def compute_stats(vals):
    arr = np.array([v for v in vals if not np.isnan(v)], float)
    if arr.size == 0:
        return np.nan, np.nan, np.nan
    return arr.mean(), arr.min(), arr.max()


linestyles = ['-', '--', '-.', ':', (0,(3,1,1,1)), (0,(5,1))]

# ============================================================
#               DIRECT IMAGING PLOT
# ============================================================
method = "Direct Imaging"     # 手动指定
plt.figure(figsize=(7.6,5))
ax = plt.subplot(111)

ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.spines["top"].set_linewidth(2)
ax.spines["right"].set_linewidth(2)

ax.set_title("Direct imaging", fontsize=30)

for i, S in enumerate(S_list):
    ls = linestyles[i % len(linestyles)]
    means, mins, maxs = [], [], []
    for k in K:
        m, mn, mx = compute_stats(acc_records[method][S][k])
        means.append(m); mins.append(mn); maxs.append(mx)
    means = np.array(means); mins = np.array(mins); maxs = np.array(maxs)
    ax.plot(K, means, linewidth=3, linestyle=ls, label=f"S = {S:.0e}")
    ax.fill_between(K, mins, maxs, alpha=0.15)

ax.set_ylim(0,1.0)
ax.set_xticks([0,5,10,15])
ax.set_xticklabels(["0","5","10","15"], fontsize=23)
ax.tick_params(axis="y", labelsize=23)
#ax.legend(fontsize=17, bbox_to_anchor=(1.02,0.5), loc="center left")

plt.savefig("plot_DI.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved plot_DI.png")


# ============================================================
#               SEPARATE SPADE PLOT
# ============================================================
method = "Separate SPADE"
plt.figure(figsize=(7.6,5))
ax = plt.subplot(111)

for side in ("bottom","left","top","right"):
    ax.spines[side].set_linewidth(2)

ax.set_title("Separate SPADE", fontsize=30)

for i, S in enumerate(S_list):
    ls = linestyles[i % len(linestyles)]
    means, mins, maxs = [], [], []
    for k in K:
        m, mn, mx = compute_stats(acc_records[method][S][k])
        means.append(m); mins.append(mn); maxs.append(mx)
    means = np.array(means); mins = np.array(mins); maxs = np.array(maxs)
    ax.plot(K, means, linewidth=3, linestyle=ls, label=f"S = {S:.0e}")
    ax.fill_between(K, mins, maxs, alpha=0.15)

ax.set_ylim(0,1.0)
ax.set_xticks([0,5,10,15])
ax.set_xticklabels(["0","5","10","15"], fontsize=23)
ax.tick_params(axis="y", labelsize=23)
#ax.legend(fontsize=17, bbox_to_anchor=(1.02,0.5), loc="center left")

plt.savefig("plot_sSPADE.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved plot_sSPADE.png")


# ============================================================
#               ORTHOGONALIZED SPADE PLOT
# ============================================================
method = "Orthogonal SPADE"
plt.figure(figsize=(7.6,5))
ax = plt.subplot(111)

for side in ("bottom","left","top","right"):
    ax.spines[side].set_linewidth(2)

ax.set_title("Orthogonalized SPADE", fontsize=30)

for i, S in enumerate(S_list):
    ls = linestyles[i % len(linestyles)]
    means, mins, maxs = [], [], []
    for k in K:
        m, mn, mx = compute_stats(acc_records[method][S][k])
        means.append(m); mins.append(mn); maxs.append(mx)
    means = np.array(means); mins = np.array(mins); maxs = np.array(maxs)
    ax.plot(K, means, linewidth=3, linestyle=ls, label=f"S = {S:.0e}")
    ax.fill_between(K, mins, maxs, alpha=0.15)

ax.set_ylim(0,1.0)
ax.set_xticks([0,5,10,15])
ax.set_xticklabels(["0","5","10","15"], fontsize=23)
ax.tick_params(axis="y", labelsize=23)
#ax.legend(fontsize=17, bbox_to_anchor=(1.02,0.5), loc="center left")

plt.savefig("plot_oSPADE.png", dpi=300, bbox_inches="tight")
plt.show()
print("Saved plot_oSPADE.png")
