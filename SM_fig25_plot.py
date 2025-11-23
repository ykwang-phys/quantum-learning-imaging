#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot:
1) -log(1 - peak) vs alpha with LR / LRT / Chernoff for each method
2) peak vs alpha for each method

Reads results saved by combined_*.py (e.g. combined_point_nogrid.pkl).
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt


def plot_three(results, methods, fig_prefix='point_nogrid'):
    """
    Plot -log(1 - peak) vs alpha for each method (DI, sSPADE, oSPADE),
    showing LR (scatter), LRT (dashed line), and Chernoff (solid line).
    """
    alpha_grid = np.asarray(results['alpha_grid'], dtype=float)
    sam_nums   = results['sam_nums']
    emp_mode   = results['emp_mode']
    #methods    = results['methods']
    peaks_lrt  = results['peaks_lrt']      # dict[meas][S][ai]
    peaks_lr   = results['peaks_lr']       # dict[meas][S][ai]
    Cinfo      = results['chernoff_info']  # dict[meas][ai]

    # Choose titles for each method
    titles = {
        'DI': 'Direct Imaging',
        'sSPADE': 'Separate SPADE',
        'oSPADE': 'Orthogonal SPADE',
    }

    for meas in methods:
        
        #print (meas)
        title = titles.get(meas, meas)

        fig, ax = plt.subplots()
        for side in ('bottom', 'left', 'top', 'right'):
            ax.spines[side].set_linewidth(2)

        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        # Logistic regression (best over K) — scatter
        for S in sam_nums:
            vals = np.asarray(peaks_lr[meas][S], dtype=float)
            yplot = -np.log(np.clip(1.0 - vals, 1e-12, 1.0))
            ax.scatter(alpha_grid, yplot,
                       s=40,
                       label=f'LR best, S={S:.0e}')

        # LRT (all outcomes) — dashed line
        for S in sam_nums:
            vals = np.asarray(peaks_lrt[meas][S], dtype=float)
            yplot = -np.log(np.clip(1.0 - vals, 1e-12, 1.0))
            ax.plot(alpha_grid, yplot,
                    linestyle='--',
                    linewidth=3.0,
                    label=f'LRT, S={S:.0e}')

        # Chernoff — solid line, y = S * C(alpha)
        C_alpha = np.asarray(Cinfo[meas], dtype=float)
        for S in sam_nums:
            ax.plot(alpha_grid, C_alpha * S,
                    linestyle='-',
                    linewidth=3.0,
                    label=f'Chernoff, S={S:.0e}')

        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.set_ylabel(r'$-\log(1-\mathrm{peak})$', fontsize=20)
        ax.set_title(f'Peak vs $\\alpha$ — {title}  [{emp_mode}]', fontsize=14)

        leg = ax.legend(fontsize=14,
                        loc='center left',
                        bbox_to_anchor=(1.02, 0.5),
                        ncol=1)


        out_fig = f'{fig_prefix}_{meas}_minuslog1mpeak_vs_alpha_three_{emp_mode}.jpg'
        fig.savefig(out_fig, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[plot] Saved → {out_fig}")


def plot_peaks(results, methods, fig_prefix='point_nogrid'):
    """
    Plot peak vs alpha (linear y-axis) for each method,
    showing LR (solid line with circles) and LRT (dashed line with squares).
    """
    alpha_grid = np.asarray(results['alpha_grid'], dtype=float)
    sam_nums   = results['sam_nums']
    #methods    = results['methods']
    peaks_lrt  = results['peaks_lrt']
    peaks_lr   = results['peaks_lr']

    titles = {
        'DI': 'Direct Imaging',
        'sSPADE': 'Separate SPADE',
        'oSPADE': 'Orthogonal SPADE',
    }

    for meas in methods:
        title = titles.get(meas, meas)

        fig, ax = plt.subplots()
        for side in ('bottom', 'left', 'top', 'right'):
            ax.spines[side].set_linewidth(2)

        ax.set_xscale('log')   # alpha on log-scale
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        # LR peaks
        for S in sam_nums:
            vals_lr = np.asarray(peaks_lr[meas][S], dtype=float)
            ax.scatter(alpha_grid, vals_lr,
                    marker='o',
                    linewidth=2.5,
                    label=f'LR best, S={S:.0e}')

        # LRT peaks
        for S in sam_nums:
            vals_lrt = np.asarray(peaks_lrt[meas][S], dtype=float)
            ax.plot(alpha_grid, vals_lrt,
                    linestyle='-',
                    #marker='s',
                    linewidth=2.5,
                    label=f'LRT, S={S:.0e}')

        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.set_ylabel(r'Peak success', fontsize=20)
        ax.set_title(f'Peak vs $\\alpha$ — {title}', fontsize=14)
        leg = ax.legend(
            fontsize=14,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            ncol=1
        )

        out_fig = f'{fig_prefix}_{meas}_peak_vs_alpha.jpg'
        fig.savefig(out_fig, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"[plot] Saved → {out_fig}")


# -------- Run on a given pickle file --------
pkl_path  = 'combined_point_nogrid_DI.pkl'
fig_prefix = 'point_nogrid'
methods=['DI']

with open(pkl_path, 'rb') as f:
    results = pickle.load(f)

plot_three(results,methods, fig_prefix=fig_prefix)
plot_peaks(results, methods,fig_prefix=fig_prefix)


pkl_path  = 'combined_point_nogrid_SPADE.pkl'
fig_prefix = 'point_nogrid'
methods=['sSPADE']

with open(pkl_path, 'rb') as f:
    results = pickle.load(f)

plot_three(results,methods, fig_prefix=fig_prefix)
plot_peaks(results, methods,fig_prefix=fig_prefix)
