#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple plotting script for face-based peak success and C_T vs alpha.
Only uses data with alpha < 0.6.
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

mask_threshold=0.7

def _fit_loglog_slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if mask.sum() < 2:
        return None

    X = np.log10(x[mask])
    Y = np.log10(y[mask])
    A = np.vstack([X, np.ones_like(X)]).T
    m, b = np.linalg.lstsq(A, Y, rcond=None)[0]
    return float(m), float(b), mask


def _apply_alpha_mask(alpha_grid, data_dict, sam_nums):
    """
    Filters peaks or CT_mean to alpha < 0.6.
    data_dict[case][S] must be an array of same length as alpha_grid.
    Returns a new filtered dict.
    """
    mask = (alpha_grid < mask_threshold)
    filtered = {}

    for case in data_dict:
        filtered[case] = {}
        for S in sam_nums:
            arr = np.asarray(data_dict[case][S], dtype=float)
            if len(arr) == len(alpha_grid):
                filtered[case][S] = arr[mask]
            else:
                # mismatch—keep original
                filtered[case][S] = arr
    return mask, filtered


def plot_peaks_from_file(dat, fig_prefix='peak_faces'):
    alpha_grid = np.asarray(dat['alpha_grid'], dtype=float)
    sam_nums   = dat['sam_nums']
    peaks_full = dat['peaks']

    # -------------------------------
    # APPLY α < 0.6 FILTER
    # -------------------------------
    mask = (alpha_grid < mask_threshold)
    alpha_grid_f = alpha_grid[mask]

    peaks = {}
    for case in peaks_full:
        peaks[case] = {}
        for S in sam_nums:
            arr = np.asarray(peaks_full[case][S], dtype=float)
            peaks[case][S] = arr[mask]

    # -------------------------------
    # PLOTTING
    # -------------------------------
    cases = [
        ('DI',     'DI'),
        ('sSPADE', 'Separate SPADE'),
        ('oSPADE', 'Orthogonal SPADE'),
    ]

    for case_name, title in cases:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

        for side in ('bottom', 'left', 'top', 'right'):
            ax.spines[side].set_linewidth(2)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        for S_test in sam_nums:
            yvals = peaks[case_name][S_test]
            safe  = np.clip(1.0 - yvals, 1.0e-12, 1.0)
            yplot = -np.log(safe)

            # ------------------------
            # FIXED: use markersize, not s
            # ------------------------
            line, = ax.plot(
                alpha_grid_f,
                yplot,
                marker='o',
                markersize=6,
                zorder=3,
                label=f'{S_test:.0e}',
            )

            # slope fit
            fit = _fit_loglog_slope(alpha_grid_f, yplot)
            if fit is not None:
                m, b, msk2 = fit
                xfit = alpha_grid_f[msk2]
                yfit = (10 ** b) * (xfit ** m)

                # color = line.get_color()   # unused for now

                #ax.plot(xfit,yfit,linestyle='--',linewidth=1.8,color=color)

                # ------------------------
                # FIXED: change sc -> line
                # ------------------------
                line.set_label(f'{S_test:.0e} (slope={m:.2f})')

        ax.set_xscale('log')
        ax.set_yscale('log')
        #ax.set_xlabel(r'$\alpha$', fontsize=20)
        #ax.set_ylabel(r'$-\log(1-\mathrm{peak})$', fontsize=20)
        ax.set_title( f' — {title}', fontsize=22)
        ax.set_ylim(3.2e-1, 2)

        ax.yaxis.set_major_locator(
            ticker.LogLocator(base=10, subs=[1, 2, 4, 6, 8])
        )

        ax.yaxis.set_major_formatter(
            ticker.LogFormatterSciNotation(base=10)
        )

        ax.legend(fontsize=14,
                  loc='lower center',
                  bbox_to_anchor=(0.5, 1.02),
                  ncol=len(sam_nums))

        plt.gcf().savefig(f'{fig_prefix}_{case_name}_peak_vs_alpha_alpha_lt_06.jpg',
                          dpi=300, bbox_inches='tight')
        plt.show()
        
        
def plot_P_from_file(dat, fig_prefix='peak_faces'):
    alpha_grid = np.asarray(dat['alpha_grid'], dtype=float)
    sam_nums   = dat['sam_nums']
    peaks_full = dat['peaks']

    mask = (alpha_grid < mask_threshold)
    alpha_grid_f = alpha_grid[mask]

    peaks = {}
    for case in peaks_full:
        peaks[case] = {}
        for S in sam_nums:
            arr = np.asarray(peaks_full[case][S], dtype=float)
            peaks[case][S] = arr[mask]

    cases = [
        ('DI',     'DI'),
        ('sSPADE', 'Separate SPADE'),
        ('oSPADE', 'Orthogonal SPADE'),
    ]

    for case_name, title in cases:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

        for side in ('bottom', 'left', 'top', 'right'):
            ax.spines[side].set_linewidth(2)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        for S_test in sam_nums:
            yplot = peaks[case_name][S_test]

            # ------------------------
            # FIXED: use markersize instead of s
            # ------------------------
            line, = ax.plot(
                alpha_grid_f,
                yplot,
                marker='o',
                markersize=6,
                zorder=3,
                label=f'{S_test:.0e}',
            )

            fit = _fit_loglog_slope(alpha_grid_f, yplot)
            if fit is not None:
                m, b, msk2 = fit

                # ------------------------
                # FIXED
                # ------------------------
                line.set_label(f'{S_test:.0e} (slope={m:.2f})')

        ax.set_xscale('log')
        #ax.set_xlabel(r'$\alpha$', fontsize=20)
        #ax.set_ylabel(r'$-\log(1-\mathrm{peak})$', fontsize=20)
        ax.set_title( f' — {title}', fontsize=22)
        ax.set_ylim(0.3, 0.9)

        ax.legend(fontsize=14,
                  loc='lower center',
                  bbox_to_anchor=(0.5, 1.02),
                  ncol=len(sam_nums))

        plt.gcf().savefig(f'{fig_prefix}_{case_name}_P_vs_alpha_alpha_lt_06.jpg',
                          dpi=300, bbox_inches='tight')
        plt.show()



def plot_CT_from_file(dat, fig_prefix='CT_faces'):
    if 'CT_mean' not in dat:
        print("No CT_mean in data; nothing to plot.")
        return

    alpha_grid = np.asarray(dat['alpha_grid'], dtype=float)
    sam_nums   = dat['sam_nums']
    CT_full    = dat['CT_mean']

    mask = (alpha_grid < mask_threshold)
    alpha_grid_f = alpha_grid[mask]

    CT = {}
    for case in CT_full:
        CT[case] = {}
        for S in sam_nums:
            arr = np.asarray(CT_full[case][S], dtype=float)
            CT[case][S] = arr[mask]

    cases = [
        ('DI',     'DI'),
        ('sSPADE', 'Separate SPADE'),
        ('oSPADE', 'Orthogonal SPADE'),
    ]

    for case_name, title in cases:
        plt.figure()
        ax = plt.subplot(1, 1, 1)

        for side in ('bottom', 'left', 'top', 'right'):
            ax.spines[side].set_linewidth(2)

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        for S_test in sam_nums:
            yvals = CT[case_name][S_test]
            ax.plot(alpha_grid_f, yvals, marker='o', label=f'{S_test:.0e}')

        ax.set_xscale('log')
        #ax.set_xlabel(r'$\alpha$', fontsize=20)
        #ax.set_ylabel(r'$C_T$', fontsize=20)
        ax.set_title(r'$C_T$ ' + f' — {title}', fontsize=22)

        ax.legend(fontsize=14,
                  loc='lower center',
                  bbox_to_anchor=(0.5, 1.02),
                  ncol=len(sam_nums))

        plt.gcf().savefig(f'{fig_prefix}_{case_name}_CT_vs_alpha_alpha_lt_06.jpg',
                          dpi=300, bbox_inches='tight')
        plt.show()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

if __name__ == '__main__':
    PKL_PATH = 'results_faces/peak_success_vs_alpha_faces.pkl'

    with open(PKL_PATH, 'rb') as f:
        data = pickle.load(f)

    plot_peaks_from_file(data)
    plot_CT_from_file(data)
    plot_P_from_file(data)
