#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 19 21:16:13 2025

@author: yunkaiwang
"""

import pickle
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import eig,eigh,inv
import matplotlib.pyplot as plt
import random
import time
from scipy import integrate
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from matplotlib.ticker import MultipleLocator, LogLocator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from scipy.linalg import qr
from scipy.special import hermite
import math
from scipy.linalg import eig,eigh,inv
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm   

with open('success_plot_data9.pkl', 'rb') as f:
    data = pickle.load(f)
    
    rates_DI     = data['rates_DI']
    rates_sSPADE = data['rates_sSPADE']
    rates_oSPADE = data['rates_oSPADE']
    sam_nums     = data['sam_nums']
    C            = data['total_task']
    total_task =data['total_task']

        
    # -- after you’ve computed rates_DI, rates_sSPADE, rates_oSPADE and have sam_nums, total_task --
    
    # 1) Compute global y‐limits
    all_mins = []
    all_maxs = []
    
    for rates in (rates_DI, rates_sSPADE, rates_oSPADE):
        for sam in sam_nums:
            data = rates[sam]            # shape (repeats, total_task)
            all_mins.append(data.min())
            all_maxs.append(data.max())
    
    ymin = min(all_mins)
    ymax = max(all_maxs)
    # add a small padding (e.g. 2% of the span)
    pad = 0.02 * (ymax - ymin)
    ymin -= pad
    ymax += pad
    
    # 2) Updated plotting function to accept y‐limits
    def plot_success(success_dict, case_name, C, sam_nums, y_limits):
        plt.figure()
        ax = plt.subplot(1,1,1)
        ax.xaxis.set_major_locator(MultipleLocator(5))
        for side in ('bottom','left','top','right'):
            ax.spines[side].set_linewidth(2)
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)
    
        colors     = ['C0','C1','C2']
        linestyles = ['-','--','-.']
    
        for idx, sam in enumerate(sam_nums):
            data      = success_dict[sam]
            mean_succ = np.mean(data, axis=0)
            min_succ  = np.min(data, axis=0)
            max_succ  = np.max(data, axis=0)
    
            x_vals = np.arange(1, C+1)
            ax.plot(x_vals, mean_succ,
                    linestyle=linestyles[idx],
                    color=colors[idx],
                    label=f'{sam:.0e}')
            ax.fill_between(x_vals, min_succ, max_succ,
                            alpha=0.2, color=colors[idx])
    
        ax.set_ylim(*y_limits)
        ax.set_xlim(-1,21)
        #ax.set_title(case_name, fontsize=28)
        #ax.set_xlabel('Fitting order', fontsize=24)
        #ax.set_ylabel('Success rate', fontsize=24)
        #ax.legend(fontsize=20,loc='center left',bbox_to_anchor=(1.02, 0.5),ncol=1)
    
        fig = plt.gcf()
        fig.savefig(f'{case_name.replace(" ","_")}_success_rates.jpg',
                    dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3) Call it for each case with the same y‐limits
    C = total_task
    y_limits = (ymin, ymax)
    
    plot_success(rates_DI,    'Direct Imaging',   C, sam_nums, y_limits)
    plot_success(rates_sSPADE,'Separate SPADE',   C, sam_nums, y_limits)
    plot_success(rates_oSPADE,'Orthogonal SPADE', C, sam_nums, y_limits)
    