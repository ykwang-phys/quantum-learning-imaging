# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 05:48:35 2025

@author: 27432

Modified from rate_face_20251113_1.py to add CT - alpha plot
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Nov 15 02:37:10 2025

@author: 27432

Modified from rate_face_20251113_1.py to add CT - alpha plot
and to use S_eig = sam_nums (each S_test) in the eigensystem.
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 19:59:54 2025

@author: 27432
"""

# -*- coding: utf-8 -*-
"""
Peak success vs alpha with faces — DI / sSPADE / oSPADE

- Saves all results to a pickle
- Returns full results dict
- Submits ALL (alpha, repeat) tasks at once to a ProcessPool
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, hashlib, pickle, math
from pathlib import Path

# --- Limit BLAS threads globally (safety; also done in workers) ---
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import integrate
from scipy.linalg import eigh, inv, qr
from scipy.special import hermite
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context, freeze_support

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_olivetti_faces

# =========================
# GLOBAL SHARED STATE (in workers)
# =========================

_SHARED = {}

# =========================
# CACHE HELPERS
# =========================

_CACHE_DIR = Path("cache_kernels")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _as_jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, (list, tuple)):
        return [_as_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _as_jsonable(v) for k, v in obj.items()}
    return obj


def _hash_params(name, params_dict):
    payload = {"name": name, "params": _as_jsonable(params_dict)}
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib as _hashlib
    return _hashlib.sha1(payload_bytes).hexdigest()[:16]


def _cache_path(name, params_dict):
    key = _hash_params(name, params_dict)
    return _CACHE_DIR / f"{name}_{key}.npz"


def _save_arrays(path: Path, **arrays):
    np.savez_compressed(path, **arrays)


def _try_load_arrays(path: Path):
    if path.exists():
        data = np.load(path, allow_pickle=False)
        return {k: data[k] for k in data.files}
    return None

# =========================
# SPADE / optics utilities
# =========================

def inner_product(vectors):
    V = np.column_stack(vectors)
    Q, _ = qr(V, mode='economic')
    product = Q.T @ V
    return product


def inner_product_directSPADE(vectors, nmax, M):
    V = np.column_stack(vectors)
    product = np.zeros([(nmax+1)*M, (nmax+1)*M])
    for i in range(M):
        vectors_temp = [vectors[i+M*j] for j in range(nmax+1)]
        V_temp = np.column_stack(vectors_temp)
        Q, _ = qr(V_temp, mode='economic')
        product_temp = Q.T @ V
        for j in range(nmax+1):
            product[i+M*j, :] = product_temp[j, :]
    return product


def psi_fun(y, y0, sigma, n):
    Hn = hermite(n)
    return (1 / (2 * np.pi * sigma**2)**(1/4)
            * (-1)**n
            * Hn((y - y0) / (2 * sigma))
            * np.exp(-(y - y0)**2 / (4 * sigma**2))
            / (2 * sigma)**n)


def psi_array_fun(L, Lmax, sigma, N, M, nmax, y0):
    result = np.zeros([N, M, nmax + 1])
    y = np.linspace(-Lmax/2, Lmax/2, N)
    for i in range(nmax + 1):
        for j in range(M):
            result[:, j, i] = psi_fun(y, y0[j], sigma, i) * np.sqrt(Lmax/N)
    return result


def a_coeff_fun(product, M, nmax, sigma):
    n = nmax + 1
    result = np.zeros([n, M, n, M])
    for i1 in range(n):
        for i2 in range(n):
            for j1 in range(M):
                for j2 in range(M):
                    right = i1*M + j1
                    left = i2*M + j2
                    result[i1, j1, i2, j2] = product[left, right] * sigma**i1 / math.factorial(i1)
    return result


def C_coeff(n, M, a):
    result = np.zeros([n, 2*M, n, M])
    for l in range(n):
        for p in range(n):
            for j in range(M):
                for k in range(M):
                    for m in range(p+1):
                        s = p - m
                        result[l, 2*j, p, k] += a[s, k, l, j]*a[m, k, l, j].conjugate() + a[s, k, l+1, j]*a[m, k, l+1, j].conjugate()
                        result[l, 2*j, p, k] += a[s, k, l+1, j]*a[m, k, l, j].conjugate() + a[s, k, l, j]*a[m, k, l+1, j].conjugate()
                        result[l, 2*j+1, p, k] += a[s, k, l, j]*a[m, k, l, j].conjugate() + a[s, k, l+1, j]*a[m, k, l+1, j].conjugate()
                        result[l, 2*j+1, p, k] -= a[s, k, l+1, j]*a[m, k, l, j].conjugate() + a[s, k, l, j]*a[m, k, l+1, j].conjugate()
    return result/4


def C_SPADE_coeff(n, M, a):
    result = np.zeros([n, 2*M, n, M])
    for l in range(n):
        for p in range(n):
            for j in range(M):
                for k in range(M):
                    for m in range(p+1):
                        s = p - m
                        result[l, 2*j, p, k] += a[s, k, l, j]*a[m, k, l, j].conjugate() + a[s, k, l+1, j]*a[m, k, l+1, j].conjugate()
                        result[l, 2*j, p, k] += a[s, k, l+1, j]*a[m, k, l, j].conjugate() + a[s, k, l, j]*a[m, k, l+1, j].conjugate()
                        result[l, 2*j+1, p, k] += a[s, k, l, j]*a[m, k, l, j].conjugate() + a[s, k, l+1, j]*a[m, k, l+1, j].conjugate()
                        result[l, 2*j+1, p, k] -= a[s, k, l+1, j]*a[m, k, l, j].conjugate() + a[s, k, l, j]*a[m, k, l+1, j].conjugate()
    return result/(4*M)


def C_SPADE_coeff_extra(n, M, a):
    result = np.zeros([M, n, M])
    for p in range(n):
        for q in range(M):
            for j in range(M):
                for m in range(p+1):
                    s = p - m
                    result[j, p, q] += a[m, q, 0, j]*a[s, q, 0, j].conjugate()
    return result/(2*M)


def C_new_coeff_extra(n, M, a):
    result = np.zeros([M, n, M])
    for p in range(n):
        for q in range(M):
            for j in range(M):
                for m in range(p+1):
                    s = p - m
                    result[j, p, q] += a[m, q, 0, j]*a[s, q, 0, j].conjugate()
    return result/2


def C_directimaging_coeff_integral(n, M, N_measure, L_measure, y0, sigma):
    u_temp = np.linspace(-L_measure/2, L_measure/2, N_measure+1)
    u_array = (u_temp[:-1] + u_temp[1:]) / 2
    l = L_measure/N_measure
    result = np.zeros([N_measure, n, M])

    def integrate_psi(u, y0_, sigma_, p, q, i, k):
        return (psi_fun(u, y0_, sigma_, p)
                * np.conjugate(psi_fun(u, y0_, sigma_, q))
                / (math.factorial(p) * math.factorial(q))
                * sigma_**i)

    for i in range(n):
        for k in range(M):
            for p in range(i+1):
                q = i - p
                temp = np.zeros(N_measure)
                for j in range(N_measure):
                    temp[j], _ = integrate.quad(
                        integrate_psi,
                        u_array[j]-l/2,
                        u_array[j]+l/2,
                        args=(y0[k], sigma, p, q, i, k)
                    )
                result[:, i, k] += temp.astype(np.float64)
    return result

# =========================
# Matrix assembly & eigens
# =========================

def GD(g, d, C, alpha, n, M, C_extra):
    D = np.zeros([n*2*M+M, n*2*M+M])
    G = np.zeros([n*2*M+M, n*2*M+M])
    for m in range(n):
        for k in range(M):
            D += alpha**m * d[m, k]*np.diag(np.append(C[:, :, m, k].flatten(), C_extra[:, m, k]))
    for m in range(n):
        for k1 in range(M):
            for k2 in range(M):
                for m1 in range(m+1):
                    m2 = m - m1
                    left = np.array([np.append(C[:, :, m1, k1].flatten(), C_extra[:, m1, k1])]).T
                    right = np.array([np.append(C[:, :, m2, k2].flatten(), C_extra[:, m2, k2])])
                    index1 = m1*M + k1
                    index2 = m2*M + k2
                    G += alpha**m * g[index1, index2] * left @ right
    return G, D


def GD_directimaging(g, d, C, alpha, n, M, N_measure):
    D = np.zeros([N_measure, N_measure])
    G = np.zeros([N_measure, N_measure])
    for m in range(n):
        for k in range(M):
            D += alpha**m * d[m, k] * np.diag(C[:, m, k])
    for m in range(n):
        for k1 in range(M):
            for k2 in range(M):
                for m1 in range(m+1):
                    m2 = m - m1
                    left = np.array([C[:, m1, k1]]).T
                    right = np.array([C[:, m2, k2]])
                    index1 = m1*M + k1
                    index2 = m2*M + k2
                    G += alpha**m * g[index1, index2] * left @ right
    return G, D


def Dhalfinv_directimaging_fun(g, d, C, alpha, n, M, N_measure):
    result = np.zeros(N_measure)
    for m in range(n):
        for k in range(M):
            result += alpha**m * d[m, k]*C[:, m, k]
    for i in range(N_measure):
        if result[i] < 0 and np.abs(result[i]) < 1e-14:
            result[i] += 1e-14
        elif result[i] < 0:
            print('Dhalfinv Error (DI)')
    return np.diag(1/np.sqrt(result))


def Dhalfinv_fun(g, d, C, alpha, n, M, C_extra):
    result = np.zeros(n*2*M+M)
    for m in range(n):
        for k in range(M):
            result += alpha**m * d[m, k]*(np.append(C[:, :, m, k].flatten(), C_extra[:, m, k]))
    for i in range(n*2*M):
        if result[i] < 0 and np.abs(result[i]) < 1e-14:
            result[i] += 1e-14
        elif result[i] < 0:
            print('Dhalfinv Error (SPADE)')
    return np.diag(1/np.sqrt(result))


def _eig_main(G, D, Dhalfinv, S):
    """
    S is the effective sample number, here taken as S_eig = S_test.
    """
    M_ = Dhalfinv @ G @ Dhalfinv
    l, v = eigh(M_)
    sort_indices = np.argsort(np.abs(np.real(l)))
    sorted_l = l[sort_indices]
    sorted_v = v[:, sort_indices]
    M1 = G + (D - G)/S
    M2 = inv(M1) @ G
    CT = float(np.matrix.trace(M2))
    return Dhalfinv, M_, sorted_l, sorted_v, CT

# =========================
# Probabilities & sampling (faces version)
# =========================

def Prob_array_direct_imaging(C1, alpha, x_array):
    (M, n, W_) = np.shape(x_array)
    (N_measure, n2, M2) = np.shape(C1)
    assert n2 == n and M2 == M
    result = np.zeros([N_measure, W_])
    for i in range(W_):
        res = np.zeros(N_measure)
        for j in range(n):
            for k in range(M):
                res += C1[:, j, k] * x_array[k, j, i] * alpha**j
        result[:, i] = res
    return result


def Prob_array_separate_SPADE(C2, C2_extra, alpha, x_array):
    (M, n, W_) = np.shape(x_array)
    (n2, M2, _, M3) = np.shape(C2)
    assert n2 == n and M2 == 2*M and M3 == M
    out = np.zeros([(n*2*M+M), W_])
    for i in range(W_):
        v = np.zeros(n*2*M+M)
        for j in range(n):
            for k in range(M*2):
                for m in range(M):
                    v[k + j*M*2] += C2[j, k, 0, m] * x_array[m, 0, i]
                    for l in range(1, n):
                        v[k + j*M*2] += C2[j, k, l, m] * x_array[m, l, i] * (alpha**l)
        for k in range(M):
            for m in range(M):
                for l in range(n):
                    v[k + n*M*2] += C2_extra[k, l, m] * x_array[m, l, i] * (alpha**l)
        out[:, i] = v
    return out


def Prob_array_orthogonal_SPADE(C3, C3_extra, alpha, x_array):
    (M, n, W_) = np.shape(x_array)
    out = np.zeros([(n*2*M+M), W_])
    (n2, M2, _, M3) = np.shape(C3)
    assert n2 == n and M2 == 2*M and M3 == M
    for i in range(W_):
        v = np.zeros(n*2*M+M)
        for j in range(n):
            for k in range(M*2):
                for m in range(M):
                    v[k + j*M*2] += C3[j, k, 0, m] * x_array[m, 0, i]
                    for l in range(1, n):
                        v[k + j*M*2] += C3[j, k, l, m] * x_array[m, l, i] * (alpha**l)
        for k in range(M):
            for m in range(M):
                for l in range(n):
                    v[k + n*M*2] += C3_extra[k, l, m] * x_array[m, l, i] * (alpha**l)
        out[:, i] = v
    return out

# --------- Faces utilities ---------

def _olivetti_flat_signals(num_persons=20):
    """
    Returns:
      images_1d: (num_images, 4096)
      persons:   (num_images,) labels 0..(num_persons-1)
    """
    faces = fetch_olivetti_faces()
    imgs = faces.images
    persons = faces.target  
    mask = persons < num_persons
    imgs = imgs[mask]
    persons = persons[mask]

    #flattened = []
    #for img in imgs:
        #rows_1d = [np.concatenate([row[:32], row[32:]]) for row in img]
        #sig = np.hstack(rows_1d)
        #flattened.append(sig)
    #images_1d = np.vstack(flattened).astype(np.float64)
    flattened = []
    for img in imgs:
        # Convert 64×64 → length-4096
        sig = img.flatten()      # OR img.reshape(-1)
        flattened.append(sig)

    images_1d = np.vstack(flattened).astype(np.float64)

    return images_1d, persons


def split_train_test_per_person(persons, num_persons=40, k_train_per_person=9, seed=0):
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for pid in range(num_persons):
        idx = np.where(persons == pid)[0]
        k = min(k_train_per_person, len(idx) - 1)
        pick_train = rng.choice(idx, size=k, replace=False)
        remain = np.setdiff1d(idx, pick_train)
        train_idx.extend(pick_train.tolist())
        test_idx.extend(remain.tolist())
    return np.array(train_idx, dtype=int), np.array(test_idx, dtype=int)


def _build_x_from_I(I, M, n, L):
    num_samples, N = I.shape
    row_sums = I.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    I_normalized = I / row_sums
    l = L / M
    y = np.linspace(-L/2, L/2, N)
    y0_temp = np.linspace(-L/2, L/2, M + 1)
    y0 = (y0_temp[:-1] + y0_temp[1:]) / 2
    m = N // M
    x = np.zeros((M, n, num_samples))
    for i in range(M):
        y_diff = (y[i*m:(i+1)*m] - y0[i]) / l
        y_diff_matrix = np.array([y_diff**j for j in range(n)])
        x[i, :, :] = np.dot(y_diff_matrix, I_normalized[:, i*m:(i+1)*m].T)
    d = np.mean(x, axis=2).T
    x_reshaped = x.transpose(1, 0, 2).reshape(M * n, num_samples)
    g = np.dot(x_reshaped, x_reshaped.T) / num_samples
    return d, g, x

# ---------- multinomial sampling ----------
_MAX_CHUNK = 1_000_000

def _sample_one_prob_vec(prob_vec, sam_num, rng):
    P = np.clip(prob_vec, 0, None)
    s = P.sum()
    if s <= 0:
        P = np.ones_like(P, dtype=float) / P.size
    else:
        P = P / s
    sam = int(sam_num)
    if sam <= _MAX_CHUNK:
        counts = rng.multinomial(sam, P)
        return counts.astype(float) / sam
    full = sam // _MAX_CHUNK
    rem  = sam - full * _MAX_CHUNK
    counts = np.zeros_like(P, dtype=np.int64)
    for _ in range(full):
        counts += rng.multinomial(_MAX_CHUNK, P)
    if rem > 0:
        counts += rng.multinomial(rem)
    return counts.astype(float) / sam


def sample_empirical_matrix(P_matrix, sam_num, seed=None):
    rng = np.random.default_rng(seed)
    M_out, W = P_matrix.shape
    out = np.zeros_like(P_matrix, dtype=float)
    for j in range(W):
        out[:, j] = _sample_one_prob_vec(P_matrix[:, j], sam_num, rng)
    return out

# ---------- feature construction ----------

def build_personwise_features(I_samples, C1, C2, C2_extra, C3, C3_extra,
                              alpha_scalar, r_matrix, which_case, M, n, L,
                              sample_to_empirical=False, sam_num=None, seed=None):
    _, _, x_samps = _build_x_from_I(I_samples, M=M, n=n, L=L)
    if which_case == 'DI':
        P = Prob_array_direct_imaging(C1, alpha_scalar, x_samps)
    elif which_case == 'sSPADE':
        P = Prob_array_separate_SPADE(C2, C2_extra, alpha_scalar, x_samps)
    else:
        P = Prob_array_orthogonal_SPADE(C3, C3_extra, alpha_scalar, x_samps)
    if sample_to_empirical:
        P = sample_empirical_matrix(P, sam_num=sam_num, seed=seed)
    Z = (P.T @ r_matrix)
    return Z

# =========================
# CACHED BUILDERS FOR KERNELS
# =========================

def get_C1_cached(n, M, N_measure, L_measure, y0, sigma, verbose=True):
    params = dict(n=n, M=M, N_measure=N_measure, L_measure=L_measure,
                  y0=np.asarray(y0, dtype=float), sigma=float(sigma))
    path = _cache_path("C1_directimaging", params)
    loaded = _try_load_arrays(path)
    if loaded is not None and "C1" in loaded:
        if verbose:
            print(f"[cache] Loaded C1 from {path.name}")
        return loaded["C1"]
    if verbose:
        print("[cache] Computing C1 ...")
    C1 = C_directimaging_coeff_integral(n, M, N_measure, L_measure, np.asarray(y0, dtype=float), sigma)
    _save_arrays(path, C1=C1, meta=json.dumps(_as_jsonable(params), sort_keys=True))
    if verbose:
        print(f"[cache] Saved C1 to {path.name}")
    return C1


def get_C2_cached(n, M, nmax, sigma, L, Lmax, N, y0, psi_array, vectors, verbose=True):
    params = dict(n=n, M=M, nmax=nmax, sigma=float(sigma),
                  L=float(L), Lmax=float(Lmax), N=int(N),
                  y0=np.asarray(y0, dtype=float))
    path = _cache_path("C2_sSPADE", params)
    loaded = _try_load_arrays(path)
    if loaded is not None and "C2" in loaded and "C2_extra" in loaded:
        if verbose:
            print(f"[cache] Loaded C2 & C2_extra from {path.name}")
        return loaded["C2"], loaded["C2_extra"]
    if verbose:
        print("[cache] Computing C2 & C2_extra ...")
    product2 = inner_product_directSPADE(vectors, nmax, M)
    a2 = a_coeff_fun(product2, M, nmax, sigma)
    C2 = C_SPADE_coeff(n, M, a2)
    C2_extra = C_SPADE_coeff_extra(n, M, a2)
    _save_arrays(path, C2=C2, C2_extra=C2_extra, meta=json.dumps(_as_jsonable(params), sort_keys=True))
    if verbose:
        print(f"[cache] Saved C2 & C2_extra to {path.name}")
    return C2, C2_extra


def get_C3_cached(n, M, nmax, sigma, L, Lmax, N, y0, psi_array, vectors, verbose=True):
    params = dict(n=n, M=M, nmax=nmax, sigma=float(sigma),
                  L=float(L), Lmax=float(Lmax), N=int(N),
                  y0=np.asarray(y0, dtype=float))
    path = _cache_path("C3_oSPADE", params)
    loaded = _try_load_arrays(path)
    if loaded is not None and "C3" in loaded and "C3_extra" in loaded:
        if verbose:
            print(f"[cache] Loaded C3 & C3_extra from {path.name}")
        return loaded["C3"], loaded["C3_extra"]
    if verbose:
        print("[cache] Computing C3 & C3_extra ...")
    product3 = inner_product(vectors)
    a3 = a_coeff_fun(product3, M, nmax, sigma)
    C3 = C_coeff(n, M, a3)
    C3_extra = C_new_coeff_extra(n, M, a3)
    _save_arrays(path, C3=C3, C3_extra=C3_extra, meta=json.dumps(_as_jsonable(params), sort_keys=True))
    if verbose:
        print(f"[cache] Saved C3 & C3_extra to {path.name}")
    return C3, C3_extra


def build_measurement_kernels_cached(n, M, nmax, sigma, L, Lmax, N, y0, N_measure, L_measure, verbose=True):
    psi_array = psi_array_fun(L, Lmax, sigma, N, M, nmax, y0)
    vectors = [psi_array[:, i, j] for j in range(nmax+1) for i in range(M)]
    C1 = get_C1_cached(n=n, M=M, N_measure=N_measure, L_measure=L_measure, y0=y0, sigma=sigma,
                       verbose=verbose)
    C2, C2_extra = get_C2_cached(n=n, M=M, nmax=nmax, sigma=sigma, L=L, Lmax=Lmax, N=N, y0=y0,
                                 psi_array=psi_array, vectors=vectors, verbose=verbose)
    C3, C3_extra = get_C3_cached(n=n, M=M, nmax=nmax, sigma=sigma, L=L, Lmax=Lmax, N=N, y0=y0,
                                 psi_array=psi_array, vectors=vectors, verbose=verbose)
    return C1, C2, C2_extra, C3, C3_extra

# =========================
# SLOPE FITTING HELPER
# =========================

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

# =========================
# Worker for classification
# =========================

def _clf_job(case, K_keep, Ztr_full, Zte_full, ytr, yte):
    if K_keep <= 0:
        return float('nan')
    try:
        Xtr = Ztr_full[:, -K_keep:]
        Xte = Zte_full[:, -K_keep:]
        scaler = np.mean(np.abs(Xtr), axis=0)
        scaler[scaler == 0] = 1.0
        Xtr = Xtr / scaler
        Xte = Xte / scaler
        clf = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs', max_iter=5000, n_jobs=1)
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        acc = float(np.mean(ypred == yte))
        return acc
    except Exception:
        return float('nan')

# =========================
# PROCESS INITIALIZER (runs once per worker)
# =========================

def _worker_init_shared(L, Lmax, sigma, N, M, nmax, n,
                        N_measure, L_measure,
                        NUM_PERSONS, y0, verbose=False):
    """
    Each process builds (or loads) Olivetti, measurement kernels, and stores in _SHARED.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    global _SHARED
    X1d_all, persons_all = _olivetti_flat_signals(num_persons=NUM_PERSONS)
    C1, C2, C2_extra, C3, C3_extra = build_measurement_kernels_cached(
        n=n, M=M, nmax=nmax, sigma=sigma,
        L=L, Lmax=Lmax, N=N, y0=y0,
        N_measure=N_measure, L_measure=L_measure,
        verbose=verbose
    )
    _SHARED = dict(
        L=L, M=M, n=n,
        N_measure=N_measure,
        X1d_all=X1d_all, persons_all=persons_all,
        C1=C1, C2=C2, C2_extra=C2_extra, C3=C3, C3_extra=C3_extra
    )
    if verbose:
        print("[worker] Initialized shared state.")

# =========================
# ONE (alpha, repeat) JOB
# =========================

def _alpha_repeat_job(payload):
    """
    payload: (idx_ai, a_val, rep, K_sweep, sam_nums, k_train_per_person, NUM_PERSONS)
    Uses global _SHARED initialized in _worker_init_shared.
    Returns:
      - idx_ai
      - results_case_S[case][S_test] = array of accuracies over K_sweep
      - CT_cases[case][S_test] = CT value for this (alpha, repeat, case, S_test)
    """
    (idx_ai, a_val, rep, K_sweep, sam_nums,
     k_train_per_person, NUM_PERSONS) = payload

    global _SHARED
    L = _SHARED['L']
    M = _SHARED['M']
    n = _SHARED['n']
    N_measure = _SHARED['N_measure']
    X1d_all = _SHARED['X1d_all']
    persons_all = _SHARED['persons_all']
    C1 = _SHARED['C1']
    C2 = _SHARED['C2']
    C2_extra = _SHARED['C2_extra']
    C3 = _SHARED['C3']
    C3_extra = _SHARED['C3_extra']

    seed = 1234 + rep
    train_idx, test_idx = split_train_test_per_person(
        persons_all,
        num_persons=NUM_PERSONS,
        k_train_per_person=k_train_per_person,
        seed=seed
    )
    I_train = X1d_all[train_idx, :]
    I_test  = X1d_all[test_idx, :]

    d, g, _ = _build_x_from_I(I_train, M=M, n=n, L=L)

    # Precompute D^(-1/2), G, D for each measurement (independent of S)
    # --- Direct Imaging ---
    Dhi_DI = Dhalfinv_directimaging_fun(g, d, C1, a_val, n, M, N_measure)
    G_DI, D_DI = GD_directimaging(g, d, C1, a_val, n, M, N_measure)

    # --- Separate SPADE ---
    Dhi_s = Dhalfinv_fun(g, d, C2, a_val, n, M, C2_extra)
    G_s, D_s = GD(g, d, C2, a_val, n, M, C2_extra)

    # --- Orthogonal SPADE ---
    Dhi_o = Dhalfinv_fun(g, d, C3, a_val, n, M, C3_extra)
    G_o, D_o = GD(g, d, C3, a_val, n, M, C3_extra)

    ytr = persons_all[train_idx]
    yte = persons_all[test_idx]

    results_case_S = {
        case: {S_test: np.zeros(len(K_sweep), dtype=float) for S_test in sam_nums}
        for case in ['DI', 'sSPADE', 'oSPADE']
    }

    # Store CT for each case and S_test
    CT_cases = {
        case: {S_test: None for S_test in sam_nums}
        for case in ['DI', 'sSPADE', 'oSPADE']
    }

    for S_test in sam_nums:
        # For this S_test, use S_eig = S_test in eigensystem
        _, _, _, v_DI, CT_DI = _eig_main(G_DI, D_DI, Dhi_DI, S_test)
        r_DI = Dhi_DI @ v_DI

        _, _, _, v_s, CT_s = _eig_main(G_s, D_s, Dhi_s, S_test)
        r_sSPADE = Dhi_s @ v_s

        _, _, _, v_o, CT_o = _eig_main(G_o, D_o, Dhi_o, S_test)
        r_oSPADE = Dhi_o @ v_o

        r_bank = {'DI': r_DI, 'sSPADE': r_sSPADE, 'oSPADE': r_oSPADE}
        CT_cases['DI'][S_test] = float(CT_DI)
        CT_cases['sSPADE'][S_test] = float(CT_s)
        CT_cases['oSPADE'][S_test] = float(CT_o)

        # Build features for this S_test (S_eig = S_test) and case
        Ztr_full_case = {}
        for case in ['DI', 'sSPADE', 'oSPADE']:
            Ztr_full_case[case] = build_personwise_features(
                I_train, C1, C2, C2_extra, C3, C3_extra,
                a_val, r_bank[case], case, M, n, L,
                sample_to_empirical=False
            )

        Zte_full_case = {}
        for case in ['DI', 'sSPADE', 'oSPADE']:
            Zte_full_case[case] = build_personwise_features(
                I_test, C1, C2, C2_extra, C3, C3_extra,
                a_val, r_bank[case], case, M, n, L,
                sample_to_empirical=True, sam_num=S_test, seed=98765 + rep
            )

        for case in ['DI', 'sSPADE', 'oSPADE']:
            Ztr_full = Ztr_full_case[case]
            Zte_full = Zte_full_case[case]
            for idx_k, K_keep in enumerate(K_sweep):
                acc = _clf_job(case, K_keep, Ztr_full, Zte_full, ytr, yte)
                results_case_S[case][S_test][idx_k] = acc

    return idx_ai, results_case_S, CT_cases

# =========================
# COMPUTATION (submit ALL tasks at once)
# =========================

def compute_peaks_faces_save(
    alpha, alpha_indices,
    K_sweep, sam_nums, N_REPEATS,
    NUM_PERSONS,
    M, n, L,
    Lmax, sigma, N, nmax, y0,
    N_measure, L_measure,
    save_dir='results_faces',
    save_name='peak_success_vs_alpha_faces.pkl',
    k_train_per_person=9,
    max_workers=None
):
    """
    Submit *all* (alpha_index, repeat) jobs at once to a ProcessPool
    with shared state per worker.

    Here S_eig is identified with each S_test in sam_nums: for every S_test,
    the eigensystem and CT are computed using S = S_test.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(save_dir) / save_name)

    alpha_indices = list(alpha_indices)
    alpha_grid = alpha[alpha_indices]

    peaks = {case: {S: np.zeros(len(alpha_indices)) for S in sam_nums}
             for case in ['DI', 'sSPADE', 'oSPADE']}
    mean_curves = {case: {S: [np.full(len(K_sweep), np.nan) for _ in alpha_indices]
                          for S in sam_nums}
                   for case in ['DI', 'sSPADE', 'oSPADE']}

    acc_lists = {
        idx_ai: {
            case: {
                S: {K: [] for K in K_sweep}
                for S in sam_nums
            } for case in ['DI', 'sSPADE', 'oSPADE']
        } for idx_ai in range(len(alpha_indices))
    }

    # Store CT values per (alpha_index, case, S_test) over repeats
    CT_lists = {
        idx_ai: {
            case: {S: [] for S in sam_nums}
            for case in ['DI', 'sSPADE', 'oSPADE']
        } for idx_ai in range(len(alpha_indices))
    }

    jobs = []
    for idx_ai, ai in enumerate(alpha_indices):
        a_val = float(alpha[ai])
        for rep in range(N_REPEATS):
            payload = (idx_ai, a_val, rep, K_sweep, sam_nums,
                       k_train_per_person, NUM_PERSONS)
            jobs.append(payload)

    total_jobs = len(jobs)
    print(f"Submitting {total_jobs} (alpha, repeat) jobs to ProcessPool...")

    ctx = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=ctx,
        initializer=_worker_init_shared,
        initargs=(L, Lmax, sigma, N, M, nmax, n,
                  N_measure, L_measure,
                  NUM_PERSONS, y0, False)  # verbose=False → quiet workers
    ) as ex:
        futures = [ex.submit(_alpha_repeat_job, payload) for payload in jobs]
        for fut in tqdm(as_completed(futures), total=total_jobs,
                        unit="job", desc="All α, all repeats"):
            idx_ai, results_case_S, CT_cases = fut.result()
            for case in ['DI', 'sSPADE', 'oSPADE']:
                for S_test in sam_nums:
                    CT_lists[idx_ai][case][S_test].append(float(CT_cases[case][S_test]))
                    acc_vec = results_case_S[case][S_test]
                    for idx_k, K_keep in enumerate(K_sweep):
                        acc_lists[idx_ai][case][S_test][K_keep].append(float(acc_vec[idx_k]))

    for idx_ai, ai in enumerate(alpha_indices):
        for case in ['DI', 'sSPADE', 'oSPADE']:
            for S_test in sam_nums:
                mean_curve = np.array(
                    [np.nanmean(acc_lists[idx_ai][case][S_test][K]) if len(acc_lists[idx_ai][case][S_test][K]) > 0 else np.nan
                     for K in K_sweep],
                    dtype=float
                )
                mean_curves[case][S_test][idx_ai] = mean_curve
                finite_mask = np.isfinite(mean_curve)
                peaks[case][S_test][idx_ai] = float(np.nanmax(mean_curve[finite_mask])) if finite_mask.any() else np.nan

    # Compute mean CT vs alpha for each method and each S_test
    CT_mean = {
        case: {S: np.zeros(len(alpha_indices), dtype=float) for S in sam_nums}
        for case in ['DI', 'sSPADE', 'oSPADE']
    }
    for idx_ai in range(len(alpha_indices)):
        for case in ['DI', 'sSPADE', 'oSPADE']:
            for S_test in sam_nums:
                if len(CT_lists[idx_ai][case][S_test]) > 0:
                    CT_mean[case][S_test][idx_ai] = float(np.mean(CT_lists[idx_ai][case][S_test]))
                else:
                    CT_mean[case][S_test][idx_ai] = float('nan')

    out = {
        'alpha_grid': alpha_grid,
        'alpha_indices': alpha_indices,
        'K_sweep': list(K_sweep),
        'sam_nums': list(sam_nums),
        'N_REPEATS': int(N_REPEATS),
        'NUM_PERSONS': int(NUM_PERSONS),
        'peaks': peaks,
        'mean_curves': mean_curves,
        'CT_mean': CT_mean,  # CT vs alpha for each method and each S_test
        'save_path': out_path,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"Saved face-based peaks to {out_path}")
    return out

# =========================
# PLOTTING
# =========================

def plot_peaks_from_file(pkl_or_data, fig_prefix='peak_faces'):
    if isinstance(pkl_or_data, dict):
        dat = pkl_or_data
    else:
        with open(pkl_or_data, 'rb') as f:
            dat = pickle.load(f)

    alpha_grid = np.asarray(dat['alpha_grid'], dtype=float)
    sam_nums = dat['sam_nums']
    peaks = dat['peaks']

    for case_name, title in [('DI', 'DI'),
                             ('sSPADE', 'Separate SPADE'),
                             ('oSPADE', 'Orthogonal SPADE')]:

        plt.figure()
        ax = plt.subplot(1, 1, 1)
        for side in ('bottom', 'left', 'top', 'right'):
            ax.spines[side].set_linewidth(2)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        for S_test in sam_nums:
            yvals = np.asarray(peaks[case_name][S_test], dtype=float)
            safe = np.clip(1.0 - yvals, 1.0e-12, 1.0)
            yplot = -np.log(safe)

            sc = ax.scatter(alpha_grid, yplot,
                            s=40,
                            marker='o',
                            zorder=3,
                            label=f'{S_test:.0e}')

            fit = _fit_loglog_slope(alpha_grid, yplot)
            if fit is not None:
                m, b, mask = fit
                xfit = alpha_grid[mask]
                yfit = (10.0 ** b) * (xfit ** m)

                color = sc.get_facecolors()[0] if sc.get_facecolors().size > 0 else None
                ax.plot(xfit, yfit,
                        linestyle='--',
                        linewidth=1.8,
                        color=color)

                sc.set_label(f'{S_test:.0e} (slope={m:.2f})')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.set_ylabel(r'-\log(1-\mathrm{peak})$', fontsize=20)
        ax.set_title('Peak Success vs ' + r'$\alpha$' + f' — {title}', fontsize=22)

        ax.legend(fontsize=14, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                  ncol=len(sam_nums))
        plt.gcf().savefig(f'{fig_prefix}_{case_name}_peak_vs_alpha.jpg',
                          dpi=300, bbox_inches='tight')
        plt.show()


def plot_CT_from_file(pkl_or_data, fig_prefix='CT_faces'):
    """
    Plot CT as a function of alpha.
    If CT_mean is stored per S_test (new format), we plot one curve per S_test.
    If it's stored as a single array (old format), we plot just one curve.
    """
    if isinstance(pkl_or_data, dict):
        dat = pkl_or_data
    else:
        with open(pkl_or_data, 'rb') as f:
            dat = pickle.load(f)

    if 'CT_mean' not in dat:
        print("No CT_mean in data; nothing to plot.")
        return

    alpha_grid = np.asarray(dat['alpha_grid'], dtype=float)
    CT_mean = dat['CT_mean']
    sam_nums = dat.get('sam_nums', [])

    for case_name, title in [('DI', 'DI'),
                             ('sSPADE', 'Separate SPADE'),
                             ('oSPADE', 'Orthogonal SPADE')]:

        CT_case = CT_mean[case_name]

        plt.figure()
        ax = plt.subplot(1, 1, 1)
        for side in ('bottom', 'left', 'top', 'right'):
            ax.spines[side].set_linewidth(2)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)

        if isinstance(CT_case, dict):
            # New format: CT_case[S_test] is an array vs alpha
            for S_test in sam_nums:
                yvals = np.asarray(CT_case[S_test], dtype=float)
                ax.plot(alpha_grid, yvals, marker='o', label=f'{S_test:.0e}')
            ax.legend(fontsize=14, loc='lower center', bbox_to_anchor=(0.5, 1.02),
                      ncol=len(sam_nums) if len(sam_nums) > 0 else 1)
        else:
            # Old format: single array
            yvals = np.asarray(CT_case, dtype=float)
            ax.plot(alpha_grid, yvals, marker='o')

        ax.set_xscale('log')
        ax.set_xlabel(r'$\alpha$', fontsize=20)
        ax.set_ylabel(r'$C_T$', fontsize=20)
        ax.set_title(r'$C_T$ vs $\alpha$' + f' — {title}', fontsize=22)

        plt.gcf().savefig(f'{fig_prefix}_{case_name}_CT_vs_alpha.jpg',
                          dpi=300, bbox_inches='tight')
        plt.show()

# =========================
# MAIN EXAMPLE
# =========================

if __name__ == '__main__':
    freeze_support()

    L = 10
    Lmax = 20
    sigma = 1
    N = 2000
    M = 3  # 3
    nmax = 150
    n = 20
    N_measure = 60
    L_measure = L + 10*sigma

    Q = 30
    alpha = np.logspace(-3, -0.2, Q)
    alpha_indices = list(range(Q))

    NUM_PERSONS = 20
    N_REPEATS = 100

    K_max_common = min(N_measure, 2*n*M + M)
    K_sweep = list(range(1, min(21, K_max_common + 1)))
    sam_nums = [int(1e10)]  # can be a list of several S_test; S_eig = each S_test

    # y0 = np.array([-0.1, 0.1, 0.2]) * L
    #y0 = np.array([0]) * L
    y0 = np.array([-0.25,-0.05, 0.2]) * L


    results = compute_peaks_faces_save(
        alpha=alpha,
        alpha_indices=alpha_indices,
        K_sweep=K_sweep,
        sam_nums=sam_nums,
        N_REPEATS=N_REPEATS,
        NUM_PERSONS=NUM_PERSONS,
        M=M, n=n, L=L,
        Lmax=Lmax, sigma=sigma, N=N, nmax=nmax, y0=y0,
        N_measure=N_measure, L_measure=L_measure,
        save_dir='results_faces',
        save_name='peak_success_vs_alpha_faces.pkl',
        k_train_per_person=9,
        max_workers=None  # or set to e.g. os.cpu_count()-1
    )

    plot_peaks_from_file(results, fig_prefix='peak_faces')
    plot_CT_from_file(results, fig_prefix='CT_faces')
