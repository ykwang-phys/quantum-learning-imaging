#!/usr/bin/env python3

# ---------------- User-tunable knobs (early) ----------------
BLOCK_SIZE = 40_000                 # streamed test block size
USE_VECTOR_DI = True                # fast DI probability builder (matrix multiply)
MAX_WORKERS = max(1, __import__('os').cpu_count() - 1)
LRT_RUN=True
# ------------------------------------------------------------

# ---- Cap parent BLAS threads BEFORE numpy/scipy import ----
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")

# Standard imports
import os, json, hashlib, pickle, math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import integrate
from scipy.linalg import qr, eigh, inv
from scipy.special import hermite
from scipy.optimize import minimize_scalar
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context, freeze_support, set_start_method
from sklearn.linear_model import LogisticRegression

# =========================
# ProcessPool helpers
# =========================
def _worker_init():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

def _process_pool(max_workers=None):
    ctx = get_context("spawn")
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_worker_init)

# =========================
# Cache helpers
# =========================
_CACHE_DIR = Path("cache_kernels")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _as_jsonable(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    if isinstance(obj, (list, tuple)): return [_as_jsonable(x) for x in obj]
    if isinstance(obj, dict): return {k: _as_jsonable(v) for k, v in obj.items()}
    return obj

def _hash_params(name, params_dict):
    payload = {"name": name, "params": _as_jsonable(params_dict)}
    payload_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload_bytes).hexdigest()[:16]

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
    return Q.T @ V

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
    return (1 / (2 * np.pi * sigma**2)**0.25) * ((-1)**n) * Hn((y - y0) / (2 * sigma)) \
           * np.exp(-(y - y0)**2 / (4 * sigma**2)) / (2 * sigma)**n

def psi_array_fun(L, Lmax, sigma, N, M, nmax, y0):
    result = np.zeros([N, M, nmax + 1])
    y = np.linspace(-Lmax/2, Lmax/2, N)
    for i in range(nmax + 1):
        for j in range(M):
            result[:, j, i] = psi_fun(y, y0[j], sigma, i) * np.sqrt(Lmax / N)
    return result

def a_coeff_fun(product, M, nmax, sigma):
    n = nmax + 1
    result = np.zeros([n, M, n, M])
    for i1 in range(n):
        for i2 in range(n):
            for j1 in range(M):
                for j2 in range(M):
                    right = i1 * M + j1
                    left  = i2 * M + j2
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
                        result[l, 2*j, p, k]   += a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l, 2*j, p, k]   += a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
                        result[l, 2*j+1, p, k] += a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l, 2*j+1, p, k] -= a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
    return result / 4.0

def C_SPADE_coeff(n, M, a):
    result = np.zeros([n, 2*M, n, M])
    for l in range(n):
        for p in range(n):
            for j in range(M):
                for k in range(M):
                    for m in range(p+1):
                        s = p - m
                        result[l, 2*j, p, k]   += a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l, 2*j, p, k]   += a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
                        result[l, 2*j+1, p, k] += a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l, 2*j+1, p, k] -= a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
    return result / (2.0 * M * 2.0)

def C_SPADE_coeff_extra(n, M, a):
    result = np.zeros([M, n, M])
    for p in range(n):
        for q in range(M):
            for j in range(M):
                for m in range(p+1):
                    s = p - m
                    result[j, p, q] += a[m, q, 0, j] * a[s, q, 0, j].conjugate()
    return result / (2.0 * M)

def C_new_coeff_extra(n, M, a):
    result = np.zeros([M, n, M])
    for p in range(n):
        for q in range(M):
            for j in range(M):
                for m in range(p+1):
                    s = p - m
                    result[j, p, q] += a[m, q, 0, j] * a[s, q, 0, j].conjugate()
    return result / 2.0

def C_directimaging_coeff_integral(n, M, N_measure, L_measure, y0, sigma):
    u_temp = np.linspace(-L_measure/2, L_measure/2, N_measure+1)
    u_array = (u_temp[:-1] + u_temp[1:]) / 2
    l = L_measure / N_measure
    result = np.zeros([N_measure, n, M])

    def integrand(u, y0, sigma, p, q, i, k):
        return psi_fun(u, y0, sigma, p) * np.conjugate(psi_fun(u, y0, sigma, q)) \
               / math.factorial(p) / math.factorial(q) * sigma**i

    for i in range(n):
        for k in range(M):
            for p in range(i+1):
                q = i - p
                temp = np.zeros(N_measure)
                for j in range(N_measure):
                    temp[j], _ = integrate.quad(integrand,
                                                u_array[j]-l/2, u_array[j]+l/2,
                                                args=(y0[k], sigma, p, q, i, k))
                result[:, i, k] += temp.astype(np.float64)
    return result

# =========================
# Probabilities (three measurements)
# =========================
def Prob_array_direct_imaging(C1, alpha, x_array):
    (M, n, W_) = np.shape(x_array)
    (N_measure, nC, M_) = np.shape(C1); assert nC == n and M_ == M
    out = np.zeros([N_measure, W_])
    for i in range(W_):
        res = np.zeros(N_measure)
        for j in range(n):
            for k in range(M):
                res += C1[:, j, k] * x_array[k, j, i] * alpha**j
        out[:, i] = res
    return out

def Prob_array_direct_imaging_fast(C1, alpha, x_array):
    Nm, n, M = C1.shape
    assert x_array.shape[0] == M and x_array.shape[1] == n
    a_pows = (alpha ** np.arange(n, dtype=np.float32))               # (n,)
    C1a = (C1.astype(np.float32) * a_pows[None, :, None]).reshape(Nm, n*M)
    X = x_array.transpose(1, 0, 2).reshape(n*M, x_array.shape[2]).astype(np.float32)
    return C1a @ X

def Prob_array_separate_SPADE(C2, C2_extra, alpha, x_array):
    (M, n, W_) = np.shape(x_array)
    (n2, M2, _, M3) = np.shape(C2); assert n2 == n and M2 == 2*M and M3 == M
    out = np.zeros([(n*2*M+M), W_])
    for i in range(W_):
        v = np.zeros(n*2*M+M)
        for j in range(n):
            for k in range(M*2):
                for m in range(M):
                    v[k + j*2*M] += C2[j, k, 0, m] * x_array[m, 0, i]
                    for l in range(1, n):
                        v[k + j*2*M] += C2[j, k, l, m] * x_array[m, l, i] * (alpha**l)
        for k in range(M):
            for m in range(M):
                for l in range(n):
                    v[k + n*2*M] += C2_extra[k, l, m] * x_array[m, l, i] * (alpha**l)
        out[:, i] = v
    return out

def Prob_array_orthogonal_SPADE(C3, C3_extra, alpha, x_array):
    (M, n, W_) = np.shape(x_array)
    (n2, M2, _, M3) = np.shape(C3); assert n2 == n and M2 == 2*M and M3 == M
    out = np.zeros([(n*2*M+M), W_])
    for i in range(W_):
        v = np.zeros(n*2*M+M)
        for j in range(n):
            for k in range(M*2):
                for m in range(M):
                    v[k + j*2*M] += C3[j, k, 0, m] * x_array[m, 0, i]
                    for l in range(1, n):
                        v[k + j*2*M] += C3[j, k, l, m] * x_array[m, l, i] * (alpha**l)
        for k in range(M):
            for m in range(M):
                for l in range(n):
                    v[k + n*2*M] += C3_extra[k, l, m] * x_array[m, l, i] * (alpha**l)
        out[:, i] = v
    return out

# =========================
# Robust normalization for empirical sampling
# =========================
def _normalize_probs64(p):
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, 0.0, None)
    s = p.sum()
    if not np.isfinite(s) or s <= 0.0:
        return np.full_like(p, 1.0 / p.size, dtype=np.float64)
    p /= s
    if p.size > 1:
        tail = 1.0 - p[:-1].sum()
        p[-1] = max(tail, 0.0)
        s2 = p.sum()
        if s2 <= 0.0 or not np.isfinite(s2):
            p[:] = 1.0 / p.size
        else:
            p /= s2
            p[-1] = max(1.0 - p[:-1].sum(), 0.0)
    else:
        p[0] = 1.0
    return p

def _normalize_cols64(P):
    P = np.asarray(P, dtype=np.float64)
    P = np.clip(P, 0.0, None)
    colsum = P.sum(axis=0, keepdims=True)
    colsum[(~np.isfinite(colsum)) | (colsum <= 0.0)] = 1.0
    P /= colsum
    if P.shape[0] > 1:
        tail = 1.0 - P[:-1, :].sum(axis=0, keepdims=True)
        P[-1, :] = np.clip(tail, 0.0, None)
        colsum = P.sum(axis=0, keepdims=True)
        P /= colsum
        P[-1, :] = np.maximum(1.0 - P[:-1, :].sum(axis=0, keepdims=True), 0.0)
    else:
        P[0, :] = 1.0
    return P

# =========================
# Empirical probability generation (multinomial / gaussian)
# =========================
_MAX_CHUNK = 1_000_000

def _sample_one_prob_vec_multinomial(prob_vec, sam_num, rng):
    p = _normalize_probs64(prob_vec)
    sam = int(sam_num)
    if sam <= _MAX_CHUNK:
        counts = rng.multinomial(sam, p)
        return counts.astype(np.float32) / sam
    full = sam // _MAX_CHUNK
    rem  = sam - full * _MAX_CHUNK
    counts = np.zeros_like(p, dtype=np.int64)
    for _ in range(full): counts += rng.multinomial(_MAX_CHUNK, p)
    if rem > 0: counts += rng.multinomial(rem, p)
    return counts.astype(np.float32) / sam

def _sample_one_prob_vec_gaussian(prob_vec, sam_num, rng):
    p = _normalize_probs64(prob_vec)
    cov = np.diag(p) - np.outer(p, p)
    cov /= max(float(sam_num), 1.0)
    w, V = eigh(cov)
    w = np.clip(w, 0.0, None)
    z = rng.standard_normal(len(p))
    noise = V @ (np.sqrt(w) * z)
    x = p + noise
    x = np.clip(x, 0.0, None)
    s = x.sum()
    if s <= 0 or not np.isfinite(s):
        x = p
        s = 1.0
    x = x / s
    if x.size > 1:
        x[-1] = max(1.0 - x[:-1].sum(), 0.0)
        x /= x.sum()
        x[-1] = max(1.0 - x[:-1].sum(), 0.0)
    else:
        x[0] = 1.0
    return x.astype(np.float32)

def sample_empirical_matrix(P_matrix, sam_num, mode='multinomial', seed=None):
    """
    mode ∈ {'multinomial','gaussian'}
    Normalize columns in float64 to EXACT 1.0, then sample/perturb.
    """
    P64 = _normalize_cols64(P_matrix)
    rng = np.random.default_rng(seed)
    M_out, W = P64.shape
    out = np.empty((M_out, W), dtype=np.float32)
    if mode == 'gaussian':
        for j in range(W):
            out[:, j] = _sample_one_prob_vec_gaussian(P64[:, j], sam_num, rng)
        return out
    for j in range(W):
        out[:, j] = _sample_one_prob_vec_multinomial(P64[:, j], sam_num, rng)
    return out

# =========================
# Build x directly from *points* (no source grid)
# =========================
def _segment_centers_edges(L, M):
    edges = np.linspace(-L/2, L/2, M+1)
    centers = 0.5*(edges[:-1] + edges[1:])
    l = L / M
    return centers, edges, l

def _x_from_points_single_sample(points, weights, L, M, n):
    y0, edges, l = _segment_centers_edges(L, M)
    x_single = np.zeros((M, n), dtype=float)
    for y, w in zip(points, weights):
        i = np.searchsorted(edges, y, side='right') - 1
        i = min(max(i, 0), M-1)
        dy = (y - y0[i]) / l
        pow_vec = np.array([dy**j for j in range(n)], dtype=float)
        x_single[i, :] += w * pow_vec
    return x_single

def _build_x_from_points_batch(samples_points, samples_weights, L, M, n):
    W = len(samples_points)
    x = np.zeros((M, n, W), dtype=float)
    for w in range(W):
        x[:, :, w] = _x_from_points_single_sample(samples_points[w], samples_weights[w], L, M, n)
    d = np.mean(x, axis=2).T
    Xmat = x.transpose(1, 0, 2).reshape(M*n, W)
    g = (Xmat @ Xmat.T) / W
    return d, g, x

# =========================
# Point samples (no source grid)
# =========================
def _clip_to_fov(y, L, eps=1e-9):
    return float(np.clip(y, -L/2+eps, L/2-eps))

def make_point_samples(W_train_per_class, W_test_per_class, L, mu_sep, sigma_sep, seed=12345):
    rng = np.random.default_rng(seed)

    def gen_block(Wnum, is_double):
        pts_list, wts_list = [], []
        for _ in range(Wnum):
            if not is_double:
                pts = [0.0]; wts = [1.0]
            else:
                d = sigma_sep
                yL = _clip_to_fov(-0.5*d, L)
                yR = _clip_to_fov(+0.5*d, L)
                pts = [yL, yR]; wts = [0.5, 0.5]
            pts_list.append(pts)
            wts_list.append(wts)
        return pts_list, wts_list

    t0_pts, t0_wts = gen_block(W_train_per_class, is_double=False)
    t1_pts, t1_wts = gen_block(W_train_per_class, is_double=True)
    train_points  = t0_pts + t1_pts
    train_weights = t0_wts + t1_wts
    y_train = np.hstack([np.zeros(W_train_per_class, dtype=int),
                         np.ones(W_train_per_class, dtype=int)])

    s0_pts, s0_wts = gen_block(W_test_per_class, is_double=False)
    s1_pts, s1_wts = gen_block(W_test_per_class, is_double=True)
    test_points  = s0_pts + s1_pts
    test_weights = s0_wts + s1_wts
    y_test = np.hstack([np.zeros(W_test_per_class, dtype=int),
                        np.ones(W_test_per_class, dtype=int)])
    return train_points, train_weights, test_points, test_weights, y_train, y_test

# =========================
# CACHED BUILDERS FOR KERNELS
# =========================
def get_C1_cached(n, M, N_measure, L_measure, y0, sigma):
    params = dict(n=n, M=M, N_measure=N_measure, L_measure=L_measure,
                  y0=np.asarray(y0, dtype=float), sigma=float(sigma))
    path = _cache_path("C1_directimaging", params)
    loaded = _try_load_arrays(path)
    if loaded is not None and "C1" in loaded:
        print(f"[cache] Loaded C1 from {path.name}")
        return loaded["C1"]
    print("[cache] Computing C1 ...")
    C1 = C_directimaging_coeff_integral(n, M, N_measure, L_measure, np.asarray(y0, dtype=float), sigma)
    _save_arrays(path, C1=C1, meta=json.dumps(_as_jsonable(params), sort_keys=True))
    print(f"[cache] Saved C1 to {path.name}")
    return C1

def get_C2_cached(n, M, nmax, sigma, L, Lmax, N, y0, psi_array, vectors):
    params = dict(n=n, M=M, nmax=nmax, sigma=float(sigma),
                  L=float(L), Lmax=float(Lmax), N=int(N),
                  y0=np.asarray(y0, dtype=float))
    path = _cache_path("C2_sSPADE", params)
    loaded = _try_load_arrays(path)
    if loaded is not None and "C2" in loaded and "C2_extra" in loaded:
        print(f"[cache] Loaded C2 & C2_extra from {path.name}")
        return loaded["C2"], loaded["C2_extra"]
    print("[cache] Computing C2 & C2_extra ...")
    product2 = inner_product_directSPADE(vectors, nmax, M)
    a2 = a_coeff_fun(product2, M, nmax, sigma)
    C2 = C_SPADE_coeff(n, M, a2)
    C2_extra = C_SPADE_coeff_extra(n, M, a2)
    _save_arrays(path, C2=C2, C2_extra=C2_extra, meta=json.dumps(_as_jsonable(params), sort_keys=True))
    print(f"[cache] Saved C2 & C2_extra to {path.name}")
    return C2, C2_extra

def get_C3_cached(n, M, nmax, sigma, L, Lmax, N, y0, psi_array, vectors):
    params = dict(n=n, M=M, nmax=nmax, sigma=float(sigma),
                  L=float(L), Lmax=float(Lmax), N=int(N),
                  y0=np.asarray(y0, dtype=float))
    path = _cache_path("C3_oSPADE", params)
    loaded = _try_load_arrays(path)
    if loaded is not None and "C3" in loaded and "C3_extra" in loaded:
        print(f"[cache] Loaded C3 & C3_extra from {path.name}")
        return loaded["C3"], loaded["C3_extra"]
    print("[cache] Computing C3 & C3_extra ...")
    product3 = inner_product(vectors)
    a3 = a_coeff_fun(product3, M, nmax, sigma)
    C3 = C_coeff(n, M, a3)
    C3_extra = C_new_coeff_extra(n, M, a3)
    _save_arrays(path, C3=C3, C3_extra=C3_extra, meta=json.dumps(_as_jsonable(params), sort_keys=True))
    print(f"[cache] Saved C3 & C3_extra to {path.name}")
    return C3, C3_extra

# =========================
# Templates & LLR helpers (ALL outcomes)
# =========================
def _template_probs_for_method(which_meas, a_val, L, M, n,
                               C1, C2, C2_extra, C3, C3_extra,
                               sigma_sep):
    # H0: single at 0
    x0 = _x_from_points_single_sample([0.0], [1.0], L, M, n)
    # H1: two at ±sigma_sep/2
    yL, yR = -0.5*sigma_sep, +0.5*sigma_sep
    x1 = _x_from_points_single_sample([yL, yR], [0.5, 0.5], L, M, n)

    x0 = x0[:, :, None]; x1 = x1[:, :, None]

    if which_meas == 'DI':
        p0 = Prob_array_direct_imaging(C1, a_val, x0)[:, 0]
        p1 = Prob_array_direct_imaging(C1, a_val, x1)[:, 0]
    elif which_meas == 'sSPADE':
        p0 = Prob_array_separate_SPADE(C2, C2_extra, a_val, x0)[:, 0]
        p1 = Prob_array_separate_SPADE(C2, C2_extra, a_val, x1)[:, 0]
    else:
        p0 = Prob_array_orthogonal_SPADE(C3, C3_extra, a_val, x0)[:, 0]
        p1 = Prob_array_orthogonal_SPADE(C3, C3_extra, a_val, x1)[:, 0]

    eps = 1e-300
    p0 = np.maximum(np.clip(p0, 0, None), eps)
    p1 = np.maximum(np.clip(p1, 0, None), eps)
    p0 /= p0.sum()
    p1 /= p1.sum()
    return p0, p1

# =========================
# Chernoff information (numerical)
# =========================
def _chernoff_information(p0, p1):
    """
    Compute C = -log min_{t in [0,1]} sum_i p0[i]^t p1[i]^(1-t)
    Stable via logs (natural log).
    """
    p0 = _normalize_probs64(p0)
    p1 = _normalize_probs64(p1)
    logp0 = np.log(p0)
    logp1 = np.log(p1)

    def log_sum_exp_t(t):
        z = t*logp0 + (1.0 - t)*logp1
        return np.log(np.sum(np.exp(z)))

    res = minimize_scalar(log_sum_exp_t, bounds=(0.0, 1.0), method='bounded',
                          options={'xatol': 1e-6, 'maxiter': 500})
    t_star = float(np.clip(res.x if res.success else 0.5, 0.0, 1.0))
    C = -float(log_sum_exp_t(t_star))
    return C, t_star

def compute_chernoff_grid(alpha_grid, methods_list,
                          L, M, n, C1, C2, C2_extra, C3, C3_extra,
                          sigma_sep):
    chernoff_info = {m: np.zeros(len(alpha_grid), dtype=float) for m in methods_list}
    chernoff_t    = {m: np.zeros(len(alpha_grid), dtype=float) for m in methods_list}
    for m in methods_list:
        for i, a_val in enumerate(alpha_grid):
            p0, p1 = _template_probs_for_method(m, float(a_val), L, M, n,
                                                C1, C2, C2_extra, C3, C3_extra,
                                                sigma_sep=sigma_sep)
            C, t_star = _chernoff_information(p0, p1)
            chernoff_info[m][i] = C
            chernoff_t[m][i]    = t_star
    return chernoff_info, chernoff_t

# =========================
# Eigendecomposition helpers (for LR features)
# =========================
def GD(g, d, C, alpha, n, M, C_extra):
    D = np.zeros([n*2*M+M, n*2*M+M])
    G = np.zeros([n*2*M+M, n*2*M+M])
    for m in range(n):
        for k in range(M):
            D += alpha**m * d[m, k] * np.diag(np.append(C[:, :, m, k].flatten(), C_extra[:, m, k]))
    for m in range(n):
        for k1 in range(M):
            for k2 in range(M):
                for m1 in range(m+1):
                    m2 = m - m1
                    left  = np.array([np.append(C[:, :, m1, k1].flatten(), C_extra[:, m1, k1])]).T
                    right = np.array([np.append(C[:, :, m2, k2].flatten(), C_extra[:, m2, k2])])
                    index1 = m1 * M + k1
                    index2 = m2 * M + k2
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
                    left  = np.array([C[:, m1, k1]]).T
                    right = np.array([C[:, m2, k2]])
                    index1 = m1 * M + k1
                    index2 = m2 * M + k2
                    G += alpha**m * g[index1, index2] * left @ right
    return G, D

def Dhalfinv_directimaging_fun(g, d, C, alpha, n, M, N_measure):
    result = np.zeros(N_measure)
    for m in range(n):
        for k in range(M):
            result += alpha**m * d[m, k] * C[:, m, k]
    result[result <= 0] = np.maximum(result[result <= 0], 1e-14)
    return np.diag(1 / np.sqrt(result))

def Dhalfinv_fun(g, d, C, alpha, n, M, C_extra):
    result = np.zeros(n*2*M+M)
    for m in range(n):
        for k in range(M):
            result += alpha**m * d[m, k] * (np.append(C[:, :, m, k].flatten(), C_extra[:, m, k]))
    result[result <= 0] = np.maximum(result[result <= 0], 1e-14)
    return np.diag(1 / np.sqrt(result))

def main_eig(G, D, Dhalfinv, S):
    M_ = Dhalfinv @ G @ Dhalfinv
    l, v = eigh(M_)
    idx = np.argsort(np.abs(np.real(l)))
    l_sorted = l[idx]
    v_sorted = v[:, idx]
    M1 = G + (D - G) / S
    M2 = inv(M1) @ G
    CT = np.matrix.trace(M2)
    return Dhalfinv, M_, l_sorted, v_sorted, CT

# =========================
# Feature builder (probabilities -> eigentask projections)
# =========================
def build_features_from_x(x_samps, C1, C2, C2_extra, C3, C3_extra,
                          alpha_scalar, r_matrix, which_meas):
    if which_meas == 'DI':
        P = Prob_array_direct_imaging_fast(C1, alpha_scalar, x_samps) if USE_VECTOR_DI \
            else Prob_array_direct_imaging(C1, alpha_scalar, x_samps)
    elif which_meas == 'sSPADE':
        P = Prob_array_separate_SPADE(C2, C2_extra, alpha_scalar, x_samps)
    else:
        P = Prob_array_orthogonal_SPADE(C3, C3_extra, alpha_scalar, x_samps)
    return (P.T @ r_matrix)

# =========================
# WORKER A — LRT with ALL outcomes (empirical rel.freqs)
# =========================
def _worker_lrt(ai, a_val, rep, which_meas,
                L, mu_sep, sigma_sep,
                Wtr_class, Wte_class,
                M, n, C1, C2, C2_extra, C3, C3_extra,
                S_eig, N_measure,
                sam_nums,
                emp_mode='multinomial',
                multinomial_max_S=int(1e12),
                seed_base=12345, seed_emp=98765,
                block_size=50_000):
    """
    LRT using ALL measurement outcomes (equal priors).
    Returns (ai, which_meas, peaks_per_S) — single accuracy per S.
    """
    # Build the full test label vector once
    _, _, te_pts_all, te_wts_all, _, _ = make_point_samples(Wtr_class, Wte_class, L, mu_sep, sigma_sep, seed=seed_base+rep)

    # Templates and log-ratio
    p0, p1 = _template_probs_for_method(which_meas, a_val, L, M, n,C1, C2, C2_extra, C3, C3_extra,sigma_sep=sigma_sep)
    log_ratio = np.log(p1) - np.log(p0)

    def gen_test_blocks():
        blocks = []
    
        remaining = Wte_class
        idx = 0
        while remaining > 0:
            bsz = min(block_size, remaining)
            blocks.append((te_pts_all[idx:idx+bsz],te_wts_all[idx:idx+bsz],0))
            remaining -= bsz
            idx += bsz
    
        remaining = Wte_class
        idx = Wte_class
        while remaining > 0:
            bsz = min(block_size, remaining)
            blocks.append((te_pts_all[idx:idx+bsz],te_wts_all[idx:idx+bsz],1))
            remaining -= bsz
            idx += bsz
    
        return blocks

    peaks_per_S = {}

    for S in sam_nums:
        mode_eff = ('gaussian' if (emp_mode == 'gaussian' or S > multinomial_max_S) else 'multinomial')
        correct = 0
        total   = 0

        for pts_blk, wts_blk, cls_label in gen_test_blocks():
            if not pts_blk:
                continue
            bsz = len(pts_blk)

            # x for this block only
            _, _, x_blk = _build_x_from_points_batch(pts_blk, wts_blk, L, M, n)

            # Ideal probabilities for each sample in this block
            if which_meas == 'DI':
                P_blk = Prob_array_direct_imaging(C1, a_val, x_blk)
            elif which_meas == 'sSPADE':
                P_blk = Prob_array_separate_SPADE(C2, C2_extra, a_val, x_blk)
            else:
                P_blk = Prob_array_orthogonal_SPADE(C3, C3_extra, a_val, x_blk)
            P_emp_blk = sample_empirical_matrix(P_blk, sam_num=S, mode=mode_eff, seed=seed_emp+rep)

            # LRT: dot of empirical rel.freqs with log-ratio
            llr_blk = (P_emp_blk.T @ log_ratio).astype(np.float64)  # (bsz,)
            preds_blk = (llr_blk > 0.0).astype(int)

            if cls_label == 1:
                correct += int(preds_blk.sum())
            else:
                correct += int(bsz - preds_blk.sum())

            total   += bsz

        acc = correct / float(total if total > 0 else 1)
        peaks_per_S[S] = acc

    return ai, which_meas, peaks_per_S

# =========================
# WORKER B — Logistic Regression on eigentask features
# =========================

def _worker_logreg(ai, a_val, rep, which_meas,
                   L, mu_sep, sigma_sep,
                   Wtr_class, Wte_class,
                   M, n, C1, C2, C2_extra, C3, C3_extra,
                   S_eig, N_measure,
                   K_sweep, sam_nums,
                   emp_mode='multinomial',
                   multinomial_max_S=int(1e12),
                   seed_base=12345, seed_emp=98765,
                   block_size=50_000):
    """
    Pure logistic regression (no LLR anywhere).
    Strategy for few samples:
      - Build eigentask features Z = (P^T r_use)
      - Create synthetic augmentation by sampling around the *training prototypes'*
        probability vectors (one per class here), at S close to test S
      - Tune C per-K on augmented holdout
    Returns (ai, which_meas, peaks_per_S, curves_per_S).
    """
    # 1) Data
    tr_pts, tr_wts, te_pts, te_wts, ytr, yte = make_point_samples(Wtr_class, Wte_class, L, mu_sep, sigma_sep, seed=seed_base+rep)
    d, g, x_train = _build_x_from_points_batch(tr_pts, tr_wts, L, M, n)

    # 2) Eigens and r_use
    if which_meas == 'DI':
        G, D = GD_directimaging(g, d, C1, a_val, n, M, N_measure)
        Dhi  = Dhalfinv_directimaging_fun(g, d, C1, a_val, n, M, N_measure)
        Dhi, _, _, vproj, _ = main_eig(G, D, Dhi, S_eig)
        r_use_full = Dhi @ vproj
        n_out = C1.shape[0]
    elif which_meas == 'sSPADE':
        G, D = GD(g, d, C2, a_val, n, M, C2_extra)
        Dhi  = Dhalfinv_fun(g, d, C2, a_val, n, M, C2_extra)
        Dhi, _, _, vproj, _ = main_eig(G, D, Dhi, S_eig)
        r_use_full = Dhi @ vproj
        n_out = C2.shape[1]*n + C2_extra.shape[0]
    else:
        G, D = GD(g, d, C3, a_val, n, M, C3_extra)
        Dhi  = Dhalfinv_fun(g, d, C3, a_val, n, M, C3_extra)
        Dhi, _, _, vproj, _ = main_eig(G, D, Dhi, S_eig)
        r_use_full = Dhi @ vproj
        n_out = C3.shape[1]*n + C3_extra.shape[0]

    K_max = int(max(K_sweep))
    r_use = np.asarray(r_use_full[:, -K_max:], dtype=np.float32, order='C')

    # 3) Training features (unsampled)
    Ztr_full = np.asarray(build_features_from_x(x_train, C1, C2, C2_extra, C3, C3_extra, a_val, r_use, which_meas), dtype=np.float32)

    # 3.1) Get the two *prototype* probability vectors for the training samples
    #      (no LLR; we only use them as centers for augmentation)
    if which_meas == 'DI':
        P_tr = Prob_array_direct_imaging_fast(C1, a_val, x_train) if USE_VECTOR_DI \
               else Prob_array_direct_imaging(C1, a_val, x_train)
    elif which_meas == 'sSPADE':
        P_tr = Prob_array_separate_SPADE(C2, C2_extra, a_val, x_train)
    else:
        P_tr = Prob_array_orthogonal_SPADE(C3, C3_extra, a_val, x_train)
    P_tr = _normalize_cols64(P_tr.astype(np.float64))  # columns sum exactly to 1

    # 3.2) Build augmentation around each prototype at S close to test S
    #      (choose first and last S if multiple to improve robustness)
    S_pool = sorted(set([sam_nums[0], sam_nums[-1]] if len(sam_nums) > 1 else [sam_nums[0]]))
    N_per_class = 2000

    def _augment_from_prototypes(seed=4321):
        Z_list, y_list = [], []
        for iS, S_aug in enumerate(S_pool):
            for cls_idx, cls_label in enumerate(ytr):  # ytr ~ [0,1] with one per class in your setup
                p_center = P_tr[:, cls_idx:cls_idx+1]  # (n_out, 1)
                P_synth  = sample_empirical_matrix(np.repeat(p_center, N_per_class, axis=1),
                                                   int(S_aug), mode='multinomial',
                                                   seed=seed + 11*iS + cls_idx)
                Z_synth  = (P_synth.T @ r_use).astype(np.float32)  # (N, K_max)
                Z_list.append(Z_synth)
                y_list.append(np.full(N_per_class, int(cls_label), dtype=int))
        Z_aug_full = np.vstack(Z_list).astype(np.float32)
        y_aug      = np.hstack(y_list)
        return Z_aug_full, y_aug

    Z_aug_full, y_aug = _augment_from_prototypes()

    # 4) Tune & fit per-K classifiers (pure LR, no LLR features)
    C_grid = [0.01, 0.03, 0.1, 0.3]   # widen if needed: add 0.005, 0.5
    scalers_by_K = {}
    clfs_by_K = {}

    for K in K_sweep:
        if K <= 0:
            continue

        solver = ('liblinear' if K <= 20 else 'lbfgs')

        # Slice last-K coordinates (your convention)
        X_real_K = Ztr_full[:, -K:].astype(np.float32)        # tiny set (≈1 per class)
        X_aug_K  = Z_aug_full[:, -K:].astype(np.float32)      # large synthetic set

        # Robust per-feature scaling from both real+aug
        scaler = np.mean(np.abs(np.vstack([X_real_K, X_aug_K])), axis=0).astype(np.float32)
        scaler[scaler == 0] = 1.0
        X_aug_s  = X_aug_K  / scaler
        X_real_s = X_real_K / scaler

        # 80/20 holdout on augmented data to choose C
        nA = X_aug_s.shape[0]
        cut = int(0.8 * nA)
        XtrA, ytrA = X_aug_s[:cut], y_aug[:cut]
        XvaA, yvaA = X_aug_s[cut:], y_aug[cut:]

        best_C, best_acc, best_clf = None, -1.0, None
        for Cval in C_grid:
            dual_flag = (solver == 'liblinear' and XtrA.shape[0] < XtrA.shape[1])
            clf_try = LogisticRegression(
                C=Cval, penalty='l2', solver=solver,
                max_iter=20000, tol=1e-7,
                fit_intercept=False,
                dual=(dual_flag if solver == 'liblinear' else False)
            )
            clf_try.fit(XtrA, ytrA)
            acc = clf_try.score(XvaA, yvaA)
            if acc > best_acc:
                best_acc, best_C, best_clf = acc, Cval, clf_try

        # Refit on augmented + the tiny real data together
        X_final = np.vstack([X_aug_s, X_real_s]).astype(np.float32)
        y_final = np.hstack([y_aug, ytr])
        dual_flag = (solver == 'liblinear' and X_final.shape[0] < X_final.shape[1])
        clf_final = LogisticRegression(
            C=best_C, penalty='l2', solver=solver,
            max_iter=20000, tol=1e-7,
            fit_intercept=False,
            dual=(dual_flag if solver == 'liblinear' else False)
        )
        clf_final.fit(X_final, y_final)

        scalers_by_K[K] = scaler
        clfs_by_K[K]    = clf_final

    # -------- Test-time evaluation (streamed) --------
    def gen_test_blocks():
        remaining = Wte_class; idx = 0
        while remaining > 0:
            bsz = min(block_size, remaining)
            yield te_pts[idx:idx+bsz], te_wts[idx:idx+bsz], 0
            remaining -= bsz; idx += bsz
        remaining = Wte_class; idx = Wte_class
        while remaining > 0:
            bsz = min(block_size, remaining)
            yield te_pts[idx:idx+bsz], te_wts[idx:idx+bsz], 1
            remaining -= bsz; idx += bsz

    peaks_per_S = {}
    curves_per_S = {}

    for S in sam_nums:
        mode_eff = ('gaussian' if (emp_mode == 'gaussian' or S > multinomial_max_S) else 'multinomial')
        correct_by_K = {K: 0 for K in K_sweep if K > 0}
        total_samples = 0

        for pts_blk, wts_blk, cls_label in gen_test_blocks():
            bsz = len(pts_blk)
            if bsz == 0:
                continue

            _, _, x_blk = _build_x_from_points_batch(pts_blk, wts_blk, L, M, n)

            if which_meas == 'DI':
                P_blk = Prob_array_direct_imaging_fast(C1, a_val, x_blk) if USE_VECTOR_DI \
                        else Prob_array_direct_imaging(C1, a_val, x_blk)
            elif which_meas == 'sSPADE':
                P_blk = Prob_array_separate_SPADE(C2, C2_extra, a_val, x_blk)
            else:
                P_blk = Prob_array_orthogonal_SPADE(C3, C3_extra, a_val, x_blk)
            P_blk = P_blk.astype(np.float32, copy=False)

            P_emp_blk = sample_empirical_matrix(P_blk, sam_num=S, mode=mode_eff, seed=(seed_emp+rep))
            Z_blk = (P_emp_blk.T @ r_use).astype(np.float32)  # (bsz, K_max)
            y_blk = np.full(bsz, cls_label, dtype=int)

            for K in K_sweep:
                if K <= 0:
                    continue
                scaler = scalers_by_K[K]
                X_blk_scaled = Z_blk[:, -K:] / scaler
                yhat = clfs_by_K[K].predict(X_blk_scaled)
                correct_by_K[K] += int((yhat == y_blk).sum())

            total_samples += bsz

        curve = []
        best = -np.inf
        for K in K_sweep:
            if K <= 0:
                curve.append(np.nan); continue
            acc = correct_by_K[K] / float(total_samples if total_samples > 0 else 1)
            curve.append(acc)
            if acc > best: best = acc

        peaks_per_S[S]  = best if best > -0.5 else np.nan
        curves_per_S[S] = np.array(curve, dtype=float)

    return ai, which_meas, peaks_per_S, curves_per_S


# =========================
# Run orchestration
# =========================
def run_combined(
    alpha, alpha_indices, N_REPEATS,
    K_sweep, sam_nums,
    L, mu_sep, sigma_sep, Wtr_class, Wte_class,
    M, n, nmax, Lmax, y0, sigma, N_basis,
    C1, C2, C2_extra, C3, C3_extra,
    S_eig, N_measure,
    methods_list=('DI','sSPADE','oSPADE'),
    save_path='combined_point_nogrid.pkl',
    max_workers=None,
    emp_mode='multinomial',
    multinomial_max_S=int(1e12),
    base_block_size=50_000
):
    alpha_grid = alpha[alpha_indices]

    # --- Chernoff (independent of S) ---
    chernoff_info, chernoff_t = compute_chernoff_grid(
        alpha_grid=alpha_grid, methods_list=methods_list,
        L=L, M=M, n=n, C1=C1, C2=C2, C2_extra=C2_extra, C3=C3, C3_extra=C3_extra,
        sigma_sep=sigma_sep
    )

    # Holders
    peaks_lrt  = {meas: {S: np.zeros(len(alpha_indices)) for S in sam_nums} for meas in methods_list}
    peaks_lr   = {meas: {S: np.zeros(len(alpha_indices)) for S in sam_nums} for meas in methods_list}
    curves_lr  = {meas: {S: [np.full(len(K_sweep), np.nan) for _ in alpha_indices] for S in sam_nums} for meas in methods_list}

    # Temp accumulators for repeats
    tmp_lrt  = {(idx_ai, m, S): [] for idx_ai in range(len(alpha_indices))
                for m in methods_list for S in sam_nums}
    tmp_lr_p = {(idx_ai, m, S): [] for idx_ai in range(len(alpha_indices))
                for m in methods_list for S in sam_nums}
    tmp_lr_c = {(idx_ai, m, S): [] for idx_ai in range(len(alpha_indices))
                for m in methods_list for S in sam_nums}

    # Prepare job list: for each (alpha, repeat, method) run both workers
    jobs_lrt = []
    jobs_lr  = []
    for idx_ai, ai in enumerate(alpha_indices):
        a_val = float(alpha[ai])
        for rep in range(N_REPEATS):
            for which_meas in methods_list:
                jobs_lrt.append((idx_ai, ai, a_val, rep, which_meas))
                jobs_lr.append((idx_ai, ai, a_val, rep, which_meas))

    print(f"Submitting {len(jobs_lrt)} LRT tasks and {len(jobs_lr)} LR tasks.")
    print(f"[empirical mode] {emp_mode}  |  [MAX_WORKERS] {max_workers}  |  [BLOCK_SIZE] {base_block_size}")

    if LRT_RUN==True:
        # Run LRT tasks
        with _process_pool(max_workers=max_workers) as ex:
            futs = []
            for (idx_ai, ai, a_val, rep, which_meas) in jobs_lrt:
                futs.append(
                    ex.submit(
                        _worker_lrt,
                        ai, a_val, rep, which_meas,
                        L, mu_sep, sigma_sep,
                        Wtr_class, Wte_class,
                        M, n, C1, C2, C2_extra, C3, C3_extra,
                        S_eig, N_measure,
                        sam_nums,
                        emp_mode,
                        multinomial_max_S,
                        block_size=base_block_size
                    )
                )
            for fut in tqdm(as_completed(futs), total=len(futs), unit="task", desc="LRT"):
                ai_ret, which_meas_ret, peaks_per_S = fut.result()
                idx_ai = alpha_indices.index(ai_ret)
                for S, val in peaks_per_S.items():
                    tmp_lrt[(idx_ai, which_meas_ret, S)].append(val)

    # Run LR tasks
    with _process_pool(max_workers=max_workers) as ex:
        futs = []
        for (idx_ai, ai, a_val, rep, which_meas) in jobs_lr:
            futs.append(
                ex.submit(
                    _worker_logreg,
                    ai, a_val, rep, which_meas,
                    L, mu_sep, sigma_sep,
                    Wtr_class, Wte_class,
                    M, n, C1, C2, C2_extra, C3, C3_extra,
                    S_eig, N_measure,
                    K_sweep, sam_nums,
                    emp_mode,
                    multinomial_max_S,
                    block_size=base_block_size
                )
            )
        for fut in tqdm(as_completed(futs), total=len(futs), unit="task", desc="LogReg"):
            ai_ret, which_meas_ret, peaks_per_S, curves_per_S = fut.result()
            idx_ai = alpha_indices.index(ai_ret)
            for S, val in peaks_per_S.items():
                tmp_lr_p[(idx_ai, which_meas_ret, S)].append(val)
            for S, arr in curves_per_S.items():
                tmp_lr_c[(idx_ai, which_meas_ret, S)].append(arr)

    # Aggregate repeats
    for idx_ai in range(len(alpha_indices)):
        for m in methods_list:
            for S in sam_nums:
                # LRT
                p_list = [v for v in tmp_lrt[(idx_ai, m, S)] if np.isfinite(v)]
                peaks_lrt[m][S][idx_ai] = float(np.mean(p_list)) if len(p_list) else np.nan
                # LR peaks
                p2_list = [v for v in tmp_lr_p[(idx_ai, m, S)] if np.isfinite(v)]
                peaks_lr[m][S][idx_ai] = float(np.mean(p2_list)) if len(p2_list) else np.nan
                # LR curves
                c_list = [v for v in tmp_lr_c[(idx_ai, m, S)] if np.all(np.isfinite(v))]
                curves_lr[m][S][idx_ai] = np.mean(np.stack(c_list, axis=0), axis=0) if len(c_list) \
                                           else np.full(len(K_sweep), np.nan)

    out = {
        'alpha_grid': alpha_grid,
        'alpha_indices': list(alpha_indices),
        'K_sweep': list(K_sweep),
        'sam_nums': list(sam_nums),
        'N_REPEATS': int(N_REPEATS),
        'methods': list(methods_list),
        'emp_mode': emp_mode,
        # LRT
        'peaks_lrt': peaks_lrt,
        # Logistic Regression
        'peaks_lr': peaks_lr,
        'curves_lr': curves_lr,
        # Chernoff
        'chernoff_info': chernoff_info,
        'chernoff_t': chernoff_t,
    }
    with open(save_path, 'wb') as f:
        pickle.dump(out, f)
    print(f"Saved combined results to {save_path}")
    return out



# =========================
# Main
# =========================
if __name__ == '__main__':
    freeze_support()
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass

    # -------- Scene / sampling --------
    L = 5.0
    mu_sep = 0.0
    sigma_sep = 0.6          
    W_train_per_class = 20
    W_test_per_class  = 10_000

    # -------- SPADE basis / DI measurement --------
    sigma = 1.0
    M = 1
    nmax = 150
    n = 20
    Lmax = 10.0
    y0 = np.array([0.0]) * L

    N_measure = 60
    L_measure = L + 10*sigma
    S_eig = 1000

    # α-grid, K_keep sweep, repeats, sample sizes
    Q =20
    alpha = np.logspace(-2, -1, Q)
    alpha_indices = list(range(Q))

    K_sweep = list(range(1,5))
    sam_nums = [int(1e6)]  #1e7 for DI    
    N_REPEATS = 5  # adjust for runtime vs variance

    # ---- Empirical sampling knobs ----
    EMPIRICAL_MODE    = 'multinomial'   # or 'gaussian'
    #EMPIRICAL_MODE    = 'gaussian'   # or 'gaussian'
    MAX_MULTINOMIAL_S = int(1e12)

    # Build SPADE mode functions and cached kernels
    N_basis = 2000  # ONLY for constructing orthonormal modes
    psi_array = psi_array_fun(L, Lmax, sigma, N_basis, M, nmax, y0)
    vectors = [psi_array[:, i, j] for j in range(nmax+1) for i in range(M)]

    C1 = get_C1_cached(n=n, M=M, N_measure=N_measure, L_measure=L_measure, y0=y0, sigma=sigma)
    C2, C2_extra = get_C2_cached(n=n, M=M, nmax=nmax, sigma=sigma, L=L, Lmax=Lmax, N=N_basis, y0=y0,
                                 psi_array=psi_array, vectors=vectors)
    C3, C3_extra = get_C3_cached(n=n, M=M, nmax=nmax, sigma=sigma, L=L, Lmax=Lmax, N=N_basis, y0=y0,
                                 psi_array=psi_array, vectors=vectors)

    print(f"[config] MAX_WORKERS={MAX_WORKERS}, BLOCK_SIZE={BLOCK_SIZE}")
    results = run_combined(
        alpha=alpha,
        alpha_indices=alpha_indices,
        N_REPEATS=N_REPEATS,
        K_sweep=K_sweep,
        sam_nums=sam_nums,
        L=L, mu_sep=mu_sep, sigma_sep=sigma_sep,
        Wtr_class=W_train_per_class, Wte_class=W_test_per_class,
        M=M, n=n, nmax=nmax, Lmax=Lmax, y0=y0, sigma=sigma, N_basis=N_basis,
        C1=C1, C2=C2, C2_extra=C2_extra, C3=C3, C3_extra=C3_extra,
        S_eig=S_eig, N_measure=N_measure,
        methods_list=('DI','sSPADE','oSPADE'),
        save_path='combined_point_nogrid_SPADE.pkl',
        max_workers=MAX_WORKERS,
        emp_mode=EMPIRICAL_MODE,
        multinomial_max_S=MAX_MULTINOMIAL_S,
        base_block_size=BLOCK_SIZE
    )

