# -*- coding: utf-8 -*-
"""
Created on Fri Nov 14 21:51:54 2025

@author: 27432

Same as face_20251102_7_4_NiceParameterSet
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Nov  3 22:02:23 2025

@author: 27432
"""

# -*- coding: utf-8 -*-
"""

Compared to version 7_2

Try to change the number of compact sources

"""

import sys, os, math, time, pickle, json, hashlib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.linalg import eig, eigh, inv, qr
from scipy import integrate
from scipy.special import hermite
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import fetch_olivetti_faces
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context, freeze_support

# ---------------- ProcessPool helper (Windows/Spyder safe) ----------------
def _worker_init():
    """
    Runs in each child process once. Keep it lightweight.
    Limit BLAS threads to avoid oversubscription when we spawn many workers.
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

def _process_pool(max_workers=None):
    """
    Return a *new instance* of ProcessPoolExecutor using 'spawn' context,
    which is required on Windows and safest in Spyder.
    """
    ctx = get_context("spawn")
    return ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx, initializer=_worker_init)

# ------------- Sampling controls -------------
SAM_NUM_LIST = [int(1e6),int(1e8),int(1e10)]  # sample sizes S for test-time multinomial
_MAX_CHUNK   = 1_000_000    # chunk multinomial draws to avoid overflow

# ---------------- Original helper functions ----------------
def classify_P4(u):
    if -0.50 <= u <= -0.48: return 1
    elif -0.46 <= u <= -0.43: return 1
    elif -0.40 <= u <= -0.37: return 1
    elif -0.33 <= u <= -0.30: return 1
    elif -0.25 <= u <= -0.21: return 1
    elif -0.15 <= u <= -0.12: return 1
    elif -0.05 <= u <= -0.01: return 1
    elif  0.02 <= u <=  0.06: return 1
    elif  0.10 <= u <=  0.15: return 1
    elif  0.19 <= u <=  0.26: return 1
    elif  0.30 <= u <=  0.36: return 1
    elif  0.39 <= u <=  0.43: return 1
    elif  0.48 <= u <=  0.50: return 1
    else: return 0

def psi_fun(y, y0, sigma, n):
    Hn = hermite(n)
    return 1 / (2 * np.pi * sigma**2)**(1/4) * (-1)**n * Hn((y - y0) / (2 * sigma)) * np.exp(-(y - y0)**2 / (4 * sigma**2)) / (2 * sigma)**n

def psi_array_fun(L, Lmax, sigma, N, M, nmax, y0):
    result = np.zeros([N, M, nmax + 1])
    y = np.linspace(-Lmax/2, Lmax/2, N)
    for i in range(nmax + 1):
        for j in range(M):
            result[:, j, i] = psi_fun(y, y0[j], sigma, i)*np.sqrt(Lmax/N)
    return result

def inner_product(vectors):
    V = np.column_stack(vectors)
    Q, R = qr(V, mode='economic')
    return Q.T @ V

def inner_product_directSPADE(vectors, nmax, M):
    V = np.column_stack(vectors)
    product=np.zeros([(nmax+1)*M,(nmax+1)*M])
    for i in range(M):
        vectors_temp=[vectors[i+M*j] for j in range(nmax+1)]
        V_temp = np.column_stack(vectors_temp)
        Q, R = qr(V_temp, mode='economic')
        product_temp = Q.T @ V
        for j in range(nmax+1):
            product[i+M*j,:]=product_temp[j,:]
    return product

def a_coeff_fun(product,M,nmax,sigma):
    n=nmax+1
    result=np.zeros([n,M,n,M])
    for i1 in range(n):
        for i2 in range(n):
            for j1 in range(M):
                for j2 in range(M):
                    right=i1*M+j1
                    left=i2*M+j2
                    result[i1,j1,i2,j2]=product[left,right] * sigma**i1 / math.factorial(i1)
    return result

def C_SPADE_coeff(n,M,a):
    result=np.zeros([n,2*M,n,M])
    for l in range(n):
        for p in range(n):
            for j in range(M):
                for k in range(M):
                    for m in range(p+1):
                        s=p-m
                        result[l,2*j,p,k]+=a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j,p,k]+=a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j+1,p,k]+=a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j+1,p,k]-=a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
    return result/2/M/2

def C_SPADE_coeff_extra(n,M,a):
    result=np.zeros([M,n,M])
    for p in range(n):
        for q in range(M):
            for j in range(M):
                for m in range(p+1):
                    s=p-m
                    result[j,p,q]+=a[m,q,0,j]*a[s,q,0,j].conjugate()
    return result/M/2

def C_coeff(n,M,a):
    result=np.zeros([n,2*M,n,M])
    for l in range(n):
        for p in range(n):
            for j in range(M):
                for k in range(M):
                    for m in range(p+1):
                        s=p-m
                        result[l,2*j,p,k]+=a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j,p,k]+=a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j+1,p,k]+=a[s,k,l,j]*a[m,k,l,j].conjugate()+a[s,k,l+1,j]*a[m,k,l+1,j].conjugate()
                        result[l,2*j+1,p,k]-=a[s,k,l+1,j]*a[m,k,l,j].conjugate()+a[s,k,l,j]*a[m,k,l+1,j].conjugate()
    return result/2/2

def C_new_coeff_extra(n,M,a):
    result=np.zeros([M,n,M])
    for p in range(n):
        for q in range(M):
            for j in range(M):
                for m in range(p+1):
                    s=p-m
                    result[j,p,q]+=a[m,q,0,j]*a[s,q,0,j].conjugate()
    return result/2

def C_directimaging_coeff_integral(n,M,N_measure,L_measure,y0,sigma):
    u_temp = np.linspace(-L_measure/2,L_measure/2,N_measure+1)
    u_array = (u_temp[:-1] + u_temp[1:]) / 2
    l=L_measure/N_measure
    result=np.zeros([N_measure,n,M])
    def integrate_psi(u,y0,sigma,p,q,i,k):
        return psi_fun(u,y0,sigma,p)*np.conjugate(psi_fun(u,y0,sigma,q))/math.factorial(p)/math.factorial(q)*sigma**i
    for i in range(n):
        for k in range(M):
            for p in range(i+1):
                q=i-p
                temp=np.zeros(N_measure)
                for j in range(N_measure):
                    temp[j],_ = integrate.quad(integrate_psi,u_array[j]-l/2,u_array[j]+l/2,args=(y0[k],sigma,p,q,i,k))
                result[:,i,k]+=temp.astype(np.float64)
    return result

def GD_directimaging(g,d,C,alpha,n,M,N_measure):
    D=np.zeros([N_measure,N_measure])
    G=np.zeros([N_measure,N_measure])
    for m in range(n):
        for k in range(M):
            D+=alpha**m * d[m,k] * np.diag(C[:,m,k])
    for m in range(n):
        for k1 in range(M):
            for k2 in range(M):
                for m1 in range(m+1):
                    m2=m-m1
                    left=np.array([C[:,m1,k1]]).T
                    right=np.array([C[:,m2,k2]])
                    index1=m1*M+k1
                    index2=m2*M+k2
                    G+=alpha**m * g[index1,index2] * left@right
    return G,D

def Dhalfinv_directimaging_fun(g,d,C,alpha,n,M,N_measure):
    result=np.zeros(N_measure)
    for m in range(n):
        for k in range(M):
            result+=alpha**m * d[m,k]*C[:,m,k]
    for i in range(N_measure):
        if result[i]<=0: result[i]=max(result[i],1e-14)
    return np.diag(1/np.sqrt(result))

def GD(g,d,C,alpha,n,M,C_extra):
    D=np.zeros([n*2*M+M,n*2*M+M])
    G=np.zeros([n*2*M+M,n*2*M+M])
    for m in range(n):
        for k in range(M):
            D+=alpha**m * d[m,k]*np.diag(np.append(C[:,:,m,k].flatten(),C_extra[:,m,k]))
    for m in range(n):
        for k1 in range(M):
            for k2 in range(M):
                for m1 in range(m+1):
                    m2=m-m1
                    left=np.array([np.append(C[:,:,m1,k1].flatten(),C_extra[:,m1,k1])]).T
                    right=np.array([np.append(C[:,:,m2,k2].flatten(),C_extra[:,m2,k2])])
                    index1=m1*M+k1
                    index2=m2*M+k2
                    G+=alpha**m * g[index1,index2] * left@right
    return G,D

def Dhalfinv_fun(g,d,C,alpha,n,M,C_extra):
    result=np.zeros(n*2*M+M)
    for m in range(n):
        for k in range(M):
            result+=alpha**m * d[m,k]*(np.append(C[:,:,m,k].flatten(),C_extra[:,m,k]))
    for i in range(n*2*M+M):
        if result[i]<=0: result[i]=max(result[i],1e-14)
    return np.diag(1/np.sqrt(result))

def main(G,D,Dhalfinv,S):
    M_ = Dhalfinv@G@Dhalfinv
    l, v =eigh(M_)
    idx = np.argsort(np.abs(np.real(l)))
    sorted_l = l[idx]
    sorted_v = v[:, idx]
    M1=G+(D-G)/S
    M2=inv(M1)@G
    CT=np.matrix.trace(M2)
    return Dhalfinv, M_, sorted_l, sorted_v, CT

def Prob_array_direct_imaging(C1,alpha,x_array):
    (M,n,W_)=np.shape(x_array)
    (N_measure,_,M)=np.shape(C1)
    result=np.zeros([N_measure,W_])
    for  i in range(W_):
        res=np.zeros(N_measure)
        for j in range(n):
            for k in range(M):
                res += C1[:,j,k]*x_array[k,j,i]*alpha**j
        result[:,i]=res
    return result

def Prob_array_separate_SPADE(C2,C2_extra,alpha,x_array):
    (M,n,W_)=np.shape(x_array)
    (n2,M2,_,M3)=np.shape(C2)
    assert n2==n and M2==2*M and M3==M
    out=np.zeros([(n*2*M+M), W_])
    for i in range(W_):
        v = np.zeros(n*2*M+M)
        for j in range(n):
            for k in range(M*2):
                for m in range(M):
                    v[k+j*M*2]+=C2[j,k,0,m]*x_array[m,0,i]*(alpha**0)
                    for l in range(1,n):
                        v[k+j*M*2]+=C2[j,k,l,m]*x_array[m,l,i]*(alpha**l)
        for k in range(M):
            for m in range(M):
                for l in range(n):
                    v[k+n*M*2]+=C2_extra[k,l,m]*x_array[m,l,i]*(alpha**l)
        out[:,i]=v
    return out

def Prob_array_orthogonal_SPADE(C3,C3_extra,alpha,x_array):
    (M,n,W_)=np.shape(x_array)
    out=np.zeros([(n*2*M+M), W_])
    (n2,M2,_,M3)=np.shape(C3)
    assert n2==n and M2==2*M and M3==M
    for i in range(W_):
        v = np.zeros(n*2*M+M)
        for j in range(n):
            for k in range(M*2):
                for m in range(M):
                    v[k+j*M*2]+=C3[j,k,0,m]*x_array[m,0,i]*(alpha**0)
                    for l in range(1,n):
                        v[k+j*M*2]+=C3[j,k,l,m]*x_array[m,l,i]*(alpha**l)
        for k in range(M):
            for m in range(M):
                for l in range(n):
                    v[k+n*M*2]+=C3_extra[k,l,m]*x_array[m,l,i]*(alpha**l)
        out[:,i]=v
    return out

# ---------------- Faces → 1D signals and split ----------------
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

def split_train_test_per_person(persons, num_persons=20, k_train_per_person=9, seed=0):
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

# ---------------- Build (d,g,x) from intensities ----------------
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

# ---------------- Sampling utilities ----------------
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
        counts += rng.multinomial(rem, P)
    return counts.astype(float) / sam

def sample_empirical_matrix(P_matrix, sam_num, seed=None):
    rng = np.random.default_rng(seed)
    M_out, W = P_matrix.shape
    out = np.zeros_like(P_matrix, dtype=float)
    for j in range(W):
        out[:, j] = _sample_one_prob_vec(P_matrix[:, j], sam_num, rng)
    return out

# ---------------- Eigens + features + worker ----------------
def _eig_DI_worker(i, g, d, C1, alpha_i, n, M, N_measure, S):
    G, D = GD_directimaging(g, d, C1, alpha_i, n, M, N_measure)
    Dhalfinv = Dhalfinv_directimaging_fun(g, d, C1, alpha_i, n, M, N_measure)
    D_halfinv_temp, _, evals, evecs, _ = main(G, D, Dhalfinv, S)
    return i, evals, D_halfinv_temp @ evecs

def _eig_SPADE_worker(i, g, d, C, C_extra, alpha_i, n, M, S):
    G, D = GD(g, d, C, alpha_i, n, M, C_extra)
    Dhalfinv = Dhalfinv_fun(g, d, C, alpha_i, n, M, C_extra)
    D_halfinv_temp, _, evals, evecs, _ = main(G, D, Dhalfinv, S)
    return i, evals, D_halfinv_temp @ evecs

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

def scale_train_test_lastK(Xtr_full, Xte_full, K_keep):
    if K_keep <= 0:
        print ('K_keep should be greater than 0')
        
    Xtr = Xtr_full[:, -K_keep:] 
    Xte = Xte_full[:, -K_keep:] 

    scaler = np.mean(np.abs(Xtr), axis=0)
    scaler[scaler == 0] = 1.0
    return Xtr / scaler, Xte / scaler, scaler

def _face_method_worker(payload):
    (method_name, which_case, r_use, I_train, I_test,
     C1, C2, C2_extra, C3, C3_extra, alpha_scalar, M, n, L,
     persons_all, train_idx, test_idx, K_keep, sam_num_test, rep_seed, NUM_PERSONS) = payload

    if K_keep <= 0:
        return {'method_name': method_name, 'top1_acc': float('nan'), 'per_person': {}}

    # Train on theoretical
    Ztr_full = build_personwise_features(I_train, C1, C2, C2_extra, C3, C3_extra,
                                         alpha_scalar, r_use, which_case, M, n, L,
                                         sample_to_empirical=False)
    # Test on sampled
    Zte_full = build_personwise_features(I_test,  C1, C2, C2_extra, C3, C3_extra,
                                         alpha_scalar, r_use, which_case, M, n, L,
                                         sample_to_empirical=True, sam_num=sam_num_test, seed=rep_seed)

    Xtr, Xte, _ = scale_train_test_lastK(Ztr_full, Zte_full, K_keep)
    ytr = persons_all[train_idx]
    yte = persons_all[test_idx]
    #clf = LogisticRegression(C=0.1, penalty='l2',multi_class='multinomial',solver='lbfgs', max_iter=5000)
    clf = LogisticRegression(C=0.1, penalty='l2',solver='lbfgs', max_iter=5000)
    #clf = LogisticRegression(max_iter=3000, solver='lbfgs')
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    top1_acc = float(np.mean(ypred == yte))

    per_person = {}
    for pid in range(NUM_PERSONS):
        mask = (yte == pid)
        if np.any(mask):
            per_person[pid] = float(np.mean(ypred[mask] == yte[mask]))

    return {'method_name': method_name, 'top1_acc': top1_acc, 'per_person': per_person}

# --------------------- Caching utilities ---------------------
_CACHE_DIR = Path("./cache_measurements")
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CACHE_VERSION = "v1"  # bump this if you change what goes into the cache

def _params_for_cache(L, Lmax, sigma, N, M, nmax, n, N_measure, L_measure, y0):
    return {
        "version": _CACHE_VERSION,
        "L": float(L),
        "Lmax": float(Lmax),
        "sigma": float(sigma),
        "N": int(N),
        "M": int(M),
        "nmax": int(nmax),
        "n": int(n),
        "N_measure": int(N_measure),
        "L_measure": float(L_measure),
        "y0": [float(v) for v in np.asarray(y0).ravel().tolist()],
    }

def _fingerprint_params(params_dict):
    """Stable SHA256 over a canonical JSON string of params."""
    blob = json.dumps(params_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

def _cache_path_from_params(params_dict):
    fp = _fingerprint_params(params_dict)
    return _CACHE_DIR / f"measurements_{fp}.pkl"

def load_measurements_if_cached(params_dict):
    """Return (loaded, data_dict). If not cached or mismatch, (False, None)."""
    path = _cache_path_from_params(params_dict)
    if not path.exists():
        return False, None
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Strict param equality check:
        if data.get("params") == params_dict and all(k in data for k in ("C1","C2","C2_extra","C3","C3_extra")):
            print(f"[cache] Loaded measurement matrices from {path.name}")
            return True, data
        else:
            print(f"[cache] Param mismatch in {path.name}, will recompute.")
            return False, None
    except Exception as e:
        print(f"[cache] Failed to load {path.name}: {e}. Will recompute.")
        return False, None

def save_measurements_to_cache(params_dict, C1, C2, C2_extra, C3, C3_extra):
    path = _cache_path_from_params(params_dict)
    data = {
        "params": params_dict,
        "C1": C1, "C2": C2, "C2_extra": C2_extra, "C3": C3, "C3_extra": C3_extra,
        "saved_at": time.time(),
    }
    with open(path, "wb") as f:
        pickle.dump(data, f)
    print(f"[cache] Saved measurement matrices → {path.name}")

# ===================== Main =====================
if __name__ == '__main__':

    freeze_support()

    NUM_PERSONS = 20  


    L = 10
    Lmax = 20
    sigma = 1
    N = 2000
    M = 3
    nmax = 150
    n = 20
    S_eig = 1000
    N_measure = 60
    L_measure = L + 10*sigma
    Q = 2
    alpha = np.logspace(-3, -1, Q)
    which_alpha = -1


    K_sweep = list(range(1, 21))
    N_REPEATS = 20

    t0 = time.time()
    #y0 = np.array([-0.35,-0.15, 0.1]) * L  
    y0 = np.array([-0.25,-0.05, 0.2]) * L
    #y0 = np.array([-0.35,-0.15, 0.1, 0.3]) * L

    # ---------- measurement matrices (cached) ----------
    cache_params = _params_for_cache(L, Lmax, sigma, N, M, nmax, n, N_measure, L_measure, y0)
    cached, cached_data = load_measurements_if_cached(cache_params)

    if cached:
        C1 = cached_data["C1"]
        C2 = cached_data["C2"]
        C2_extra = cached_data["C2_extra"]
        C3 = cached_data["C3"]
        C3_extra = cached_data["C3_extra"]
    else:
        # Build basis and intermediates only if needed
        psi_array = psi_array_fun(L, Lmax, sigma, N, M, nmax, y0)
        vectors = []
        for j in range(nmax + 1):
            for i in range(M):
                vectors.append(psi_array[:, i, j])

        # Compute measurement matrices
        print("[build] Computing C1, C2, C2_extra, C3, C3_extra ...")
        C1 = C_directimaging_coeff_integral(n, M, N_measure, L_measure, y0, sigma)

        product2 = inner_product_directSPADE(vectors, nmax, M)
        a2 = a_coeff_fun(product2, M, nmax, sigma)
        C2 = C_SPADE_coeff(n, M, a2)
        C2_extra = C_SPADE_coeff_extra(n, M, a2)

        product3 = inner_product(vectors)
        a3 = a_coeff_fun(product3, M, nmax, sigma)
        C3 = C_coeff(n, M, a3)
        C3_extra = C_new_coeff_extra(n, M, a3)

        save_measurements_to_cache(cache_params, C1, C2, C2_extra, C3, C3_extra)

    # Faces — keep only first 10 persons
    X1d_all, persons_all = _olivetti_flat_signals(num_persons=NUM_PERSONS)

    methods_list = ['Direct Imaging', 'Separate SPADE', 'Orthogonal SPADE']
    acc_records = {m: {S: {K: [] for K in K_sweep} for S in SAM_NUM_LIST} for m in methods_list}

    out_bundle_last = None
    summary_last_results = None

    print("\n=== Sweep over K_keep (0..20), multiple S, with repetitions — FIRST 10 PERSONS ===")
    for rep in range(N_REPEATS):
        print(f"\n=== Repetition {rep+1}/{N_REPEATS} ===")
        train_idx, test_idx = split_train_test_per_person(persons_all, num_persons=NUM_PERSONS, k_train_per_person=9, seed=1234+rep)
        I_train = X1d_all[train_idx, :]
        I_test  = X1d_all[test_idx, :]

        print("Forming (d,g) from ALL training faces (first 10 persons) ...")
        d, g, _ = _build_x_from_I(I_train, M=M, n=n, L=L)
        print("Done. d shape:", d.shape, "g shape:", g.shape)

        # Eigens (parallel) over alpha
        print("\n[DI] Eigen-decompositions (parallel)…")
        l1 = np.zeros([N_measure, Q])
        r1 = np.zeros([N_measure, N_measure, Q])
        with _process_pool() as ex:
            futures = [ex.submit(_eig_DI_worker, i, g, d, C1, alpha[i], n, M, N_measure, S_eig) for i in range(Q)]
            for fut in tqdm(as_completed(futures), total=Q, unit="α", desc="DI eigens"):
                i, evals, rproj = fut.result()
                l1[:, i] = evals
                r1[:, :, i] = rproj

        print("\n[sSPADE] Eigen-decompositions (parallel)…")
        l2 = np.zeros([2*n*M+M, Q])
        r2 = np.zeros([2*n*M+M, 2*n*M+M, Q])
        with _process_pool() as ex:
            futures = [ex.submit(_eig_SPADE_worker, i, g, d, C2, C2_extra, alpha[i], n, M, S_eig) for i in range(Q)]
            for fut in tqdm(as_completed(futures), total=Q, unit="α", desc="sSPADE eigens"):
                i, evals, rproj = fut.result()
                l2[:, i] = evals
                r2[:, :, i] = rproj

        print("\n[oSPADE] Eigen-decompositions (parallel)…")
        l3 = np.zeros([2*n*M+M, Q])
        r3 = np.zeros([2*n*M+M, 2*n*M+M, Q])
        with _process_pool() as ex:
            futures = [ex.submit(_eig_SPADE_worker, i, g, d, C3, C3_extra, alpha[i], n, M, S_eig) for i in range(Q)]
            for fut in tqdm(as_completed(futures), total=Q, unit="α", desc="oSPADE eigens"):
                i, evals, rproj = fut.result()
                l3[:, i] = evals
                r3[:, :, i] = rproj

        # Choose eigenvectors at selected alpha
        r_DI     = r1[:, :, which_alpha]
        r_sSPADE = r2[:, :, which_alpha]
        r_oSPADE = r3[:, :, which_alpha]
        tasks_base = [
            ('Direct Imaging',   'DI',      np.asarray(r_DI)),
            ('Separate SPADE',   'sSPADE',  np.asarray(r_sSPADE)),
            ('Orthogonal SPADE', 'oSPADE',  np.asarray(r_oSPADE)),
        ]

        # ---------- Big parallel pool over all (S, method, K_keep) ----------
        big_jobs = []
        for S_test in SAM_NUM_LIST:
            for K_keep in K_sweep:
                for (method_name, which_case, r_use) in tasks_base:
                    payload = (method_name, which_case, r_use, I_train, I_test,
                               C1, C2, C2_extra, C3, C3_extra, float(alpha[which_alpha]), M, n, L,
                               persons_all, train_idx, test_idx, K_keep, S_test, 98765 + rep, NUM_PERSONS)
                    big_jobs.append((S_test, K_keep, method_name, payload))

        print(f"\n[Face-ID] Running {len(big_jobs)} classification jobs in parallel…")
        results_cache = {}  # (S,K,method) -> result
        with _process_pool() as ex:
            futures = {ex.submit(_face_method_worker, payload): (S_test, K_keep, method_name)
                       for (S_test, K_keep, method_name, payload) in big_jobs}
            for fut in tqdm(as_completed(futures), total=len(futures), unit="job"):
                S_test, K_keep, method_name = futures[fut]
                res = fut.result()
                results_cache[(S_test, K_keep, method_name)] = res
                # record
                acc_records[method_name][S_test][K_keep].append(res['top1_acc'])
                # keep "last snapshot"-style outputs (overwritten until final)
                summary_last_results = {
                    m: {
                        'top1_acc': results_cache.get((S_test, K_keep, m), {'top1_acc': float('nan')})['top1_acc'],
                        'K_keep': K_keep,
                        'which_alpha_index': int(which_alpha),
                        'alpha_value': float(alpha[which_alpha]),
                        'num_train': int(len(train_idx)),
                        'num_test': int(len(test_idx)),
                        'S_test': S_test
                    } for m in methods_list
                }
                out_bundle_last = {
                    'results': summary_last_results,
                    'per_person_acc': {m: results_cache.get((S_test, K_keep, m), {'per_person': {}})['per_person'] for m in methods_list},
                    'params': {
                        'L': L, 'Lmax': Lmax, 'sigma': sigma, 'N': N, 'M': M, 'nmax': nmax, 'n': n,
                        'N_measure': N_measure, 'L_measure': L_measure, 'S_eig': S_eig,
                        'alpha_grid': alpha, 'which_alpha_index': int(which_alpha),
                        'K_keep': K_keep, 'y0': y0.tolist(),
                        'NUM_PERSONS': NUM_PERSONS
                    },
                    'r_DI': r_DI, 'r_sSPADE': r_sSPADE, 'r_oSPADE': r_oSPADE,
                    'train_idx': train_idx, 'test_idx': test_idx,
                    'S_test': S_test
                }

    # ---- Save sweep data ----
    save_bundle = {
        'acc_records': acc_records,
        'K_sweep': K_sweep,
        'methods': methods_list,
        'S_list': SAM_NUM_LIST,
        'N_REPEATS': N_REPEATS,
        'alpha_index': int(which_alpha),
        'alpha_value': float(alpha[which_alpha]),
        'fixed_params': {'L':L, 'Lmax':Lmax, 'sigma':sigma, 'N':N, 'M':M, 'nmax':nmax, 'n':n,
                         'N_measure':N_measure, 'L_measure':L_measure, 'S_eig':S_eig, 'y0': y0.tolist(),
                         'NUM_PERSONS': NUM_PERSONS},
    }
    with open('success_vs_K_keep_multiS_first10.pkl', 'wb') as f:
        pickle.dump(save_bundle, f)
    print("Saved sweep data → success_vs_K_keep_multiS_first10.pkl")

    if out_bundle_last is not None:
        with open('face_id_results_first10.pkl', 'wb') as f:
            pickle.dump(out_bundle_last, f)
        print("Saved face ID last snapshot → face_id_results_first10.pkl")

        with open('face_id_summary_first10.txt', 'w') as f:
            f.write("Top-1 accuracies (overall success rate) — last snapshot (first 10 persons):\n")
            for k,v in out_bundle_last['results'].items():
                acc = v['top1_acc']
                f.write(f"{k:>20}: {acc if not np.isnan(acc) else float('nan'):.4f} (S={v['S_test']})\n")
        print("Saved summary text → face_id_summary_first10.txt")

    # ---- Plot
    Ks = np.array(K_sweep, dtype=int)

    def _compute_stats(vals_list):
        if len(vals_list) == 0:
            return np.nan, np.nan, np.nan
        arr = np.array([v for v in vals_list if not np.isnan(v)], dtype=float)
        if arr.size == 0:
            return np.nan, np.nan, np.nan
        return np.mean(arr), np.min(arr), np.max(arr)

    for method_name in methods_list:
        plt.figure(figsize=(6.5,4.8))
        ax = plt.subplot(1,1,1)
        for side in ('bottom','left','top','right'):
            ax.spines[side].set_linewidth(2)
        plt.xticks(Ks, fontsize=12)
        plt.yticks(fontsize=12)
        ax.set_xlabel('K_keep (number of kept components)', fontsize=14)
        ax.set_ylabel('P_succ', fontsize=14)
        ax.set_title(f"{method_name} (first 10 persons)", fontsize=16)

        for S_test in SAM_NUM_LIST:
            means, mins, maxs = [], [], []
            for K in K_sweep:
                m, mn, mx = _compute_stats(acc_records[method_name][S_test][K])
                means.append(m); mins.append(mn); maxs.append(mx)
            means = np.array(means); mins = np.array(mins); maxs = np.array(maxs)
            ax.plot(Ks, means, linewidth=2, label=f"S = {S_test:.0e}")
            ax.fill_between(Ks, mins, maxs, alpha=0.15)

        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=11, title='Sample size')
        out_png = f"success_vs_K_keep_{method_name.replace(' ','_')}_first10.png"
        plt.gcf().savefig(out_png, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved plot → {out_png}")

    print("All done in", round(time.time()-t0,2), "s")