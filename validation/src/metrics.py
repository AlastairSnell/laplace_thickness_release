# src/metrics.py
import numpy as np
from scipy.spatial.distance import directed_hausdorff

__all__ = [
    "euclidean_disp", "path_length", "asymmetry",
    "hausdorff", "rmse", "percentile"
]

def euclidean_disp(p0, p2):
    return np.linalg.norm(p0 - p2)

def path_length(path):
    return np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

def asymmetry(L_fwd, L_rev):
    return abs(L_fwd - L_rev) / L_fwd

def hausdorff(P, Q):
    return max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0])

def rmse(errors):
    return np.sqrt(np.mean(errors**2))

def percentile(errors, p):
    return np.percentile(errors, p)
