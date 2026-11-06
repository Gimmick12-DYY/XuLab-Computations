"""Minimal mcool reader (h5py): balanced and raw dense submatrix for a region."""
import h5py, numpy as np
class Hic:
    def __init__(self, path, res):
        self.f = h5py.File(path, "r"); self.g = self.f[f"resolutions/{res}"]; self.res = res; g = self.g
        self.names = [x.decode() for x in g["chroms/name"][:]]; self.co = g["indexes/chrom_offset"][:]; self.b1o = g["indexes/bin1_offset"]; self.w = {}
    def _c(self, c):
        i = self.names.index(c); return int(self.co[i]), int(self.co[i + 1])
    def fetch(self, c, s, e):
        c0, c1 = self._c(c); i0 = c0 + s // self.res; i1 = min(c1, c0 + (e - 1) // self.res + 1); n = i1 - i0; g = self.g
        lo, hi = int(self.b1o[i0]), int(self.b1o[i1]); a = g["pixels/bin1_id"][lo:hi]; b = g["pixels/bin2_id"][lo:hi]; v = g["pixels/count"][lo:hi]; m = b < i1; a, b, v = a[m] - i0, b[m] - i0, v[m]
        R = np.zeros((n, n)); R[a, b] = v; R[b, a] = v; w = g["bins/weight"][i0:i1]; return R * np.outer(w, w), R, w
