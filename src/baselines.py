"""Classical baselines C-2 (Constant Velocity), C-3 (Kalman + Mahalanobis), and C-4 (C-3 + Size + Color)."""
import numpy as np
from scipy.optimize import linear_sum_assignment

def predict_constant_velocity(hist_boxes: np.ndarray, hist_len: int, gap: int) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted (cx,cy,w,h) and velocity (vx,vy) from last up to 5 valid steps."""
    L = max(1, int(hist_len))
    valid = hist_boxes[-L:]
    last = valid[-1].astype(np.float64)
    if L == 1:
        return last.astype(np.float32), np.zeros((2,), dtype=np.float32)
    k = min(5, L)
    sub = valid[-k:].astype(np.float64)
    diffs = np.diff(sub[:, :2], axis=0)
    vel = diffs.mean(axis=0)
    pred = last.copy()
    pred[0] += vel[0] * float(gap)
    pred[1] += vel[1] * float(gap)
    return pred.astype(np.float32), vel.astype(np.float32)

_KALMAN_HIST_CACHE: dict = {}

def kalman_predict_1track(hist_boxes: np.ndarray, hist_len: int, gap: int, sigma_a: float = 0.5, obs_frac: float = 0.03):
    """8-state linear Kalman filter [cx,cy,w,h,vx,vy,vw,vh] run over history then extrapolated `gap` steps.
    Returns mean (4,) and diagonal variance (4,) on (cx,cy,w,h).
    """
    L = max(1, int(hist_len))
    valid_f32 = np.ascontiguousarray(hist_boxes[-L:], dtype=np.float32)
    cache_key = (valid_f32.tobytes(), L, float(sigma_a), float(obs_frac))
    cached = _KALMAN_HIST_CACHE.get(cache_key)

    if cached is None:
        valid = valid_f32.astype(np.float64)
        h_ref = max(8.0, float(valid[-1, 3]))
        w_ref = max(8.0, float(valid[-1, 2]))

        x = np.zeros((8,), dtype=np.float64)
        x[:4] = valid[0]
        if L >= 2:
            x[4:6] = valid[1, :2] - valid[0, :2]
        P = np.diag([
            (0.1 * w_ref) ** 2,
            (0.1 * h_ref) ** 2,
            (0.1 * w_ref) ** 2,
            (0.1 * h_ref) ** 2,
            (0.15 * w_ref) ** 2,
            (0.15 * h_ref) ** 2,
            (0.05 * w_ref) ** 2,
            (0.05 * h_ref) ** 2,
        ])

        F = np.eye(8, dtype=np.float64)
        for i in range(4):
            F[i, i + 4] = 1.0
        H = np.zeros((4, 8), dtype=np.float64)
        for i in range(4):
            H[i, i] = 1.0

        q_pos = (0.5 * sigma_a) ** 2
        q_vel = (sigma_a) ** 2
        q_pos_vec = np.array([q_pos, q_pos, 0.25 * q_pos, 0.25 * q_pos], dtype=np.float64)
        q_vel_vec = np.array([q_vel, q_vel, 0.1 * q_vel, 0.1 * q_vel], dtype=np.float64)
        Q = np.diag(np.concatenate([q_pos_vec, q_vel_vec]))
        r_diag = np.array([
            (obs_frac * w_ref) ** 2,
            (obs_frac * h_ref) ** 2,
            (1.5 * obs_frac * w_ref) ** 2,
            (1.5 * obs_frac * h_ref) ** 2,
        ], dtype=np.float64)
        R = np.diag(r_diag)

        for step in range(1, L):
            x = F @ x
            P = F @ P @ F.T + Q
            z = valid[step]
            y = z - (H @ x)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            x = x + K @ y
            P = (np.eye(8) - K @ H) @ P

        p0 = np.diag(P[:4, :4]).copy()
        c0 = np.array([P[i, i + 4] for i in range(4)], dtype=np.float64)
        v0 = np.diag(P[4:, 4:]).copy()
        cached = (x, p0, c0, v0, q_pos_vec, q_vel_vec, r_diag)
        if len(_KALMAN_HIST_CACHE) > 8192:
            _KALMAN_HIST_CACHE.clear()
        _KALMAN_HIST_CACHE[cache_key] = cached

    x, p0, c0, v0, q_pos_vec, q_vel_vec, r_diag = cached
    gi = int(gap)
    g = float(gi)
    x_out = x[:4] + g * x[4:]
    s_k2 = float(gi * (gi - 1) * (2 * gi - 1) // 6)
    p_out = p0 + 2.0 * g * c0 + (g * g) * v0 + g * q_pos_vec + s_k2 * q_vel_vec
    S_out = np.maximum(p_out + r_diag, 4.0)
    return x_out.astype(np.float32), S_out.astype(np.float32)

class ConstantVelocityBaseline:
    """C-2: Constant velocity extrapolation + normalized distance + size penalty + Hungarian."""
    name = "C-2"

    def __init__(self, theta: float = 2.2, size_w: float = 0.5):
        self.theta = float(theta)
        self.size_w = float(size_w)

    def cost(self, ev: dict) -> np.ndarray:
        N = len(ev["t_lost"])
        M = len(ev["cand"])
        C = np.zeros((N, M), dtype=np.float32)
        cand = ev["cand"]
        for i in range(N):
            pred, _ = predict_constant_velocity(ev["hist"][i], int(ev["hist_len"][i]), int(ev["gap"][i]))
            h_i = max(8.0, float(ev["hist"][i, -1, 3]))
            w_i = max(8.0, float(ev["hist"][i, -1, 2]))
            dist = np.hypot(cand[:, 0] - pred[0], cand[:, 1] - pred[1]) / h_i
            sz_diff = np.abs(np.log(np.maximum(2.0, cand[:, 2]) / w_i)) + np.abs(np.log(np.maximum(2.0, cand[:, 3]) / h_i))
            C[i] = dist + self.size_w * sz_diff
        return C

    def decide(self, ev: dict) -> np.ndarray:
        C = self.cost(ev)
        N, M = C.shape
        out = np.full((N,), -1, dtype=np.int32)
        if N == 0 or M == 0:
            return out
        r_ind, c_ind = linear_sum_assignment(C)
        for r, c in zip(r_ind, c_ind):
            if C[r, c] <= self.theta:
                out[r] = int(c)
        return out

class KalmanMahalanobisBaseline:
    """C-3 (lambda_s=0, lambda_c=0) and C-4 (Kalman + size + color)."""
    def __init__(self, theta: float = 9.21, lambda_s: float = 0.0, lambda_c: float = 0.0, name: str = "C-3"):
        self.theta = float(theta)
        self.lambda_s = float(lambda_s)
        self.lambda_c = float(lambda_c)
        self.name = name

    def predict_gaussian(self, ev: dict) -> tuple[np.ndarray, np.ndarray]:
        N = len(ev["t_lost"])
        mus = np.zeros((N, 4), dtype=np.float32)
        vars_ = np.zeros((N, 4), dtype=np.float32)
        for i in range(N):
            mus[i], vars_[i] = kalman_predict_1track(ev["hist"][i], int(ev["hist_len"][i]), int(ev["gap"][i]))
        return mus, vars_

    def cost(self, ev: dict) -> np.ndarray:
        N = len(ev["t_lost"])
        M = len(ev["cand"])
        C = np.zeros((N, M), dtype=np.float32)
        cand = ev["cand"]
        mus, vars_ = self.predict_gaussian(ev)
        for i in range(N):
            mu = mus[i]
            var = vars_[i]
            d2 = ((cand[:, 0] - mu[0]) ** 2) / var[0] + ((cand[:, 1] - mu[1]) ** 2) / var[1]
            c_val = d2
            if self.lambda_s > 0.0:
                w_i = max(8.0, float(ev["hist"][i, -1, 2]))
                h_i = max(8.0, float(ev["hist"][i, -1, 3]))
                sz = np.abs(np.log(np.maximum(2.0, cand[:, 2]) / w_i)) + np.abs(np.log(np.maximum(2.0, cand[:, 3]) / h_i))
                c_val = c_val + self.lambda_s * sz
            if self.lambda_c > 0.0 and "track_color" in ev and "cand_hist" in ev:
                tc = ev["track_color"][i]
                ch = ev["cand_hist"]
                tn = float(np.linalg.norm(tc))
                cn = np.linalg.norm(ch, axis=1)
                cos = np.where((tn > 1e-6) & (cn > 1e-6), (ch @ tc) / np.maximum(tn * cn, 1e-6), 0.0)
                c_val = c_val + self.lambda_c * (1.0 - np.clip(cos, 0.0, 1.0))
            C[i] = c_val
        return C

    def decide(self, ev: dict) -> np.ndarray:
        C = self.cost(ev)
        N, M = C.shape
        out = np.full((N,), -1, dtype=np.int32)
        if N == 0 or M == 0:
            return out
        r_ind, c_ind = linear_sum_assignment(C)
        for r, c in zip(r_ind, c_ind):
            if C[r, c] <= self.theta:
                out[r] = int(c)
        return out
