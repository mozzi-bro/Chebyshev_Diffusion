import numpy as np


def normalize_point_cloud(pc):
    pc_centered = pc - np.mean(pc, axis=0)
    max_dist = np.max(np.linalg.norm(pc_centered, axis=1))
    if max_dist > 0:
        pc_normalized = pc_centered / max_dist
    else:
        pc_normalized = pc_centered
    return pc_normalized


def compute_mmd(x, y, sigma=1.0):
    def rbf_kernel(a, b, sigma):
        dist = np.sum((a[:, np.newaxis] - b[np.newaxis, :]) ** 2, axis=2)
        return np.exp(-dist / (2 * sigma ** 2))

    xx = rbf_kernel(x, x, sigma)
    yy = rbf_kernel(y, y, sigma)
    xy = rbf_kernel(x, y, sigma)

    mmd = np.mean(xx) + np.mean(yy) - 2 * np.mean(xy)
    return max(0, mmd)


def compute_jsd(p, q, num_bins=50):
    from scipy.stats import entropy

    min_val = min(np.min(p), np.min(q))
    max_val = max(np.max(p), np.max(q))
    bins = np.linspace(min_val - 1e-6, max_val + 1e-6, num_bins + 1)

    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)

    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10

    p_hist = p_hist / np.sum(p_hist)
    q_hist = q_hist / np.sum(q_hist)

    m = 0.5 * (p_hist + q_hist)
    jsd = 0.5 * entropy(p_hist, m) + 0.5 * entropy(q_hist, m)

    return jsd


def sample_point_cloud(pc, num_points=2048):
    N = pc.shape[0]
    if N >= num_points:
        idx = np.random.choice(N, num_points, replace=False)
    else:
        idx = np.random.choice(N, num_points, replace=True)
    return pc[idx]
