import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "Arial",
    "font.weight": "bold",
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import Normalize

# -----------------
# Inputs
# -----------------
OBS_CSV  = "./data/tSNE_measured_samples.csv"
TEST_CSV = "./data/tSNE_virtual_samples.csv"

OUT_DIR = "./figures/"
os.makedirs(OUT_DIR, exist_ok=True)

# -----------------
# Settings
# -----------------
USE_SIGNED_LOG = True
USE_PCA = True
PCA_DIM = 15

PERPLEXITY = 20.0
N_ITER = 1200
RANDOM_STATE = 0

VMIN = 0.0
VMAX = 40.0

GRID_BINS = 240
SMOOTH_SIGMA_BINS = 2.4
CLOUD_ALPHA_MAX = 0.92
DENS_PCTL_REF = 90

OUTLIER_GRAY = (0.55, 0.55, 0.55)
OUTLIER_ALPHA_MAX = 0.85
OUTLIER_DENS_PCTL_REF = 90

S_OBS = 45
ALPHA_OBS = 0.95
OBS_EDGE_COLOR = "k"
OBS_EDGE_LW = 0.6

ANCHOR_MARKERS = {
    "Ni3.5": "o", # circle (loop0)
    "Li0.35Al0.545Ni0.704Nb0.102Sm0.058": "D", # diamond (loop0)
    "Li0.6Ni0.8Ta0.2": "s", # square (loop12)
    "Li0.4Mn0.2Ni0.8Nb0.2": "^", # triangle (loop 12)
    "Al0.5Ni2.5Nb0.5": "P", # plus (loop 8)
}
S_ANCHOR = 80
ANCHOR_EDGE_COLOR = "white"
ANCHOR_EDGE_LW = 1.6

MODEL_LOOP = 14       # Loop used for predicted H2 yield
DESCRIPTOR_LOOP = 14  # Loop used for t-SNE embedding

# -----------------
# Helpers
# -----------------
def parse_vec(s):
    if isinstance(s, list):
        return np.asarray(s, float)
    return np.asarray(ast.literal_eval(s), float)

def load_X(df, loop):
    col = f"X_{loop}"
    if col not in df.columns:
        raise KeyError(f"Missing column: {col}")
    return np.vstack(df[col].map(parse_vec).values)

def signed_log1p(X):
    return np.sign(X) * np.log1p(np.abs(X))

def build_fixed_tsne_map(df_test, df_obs_all, loop_last):
    X_test_last = load_X(df_test, loop_last)
    X_obs_last  = load_X(df_obs_all, loop_last)
    X_all = np.vstack([X_test_last, X_obs_last])

    if USE_SIGNED_LOG:
        X_all = signed_log1p(X_all)

    X_all = StandardScaler().fit_transform(X_all)

    if USE_PCA:
        dim = X_all.shape[1]
        ncomp = min(PCA_DIM if PCA_DIM is not None else dim, dim)
        if ncomp < dim:
            X_all = PCA(n_components=ncomp, random_state=RANDOM_STATE).fit_transform(X_all)

    n_all = X_all.shape[0]
    perp = min(PERPLEXITY, max(5.0, (n_all - 1) / 3.0))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        learning_rate="auto",
        n_iter=N_ITER,
        init="pca",
        random_state=RANDOM_STATE,
        method="barnes_hut",
        verbose=1,
    )

    Z_all = tsne.fit_transform(X_all)
    Z_test = Z_all[:len(df_test)]
    Z_obs  = Z_all[len(df_test):]
    return Z_test, Z_obs

def _gaussian_kernel1d(sigma_bins, radius=None):
    sigma = float(sigma_bins)
    if sigma <= 0:
        return np.array([1.0], float)
    if radius is None:
        radius = int(max(3, np.ceil(3 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x * x) / (2 * sigma * sigma))
    k /= k.sum()
    return k

def _convolve1d_reflect(a, k, axis):
    a = np.asarray(a, float)
    k = np.asarray(k, float)
    pad = len(k) // 2

    if axis == 0:
        ap = np.pad(a, ((pad, pad), (0, 0)), mode="reflect")
        out = np.empty_like(a, float)
        for i in range(a.shape[0]):
            out[i, :] = np.tensordot(ap[i:i+len(k), :], k, axes=(0, 0))
        return out
    else:
        ap = np.pad(a, ((0, 0), (pad, pad)), mode="reflect")
        out = np.empty_like(a, float)
        for j in range(a.shape[1]):
            out[:, j] = np.tensordot(ap[:, j:j+len(k)], k, axes=(1, 0))
        return out

def _smooth2d(a, sigma_bins):
    k = _gaussian_kernel1d(sigma_bins)
    tmp = _convolve1d_reflect(a, k, axis=0)
    out = _convolve1d_reflect(tmp, k, axis=1)
    return out

def _hist2d_weighted(x, y, w, bins, xlim, ylim):
    sum_w, xedges, yedges = np.histogram2d(
        x, y, bins=bins, range=[xlim, ylim], weights=w
    )
    cnt, _, _ = np.histogram2d(
        x, y, bins=bins, range=[xlim, ylim], weights=None
    )
    return sum_w, cnt, xedges, yedges

def make_pred_field(Z_test, y_pred_raw, xlim, ylim):
    y_pred_raw = np.asarray(y_pred_raw, float)
    m = np.isfinite(y_pred_raw)
    x = Z_test[m, 0]
    y = Z_test[m, 1]

    w = np.clip(y_pred_raw[m], VMIN, VMAX)

    sum_w, cnt, xedges, yedges = _hist2d_weighted(x, y, w, GRID_BINS, xlim, ylim)

    sum_w_s = _smooth2d(sum_w, SMOOTH_SIGMA_BINS)
    cnt_s   = _smooth2d(cnt,   SMOOTH_SIGMA_BINS)

    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sum_w_s / np.maximum(cnt_s, 1e-12)
    mean[cnt_s < 1e-6] = np.nan

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    return mean, cnt_s, extent

def make_outlier_density_field(Z_test, y_pred_raw, xlim, ylim):
    y_pred_raw = np.asarray(y_pred_raw, float)
    m = np.isfinite(y_pred_raw) & (y_pred_raw > VMAX)
    if not np.any(m):
        dummy = np.zeros((GRID_BINS, GRID_BINS), float)
        extent = [xlim[0], xlim[1], ylim[0], ylim[1]]
        return dummy, extent

    x = Z_test[m, 0]
    y = Z_test[m, 1]
    cnt, xedges, yedges = np.histogram2d(x, y, bins=GRID_BINS, range=[xlim, ylim], weights=None)
    cnt_s = _smooth2d(cnt, SMOOTH_SIGMA_BINS)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return cnt_s, extent

def density_to_alpha(cnt_s, alpha_max, pctl_ref):
    d = np.asarray(cnt_s, float)
    nz = d[d > 0]
    ref = np.percentile(nz, pctl_ref) if nz.size else 1.0
    a = np.clip(d / (ref + 1e-12), 0.0, 1.0)
    a = (a ** 0.7) * alpha_max
    return a

# -----------------
# Main
# -----------------
def main():
    df_test = pd.read_csv(TEST_CSV)
    df_obs_all = pd.read_csv(OBS_CSV)

    avail_obs = df_obs_all["Available_from"].to_numpy(int)
    anchor_names = set(ANCHOR_MARKERS.keys())
    is_anchor_all = df_obs_all["Catalyst_name"].astype(str).isin(anchor_names).to_numpy(bool)
    c_obs_true_all = np.clip(df_obs_all["True_yield"].to_numpy(float), VMIN, VMAX)

    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=VMIN, vmax=VMAX, clip=True)

    # Generate one map using the selected descriptor and model loops
    Z_test, Z_obs = build_fixed_tsne_map(df_test, df_obs_all, DESCRIPTOR_LOOP)

    xmin = min(np.min(Z_test[:, 0]), np.min(Z_obs[:, 0]))
    xmax = max(np.max(Z_test[:, 0]), np.max(Z_obs[:, 0]))
    ymin = min(np.min(Z_test[:, 1]), np.min(Z_obs[:, 1]))
    ymax = max(np.max(Z_test[:, 1]), np.max(Z_obs[:, 1]))
    dx = (xmax - xmin) * 0.03
    dy = (ymax - ymin) * 0.03
    xlim = (xmin - dx, xmax + dx)
    ylim = (ymin - dy, ymax + dy)

    y_pred_raw = df_test[f"Predicted_yield_{MODEL_LOOP}"].to_numpy(float)

    mean_field, dens_field, extent = make_pred_field(Z_test, y_pred_raw, xlim, ylim)
    out_cnt_s, out_extent = make_outlier_density_field(Z_test, y_pred_raw, xlim, ylim)

    fig, ax = plt.subplots(figsize=(9, 7))

    alpha = density_to_alpha(dens_field, CLOUD_ALPHA_MAX, DENS_PCTL_REF)
    rgba = cmap(norm(np.nan_to_num(mean_field, nan=0.0)))
    rgba[..., 3] = np.where(np.isfinite(mean_field), alpha, 0.0)

    ax.imshow(rgba.transpose(1, 0, 2), origin="lower", extent=extent, interpolation="bilinear", zorder=1)

    out_alpha = density_to_alpha(out_cnt_s, OUTLIER_ALPHA_MAX, OUTLIER_DENS_PCTL_REF)
    out_rgba = np.zeros((out_cnt_s.shape[0], out_cnt_s.shape[1], 4), float)
    out_rgba[..., :3] = OUTLIER_GRAY
    out_rgba[..., 3] = out_alpha

    ax.imshow(out_rgba.transpose(1, 0, 2), origin="lower", extent=out_extent, interpolation="bilinear", zorder=2)

    mask_obs = avail_obs <= MODEL_LOOP
    mask_anchor = mask_obs & is_anchor_all
    mask_normal = mask_obs & (~is_anchor_all)

    ax.scatter(Z_obs[mask_normal, 0], Z_obs[mask_normal, 1],
               c=c_obs_true_all[mask_normal], s=S_OBS,
               alpha=ALPHA_OBS, edgecolors=OBS_EDGE_COLOR,
               linewidths=OBS_EDGE_LW, cmap=cmap, norm=norm, zorder=10)

    for name, marker in ANCHOR_MARKERS.items():
        idx = mask_anchor & (df_obs_all["Catalyst_name"].astype(str).to_numpy() == name)
        if np.any(idx):
            ax.scatter(Z_obs[idx, 0], Z_obs[idx, 1],
                       c=c_obs_true_all[idx], s=S_ANCHOR,
                       marker=marker, edgecolors=ANCHOR_EDGE_COLOR,
                       linewidths=ANCHOR_EDGE_LW, cmap=cmap, norm=norm, zorder=12)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")

    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("H\u2082 yield (%)", rotation=270, labelpad=22, fontsize=18, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"tSNE_model{MODEL_LOOP}_descriptor{DESCRIPTOR_LOOP}.png"), dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
