#!/usr/bin/env python3
# Part 2.2 — Sampling (rays from images, points along rays)
import os, argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -----------------------------
# Minimal camera math (NumPy)
# -----------------------------
def make_K(f, W, H):
    return np.array([[f, 0, W*0.5],
                     [0, f, H*0.5],
                     [0, 0, 1]], dtype=np.float64)

def transform_np(c2w: np.ndarray, x_c: np.ndarray) -> np.ndarray:
    """Camera->world for points (supports [...,3] or [...,4])."""
    x = x_c
    if x.shape[-1] == 3:
        ones = np.ones((*x.shape[:-1], 1), dtype=x.dtype)
        x = np.concatenate([x, ones], -1)
    if c2w.ndim == 2:
        Xw = x @ c2w.T
    else:
        Xw = np.einsum("bij,...j->...i", c2w, x)
    w = np.clip(Xw[..., 3:4], 1.0, None)
    return Xw[..., :3] / w

def pixel_to_camera_np(K: np.ndarray, uv: np.ndarray, s=1.0) -> np.ndarray:
    """Invert pinhole: (u,v, depth=s) -> (x,y,z) in camera coords."""
    fx, fy = K[0,0], K[1,1]; cx, cy = K[0,2], K[1,2]
    uv = np.asarray(uv, dtype=np.float64)
    z = np.broadcast_to(np.asarray(s, dtype=np.float64), uv[...,0].shape)
    x = (uv[...,0] - cx) * (z / fx)
    y = (uv[...,1] - cy) * (z / fy)
    return np.stack([x, y, z], -1)

def normalize_np(v, eps=1e-9):
    return v / np.clip(np.linalg.norm(v, axis=-1, keepdims=True), eps, None)

def pixel_to_ray_np(K, c2w, uv):
    """Ray origin & unit direction for pixels uv (…×2)."""
    if c2w.ndim == 2:
        ro = np.broadcast_to(c2w[:3,3], (uv.shape[-2], 3)).copy()
        Xw = transform_np(c2w, pixel_to_camera_np(K, uv, 1.0))
        rd = normalize_np(Xw - ro)
    else:
        B, N = uv.shape[0], uv.shape[-2]
        ro = np.broadcast_to(c2w[:, :3, 3][:, None, :], (B, N, 3)).copy()
        xc = pixel_to_camera_np(K, uv, 1.0)
        xh = np.concatenate([xc, np.ones((B, N, 1), xc.dtype)], -1)
        Xw = np.einsum("bij,bnj->bni", c2w, xh)[..., :3]
        rd = normalize_np(Xw - ro)
    return ro, rd

# -----------------------------
# Dataset helpers
# -----------------------------
def load_dataset(npz_path, which_set="train", idx=0):
    data = np.load(npz_path, allow_pickle=True)
    if which_set == "train":
        imgs = data["images_train"].astype(np.float32) / 255.0
        c2ws = data["c2ws_train"]
    elif which_set == "val":
        imgs = data["images_val"].astype(np.float32) / 255.0
        c2ws = data["c2ws_val"]
    else:
        raise ValueError("which_set must be 'train' or 'val' for sampling rays with RGB.")
    H, W = imgs.shape[1:3]
    focal = float(data["focal"]) if "focal" in data else float(data["K_rect"][0,0])
    K = make_K(focal, W, H)
    return imgs, c2ws, K, H, W

def uv_grid_hw(H, W, add_half=True):
    u, v = np.meshgrid(np.arange(W), np.arange(H), indexing="xy")
    uv = np.stack([u, v], -1).astype(np.float64)
    if add_half: uv += 0.5
    return uv  # [H,W,2]

# -----------------------------
# Ray sampling strategies
# -----------------------------
def sample_rays_global(imgs, c2ws, K, N):
    """
    Global flat sampling over all pixels of all images.
    Returns ray_o [N,3], ray_d [N,3], rgb [N,3].
    """
    S, H, W = imgs.shape[:3]
    total = S * H * W
    idx = np.random.randint(0, total, size=(N,), dtype=np.int64)
    s = idx // (H*W)
    rem = idx % (H*W)
    v = rem // W
    u = rem % W
    uv = np.stack([u, v], -1).astype(np.float64) + 0.5  # pixel centers
    # group by image for fewer transforms
    ray_o_list = []; ray_d_list = []; rgb_list = []
    for si in np.unique(s):
        mask = (s == si)
        uv_si = uv[mask]
        ro, rd = pixel_to_ray_np(K, c2ws[si], uv_si)
        ray_o_list.append(ro); ray_d_list.append(rd)
        rgb_list.append(imgs[si, v[mask], u[mask], :])
    ray_o = np.concatenate(ray_o_list, 0)
    ray_d = np.concatenate(ray_d_list, 0)
    rgb  = np.concatenate(rgb_list, 0)
    return ray_o, ray_d, rgb

def sample_rays_per_image(imgs, c2ws, K, N, M):
    """
    Pick M images uniformly; sample N//M rays from each.
    Returns ray_o [N,3], ray_d [N,3], rgb [N,3].
    """
    S, H, W = imgs.shape[:3]
    img_ids = np.random.choice(S, size=(M,), replace=(M>S))
    per = max(1, N // M)
    ros=[]; rds=[]; rgbs=[]
    for si in img_ids:
        u = np.random.randint(0, W, size=(per,), dtype=np.int64)
        v = np.random.randint(0, H, size=(per,), dtype=np.int64)
        uv = np.stack([u, v], -1).astype(np.float64) + 0.5
        ro, rd = pixel_to_ray_np(K, c2ws[si], uv)
        ros.append(ro); rds.append(rd); rgbs.append(imgs[si, v, u, :])
    return np.concatenate(ros,0)[:N], np.concatenate(rds,0)[:N], np.concatenate(rgbs,0)[:N]

# -----------------------------
# Points along rays (stratified)
# -----------------------------
def sample_along_rays(ray_o, ray_d, n_samples=64, near=2.0, far=6.0, perturb=True):
    """
    Stratified sampling along rays: returns
      t_vals [N,n], pts [N,n,3]
    """
    N = ray_o.shape[0]
    t_lin = np.linspace(near, far, n_samples+1, dtype=np.float64)  # bin edges
    t0, t1 = t_lin[:-1], t_lin[1:]
    t = t0  # left edge as base
    if perturb:
        width = (t1 - t0)  # [n]
        t = t + np.random.rand(N, n_samples) * width  # broadcast width over N
    else:
        t = 0.5*(t0 + t1)[None, :] * np.ones((N, n_samples), dtype=np.float64)
    pts = ray_o[:, None, :] + t[..., None] * ray_d[:, None, :]
    return t, pts

# -----------------------------
# Self-checks & visualization
# -----------------------------
def checks_and_viz(outdir, K, c2w, ray_o, ray_d, t_vals, pts, uv_used):
    os.makedirs(outdir, exist_ok=True)
    # numeric: (1) |rd|=1, (2) t in range, (3) reprojection error ~ 0
    dir_norm_err = np.max(np.abs(np.linalg.norm(ray_d, axis=-1) - 1.0))
    tmin, tmax = float(np.min(t_vals)), float(np.max(t_vals))

    # pick a subset, march one t, project back to uv
    w2c = np.linalg.inv(c2w)
    k = min(512, ray_o.shape[0])
    X = pts[:k, pts.shape[1]//2, :]          # middle depth sample
    Xc = transform_np(w2c, X)                # world->camera
    uv_h = (K @ Xc.T).T
    uv_proj = uv_h[:, :2] / uv_h[:, 2:3]
    reproj_rmse = float(np.sqrt(np.mean((uv_proj - uv_used[:k])**2)))

    # viz: scatter a tiny bundle of points color-coded by depth
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection="3d")
    sel = min(64, ray_o.shape[0])
    P = pts[:sel].reshape(-1,3)
    C = np.tile(t_vals[:sel], (1,1)).reshape(-1)  # color by t
    s = ax.scatter(P[:,0], P[:,1], P[:,2], c=C, s=3, cmap="viridis")
    ax.set_title("Points along sampled rays (colored by t)")
    fig.colorbar(s, ax=ax, shrink=0.7, label="t (depth)")
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, "part2_2_points3d.png"), dpi=200)

    with open(os.path.join(outdir, "part2_2_checks.txt"), "w") as f:
        f.write(f"|rd|-1 max err: {dir_norm_err:.2e}\n")
        f.write(f"t range: [{tmin:.3f}, {tmax:.3f}]\n")
        f.write(f"reproj RMSE (px) @ mid-sample: {reproj_rmse:.2e}\n")

    print(f"[✓] wrote {outdir}/part2_2_points3d.png")
    print(f"[✓] wrote {outdir}/part2_2_checks.txt")
    print(f"|rd|-1 max err={dir_norm_err:.2e} | t∈[{tmin:.2f},{tmax:.2f}] | reproj RMSE={reproj_rmse:.2e}")

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="data/lego_200x200.npz")
    ap.add_argument("--set", choices=["train","val"], default="train")
    ap.add_argument("--N", type=int, default=8192, help="rays per iteration")
    ap.add_argument("--mode", choices=["global","per_image"], default="global")
    ap.add_argument("--M", type=int, default=4, help="num images if mode=per_image")
    ap.add_argument("--n_samples", type=int, default=64)
    ap.add_argument("--near", type=float, default=2.0)
    ap.add_argument("--far", type=float, default=6.0)
    ap.add_argument("--no_perturb", action="store_true")
    ap.add_argument("--outdir", default="outputs_part2_2")
    args = ap.parse_args()

    np.random.seed()  # non-deterministic jitter each run

    imgs, c2ws, K, H, W = load_dataset(args.npz, which_set=args.set)

    if args.mode == "global":
        ray_o, ray_d, rgb = sample_rays_global(imgs, c2ws, K, args.N)
        # pick a single c2w just for the projection sanity in viz (use the first image for the subset)
        c2w_vis = c2ws[0]
        # reconstruct the uv we used for the subset reprojection
        # (approximate: assume these rays came from image 0 for the k subset)
        uv_used = np.full((min(512, args.N),2), np.nan, np.float64)
    else:
        ray_o, ray_d, rgb = sample_rays_per_image(imgs, c2ws, K, args.N, args.M)
        c2w_vis = c2ws[0]
        uv_used = np.full((min(512, args.N),2), np.nan, np.float64)

    # For a stricter reprojection check, resample a small consistent bundle from a single image:
    g = 16
    uv_grid = uv_grid_hw(H, W, add_half=True)[:: H//g or 1, :: W//g or 1].reshape(-1,2)
    ro_chk, rd_chk = pixel_to_ray_np(K, c2w_vis, uv_grid)
    t_chk, pts_chk = sample_along_rays(ro_chk, rd_chk, n_samples=args.n_samples,
                                       near=args.near, far=args.far,
                                       perturb=not args.no_perturb)

    os.makedirs(args.outdir, exist_ok=True)
    checks_and_viz(args.outdir, K, c2w_vis, ro_chk, rd_chk, t_chk, pts_chk, uv_grid)

if __name__ == "__main__":
    main()
