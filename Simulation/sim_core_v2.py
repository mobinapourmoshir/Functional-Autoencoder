"""Core code for the FPCA vs FAE nonlinear amplitude-phase simulation.

Simulator  : the amplitude-phase generator; phase/amplitude strengths exposed in SimCfg.
FPCA       : curves projected on B cubic B-splines, then trapezoid-weighted FPCA, K scores, VAR(1).
FAE        : B -> 128 -> K -> 128 -> B (tanh), B-spline projection / revert layers, curves centred
             and scaled by a common scalar from the chronological training sub-block only;
             reconstruction MSE + 2nd-difference smoothness; epoch chosen on a chronological
             85/15 validation split, then refit on all T curves; VAR(1) on the latent sequence.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, asdict
import numpy as np
import torch

torch.set_num_threads(1)

# ------------------------------------------------------------------ basis
def bspline_basis(n_basis: int, n_time: int, degree: int = 3) -> np.ndarray:
    """Cubic B-spline basis (n_time x n_basis) on a uniform grid of [0,1], uniform interior knots."""
    tau = np.linspace(0.0, 1.0, n_time)
    p = degree
    n_int = max(n_basis - p - 1, 0)
    interior = np.linspace(0.0, 1.0, n_int + 2)[1:-1] if n_int > 0 else np.array([])
    knots = np.concatenate((np.zeros(p + 1), interior, np.ones(p + 1)))
    N = np.zeros((n_basis, n_time))
    for i in range(n_basis):
        N[i] = np.where((tau >= knots[i]) & (tau < knots[i + 1]), 1.0, 0.0)
    N[-1, tau == 1.0] = 1.0
    for k in range(1, p + 1):
        Nn = np.zeros_like(N)
        for i in range(n_basis):
            dl = knots[i + k] - knots[i]
            left = (tau - knots[i]) / dl * N[i] if dl > 0 else 0.0
            right = 0.0
            if i + 1 < n_basis:
                dr = knots[i + k + 1] - knots[i + 1]
                if dr > 0:
                    right = (knots[i + k + 1] - tau) / dr * N[i + 1]
            Nn[i] = left + right
        N = Nn
    return N.T


def trapezoid_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float); dx = np.diff(x); w = np.empty_like(x)
    w[0], w[-1] = dx[0] / 2, dx[-1] / 2; w[1:-1] = (x[2:] - x[:-2]) / 2
    return w

# ------------------------------------------------------------------ simulator
@dataclass
class SimCfg:
    seed: int = 123
    T: int = 500
    P: int = 100
    d: int = 5
    Sigma_scale: float = 0.028
    meas_noise_sd: float = 0.001
    age_min: float = 15.0
    age_max: float = 49.0
    fertility_tail_strength: float = 0.012
    # --- amplitude / phase strengths
    shift_scale: float = 0.14       # global horizontal shift amplitude
    dilation_scale: float = 0.18    # stretching / compression amplitude
    warp_scale: float = 0.08        # localised warp amplitude
    peak_width: float = 0.08        # main-peak template width
    shoulder_width: float = 0.12    # shoulder template width
    trend_scale: float = 1.0        # trend on z2 (scale)
    trend_mode: str = "det"         # "det": deterministic linear trend; "ar1": stationary AR(1) trend; "det+ar1": both
    trend_rho: float = 0.995        # AR(1) coefficient of the stochastic trend (trend_mode='ar1')
    trend_sd: float = 0.577         # stationary SD of the AR(1) trend (0.577 = SD of linspace(-1,1))
    clip_value: float = 2.0
    latent_rho: float = 0.0         # 0 = A; >0 replaces its diagonal by latent_rho
    nonlin_scale: float = 1.0       # multiplies the interaction term g(h_{t-1})
    drift_coef: float = 0.05        # coefficient of the deterministic drift d_t (0 = no drift)
    latent_gain: float = 1.0        # multiplies the stochastic latent state before the decoder
    map_mode: str = "nonlinear"     # "nonlinear" or "linear"
    lin_n_basis: int = 12           # linear benchmark: generating basis size 
    lin_target_rho: float = 0.85
    lin_weight_sd: float = 0.8
    lin_exact_rho: bool = False     # True: rescale A 
    lin_d: int = 0                  # number of linear latent factors (0 = use d)
    burnin: int = 0                 # linear benchmark: discarded burn-in steps (stationary start)


def _latent_A(cfg: SimCfg) -> np.ndarray:
    A = np.array([[0.60, 0.05, 0.00, 0.00, 0.00],
                  [0.00, 0.72, 0.06, 0.00, 0.00],
                  [0.00, 0.00, 0.66, 0.05, 0.00],
                  [0.00, 0.00, 0.00, 0.68, 0.06],
                  [0.03, 0.00, 0.00, 0.00, 0.62]])[:cfg.d, :cfg.d]
    if cfg.latent_rho > 0:
        A = A - np.diag(np.diag(A)) + cfg.latent_rho * np.eye(cfg.d)
    return A


def _latent_step(z, t, A, eps, cfg: SimCfg):
    """One step of h_t = A h_{t-1} + c*g(h_{t-1}) + 0.05 d_t + eps_t (vectorised over leading dims)."""
    T = cfg.T; tau = t / max(T - 1, 1)
    drift = np.array([-0.25 * (tau - .5), 1.10 * (tau - .5), .25 * np.sin(2 * np.pi * tau),
                      .20 * np.sin(np.pi * tau), .15 * np.cos(2 * np.pi * tau)])
    z0, z1, z2, z3, z4 = (z[..., i] for i in range(5))
    g = np.stack([.10 * np.tanh(z1) - .04 * np.tanh(.8 * z0 * z1),
                  .18 * np.tanh(z2) + .04 * np.tanh(z0 ** 2),
                  -.10 * np.tanh(z0) + .08 * np.tanh(.8 * z3),
                  .08 * np.sin(np.clip(z1, -3, 3)) + .04 * np.tanh(.6 * z2 * z3),
                  .06 * np.tanh(.8 * z4) + .03 * np.tanh(.6 * z1 * z2)], -1)
    return np.clip(z @ A.T + cfg.nonlin_scale * g + cfg.drift_coef * drift + eps, -cfg.clip_value, cfg.clip_value)


def _latent_nonlinear(cfg: SimCfg) -> np.ndarray:
    rng = np.random.default_rng(cfg.seed)
    d, T = cfg.d, cfg.T
    Sigma = cfg.Sigma_scale ** 2 * np.eye(d)
    A = _latent_A(cfg)
    Z = np.zeros((T, d)); Z[0] = rng.multivariate_normal(np.zeros(d), Sigma)
    for t in range(1, T):
        eps = rng.multivariate_normal(np.zeros(d), Sigma)
        Z[t] = _latent_step(Z[t - 1], t, A, eps, cfg)
    return Z


def oracle_forecast(cfg: SimCfg, L: int = 5, n_mc: int = 400, seed: int = 0):
    """Monte-Carlo conditional mean E[X_{T-L+h} | h_{T-L}] under the TRUE generator (nonlinear only).
    A lower bound on the achievable forecast error; used only as a design diagnostic."""
    Z = _latent_nonlinear(cfg); T0 = cfg.T - L
    rng = np.random.default_rng(seed); A = _latent_A(cfg)
    z = np.repeat(Z[T0 - 1][None], n_mc, 0); out = []
    for h in range(L):
        t = T0 + h
        z = _latent_step(z, t, A, rng.normal(size=z.shape) * cfg.Sigma_scale, cfg)
        tr_t = np.full(n_mc, np.linspace(-1, 1, cfg.T)[t])
        out.append(_decoder_nonlinear(z, cfg, tr=tr_t).mean(0))
    return np.array(out)


def trend_path(cfg: SimCfg) -> np.ndarray:
    """Trend driving the main-peak timing. 'det': linspace(-1, 1, T) (non-stationary);
    'ar1': stationary AR(1), tr_t = rho tr_{t-1} + e_t, started from its stationary law N(0, trend_sd^2)."""
    if cfg.trend_mode == "det":
        return np.linspace(-1, 1, cfg.T)
    if cfg.trend_mode == "det+ar1":     # deterministic trend + stationary AR(1) fluctuation around it
        return np.linspace(-1, 1, cfg.T) + trend_path(SimCfg(**{**cfg.__dict__, "trend_mode": "ar1"}))
    r = np.random.default_rng(cfg.seed + 555)
    tr = np.empty(cfg.T); tr[0] = cfg.trend_sd * r.normal()
    e = cfg.trend_sd * np.sqrt(1 - cfg.trend_rho ** 2) * r.normal(size=cfg.T)
    for t in range(1, cfg.T): tr[t] = cfg.trend_rho * tr[t - 1] + e[t]
    return tr


def _sig(x): return 1.0 / (1.0 + np.exp(-x))


def _decoder_nonlinear(Z: np.ndarray, cfg: SimCfg, tr=None) -> np.ndarray:
    T = Z.shape[0]
    u = np.linspace(0, 1, cfg.P)[None, :]
    if tr is None:
        tr = np.linspace(-1, 1, T)
    tr = cfg.trend_scale * tr
    Z = cfg.latent_gain * Z
    z1, z2, z3, z4, z5 = (Z[:, i] for i in range(5))
    z1 = z1 - 0.20 * tr; z2 = z2 + 1.00 * tr
    amp = 0.07 + 0.07 * _sig(z1 - 0.6 * z2)
    sh_amp = 0.16 * _sig(1.1 * z4 + 0.5 * z5)
    tail_amp = cfg.fertility_tail_strength * _sig(1.1 * z5 + 0.5 * z4)
    shift = cfg.shift_scale * np.tanh(1.4 * z2 - 0.7 * z3 + 0.3 * z1 * z4)[:, None]
    dil = 1.0 + cfg.dilation_scale * np.tanh(1.1 * z3 + 0.3 * z5)[:, None]
    lw = (cfg.warp_scale * np.tanh(1.3 * z4 - 0.6 * z2 + 0.2 * z1 * z5)[:, None]
          * np.exp(-0.5 * ((u - 0.48) / 0.16) ** 2) * np.sin(np.pi * (u - 0.15)))
    us = np.clip(0.5 + (u - shift + lw - 0.5) / dil, -0.2, 1.2)
    main = np.exp(-0.5 * ((us - 0.38) / cfg.peak_width) ** 2)
    shoulder = np.exp(-0.5 * ((us - 0.58) / cfg.shoulder_width) ** 2)
    tail = _sig((us - 0.62) / 0.04) * np.exp(-np.clip(us - 0.62, 0, None) / 0.13)
    X = amp[:, None] * main + (sh_amp * amp)[:, None] * shoulder + tail_amp[:, None] * tail
    return np.clip(X, 0.0, None)


def _linear_benchmark(cfg: SimCfg) -> np.ndarray:
    """Linear benchmark: latent VAR(1) (A ~ N(0,0.15^2), shrunk if rho(A) > target_rho),
    mapped linearly (d -> 20 -> lin_n_basis, weights N(0, lin_weight_sd^2), zero bias, identity
    activation) to B-spline coefficients on [0,1]."""
    rng = np.random.default_rng(cfg.seed)
    d, T = (cfg.lin_d or cfg.d), cfg.T
    Sigma = cfg.Sigma_scale ** 2 * np.eye(d)
    A = rng.normal(size=(d, d)) * 0.15
    rho = np.max(np.abs(np.linalg.eigvals(A)))
    if (cfg.lin_exact_rho or rho > cfg.lin_target_rho) and rho > 1e-12:
        A = A * (cfg.lin_target_rho / rho)
    Tb = T + cfg.burnin
    Z = np.zeros((Tb, d)); Z[0] = rng.multivariate_normal(np.zeros(d), Sigma)
    for t in range(1, Tb):
        Z[t] = A @ Z[t - 1] + rng.multivariate_normal(np.zeros(d), Sigma)
    Z = Z[cfg.burnin:]
    rw = np.random.default_rng(cfg.seed + 7)
    W1 = rw.normal(size=(d, 20)) * cfg.lin_weight_sd
    W2 = rw.normal(size=(20, cfg.lin_n_basis)) * cfg.lin_weight_sd
    Bg = bspline_basis(cfg.lin_n_basis, cfg.P)
    return (Z @ W1 @ W2) @ Bg.T


def simulate(cfg: SimCfg):
    if cfg.map_mode == "nonlinear":
        Xc = _decoder_nonlinear(_latent_nonlinear(cfg), cfg, tr=trend_path(cfg))
    else:
        Xc = _linear_benchmark(cfg)
    rng = np.random.default_rng(cfg.seed + 999)
    Xn = Xc + cfg.meas_noise_sd * rng.normal(size=Xc.shape)
    u = (np.linspace(cfg.age_min, cfg.age_max, cfg.P) if cfg.map_mode == "nonlinear"
         else np.linspace(0.0, 1.0, cfg.P))
    return u, Xc.astype(np.float64), Xn.astype(np.float64)

# ------------------------------------------------------------------ metrics, VAR
def rel_mse(xhat, xref):
    return float(np.mean((xhat - xref) ** 2) / max(np.var(xref), 1e-300))


def fit_var1(H, ridge=1e-6):
    X = np.hstack([np.ones((len(H) - 1, 1)), H[:-1]]); Y = H[1:]
    Bm = np.linalg.solve(X.T @ X + ridge * np.eye(X.shape[1]), X.T @ Y)
    return Bm[0], Bm[1:].T


def forecast_var1(h_last, b, A, steps):
    out, h = [], h_last.copy()
    for _ in range(steps):
        h = b + A @ h; out.append(h)
    return np.array(out)

# ------------------------------------------------------------------ FPCA on a B-dim basis
def fpca_B(Xtr: np.ndarray, u: np.ndarray, K: int, B: int, steps: int, ridge=1e-6):
    """Smooth curves on B cubic B-splines (least squares), weighted FPCA, VAR(1) forecast."""
    Phi = bspline_basis(B, Xtr.shape[1])
    S = Phi @ np.linalg.solve(Phi.T @ Phi, Phi.T)          # hat matrix J x J
    Xs = Xtr @ S.T
    w = trapezoid_weights(u); sw = np.sqrt(w)
    # same raw (unsmoothed) training mean as the FAE; only deviations are smoothed
    mu = Xtr.mean(0); Xc = Xs - Xs.mean(0)
    _, _, Vt = np.linalg.svd(Xc * sw, full_matrices=False)
    phi = (Vt[:K] / sw)                                      # K x J, orthonormal in L2(w)
    phi /= np.sqrt((phi ** 2 * w).sum(1, keepdims=True))
    H = (Xc * w) @ phi.T
    recon = mu + H @ phi
    b, A = fit_var1(H, ridge)
    fore = mu + forecast_var1(H[-1], b, A, steps) @ phi
    return recon, fore

# ------------------------------------------------------------------ batched FAE
@dataclass
class FaeCfg:
    epochs: int = 900
    batch_size: int = 64
    lr: float = 7e-4
    weight_decay: float = 1e-6
    init_weight_sd: float = 0.06
    smooth_lambda: float = 2e-4
    split_rate: float = 0.85
    hidden: int = 128
    eval_every: int = 25
    seed: int = 743


def _init_params(R, B, K, H, sd, gen):
    def W(i, o): return (torch.randn(R, i, o, generator=gen) * sd).requires_grad_()
    def b(o): return torch.zeros(R, 1, o).requires_grad_()
    return [W(B, H), b(H), W(H, K), b(K), W(K, H), b(H), W(H, B), b(B)]


def _forward(p, feat):
    W1, b1, W2, b2, W3, b3, W4, b4 = p
    h = torch.tanh(torch.bmm(feat, W1) + b1)
    z = torch.bmm(h, W2) + b2
    h2 = torch.tanh(torch.bmm(z, W3) + b3)
    coef = torch.bmm(h2, W4) + b4
    return z, coef


def _decode(p, z):
    _, _, _, _, W3, b3, W4, b4 = p
    return torch.bmm(torch.tanh(torch.bmm(z, W3) + b3), W4) + b4


def _d2(coef):
    d = coef[..., 2:] - 2 * coef[..., 1:-1] + coef[..., :-2]
    return (d ** 2).sum(-1).mean(-1)          # per replicate


def _train(Xs, Pmat, Brev, K, cfg, n_epochs, gen, val=None, snap_epochs=None):
    """Train R independent FAEs on Xs (R x n x J). Returns params and/or per-rep snapshots."""
    R, n, J = Xs.shape; B = Pmat.shape[1]
    p = _init_params(R, B, K, cfg.hidden, cfg.init_weight_sd, gen)
    opt = torch.optim.Adam(p, lr=cfg.lr, weight_decay=cfg.weight_decay)
    feat_all = Xs @ Pmat                                   # R x n x B (projection is linear)
    hist = []; best = torch.full((R,), float("inf")); best_ep = torch.zeros(R, dtype=torch.long)
    snaps = [None] * R
    want = None if snap_epochs is None else snap_epochs.clone()
    for ep in range(1, n_epochs + 1):
        perm = torch.argsort(torch.rand(R, n, generator=gen), dim=1)
        for idx in perm.split(cfg.batch_size, dim=1):
            ix = idx.unsqueeze(-1)
            f = torch.gather(feat_all, 1, ix.expand(-1, -1, B))
            x = torch.gather(Xs, 1, ix.expand(-1, -1, J))
            opt.zero_grad()
            _, coef = _forward(p, f)
            xhat = coef @ Brev.T
            loss = (((xhat - x) ** 2).mean((1, 2)) + cfg.smooth_lambda * _d2(coef)).sum()
            loss.backward(); opt.step()
        if val is not None and (ep == 1 or ep % cfg.eval_every == 0 or ep == n_epochs):
            Xv, fv = val
            with torch.no_grad():
                _, cv = _forward(p, fv)
                vm = ((cv @ Brev.T - Xv) ** 2).mean((1, 2))
            imp = vm < best; best = torch.where(imp, vm, best)
            best_ep = torch.where(imp, torch.tensor(ep), best_ep); hist.append((ep, vm.numpy().copy()))
        if want is not None:
            hit = (want == ep).nonzero().flatten().tolist()
            for r in hit:
                snaps[r] = [t[r:r + 1].detach().clone() for t in p]
    return p, best_ep, hist, snaps


def fae_fit_forecast(Xtr_list, u, K, B, steps, cfg: FaeCfg, ridge=1e-6, return_hist=False):
    """Fit R independent FAEs (one per replicate dataset) and return recon & forecasts (orig. scale)."""
    R = len(Xtr_list); n, J = Xtr_list[0].shape
    n_tr = max(8, min(int(cfg.split_rate * n), n - 1))
    # training-only scaling (chronological training sub-block)
    mus = np.stack([X[:n_tr].mean(0) for X in Xtr_list])
    scs = np.array([(X[:n_tr] - X[:n_tr].mean(0)).std() for X in Xtr_list])
    Xs = torch.tensor(np.stack([(X - m) / s for X, m, s in zip(Xtr_list, mus, scs)]), dtype=torch.float32)
    tn = torch.tensor(u, dtype=torch.float32)
    dt = tn[1:] - tn[:-1]; z0 = torch.zeros(1)
    W = 0.5 * torch.cat([z0, dt]) + 0.5 * torch.cat([dt, z0])
    Bb = torch.tensor(bspline_basis(B, J), dtype=torch.float32)
    Pmat = W[:, None] * Bb                                  # J x B  (feat = x @ Pmat)
    # stage 1: epoch selection on chronological split
    gen = torch.Generator().manual_seed(cfg.seed)
    feat_v = Xs[:, n_tr:] @ Pmat
    _, best_ep, hist, _ = _train(Xs[:, :n_tr], Pmat, Bb, K, cfg, cfg.epochs, gen, val=(Xs[:, n_tr:], feat_v))
    # stage 2: refit on all T curves, snapshot each replicate at its own selected epoch
    gen = torch.Generator().manual_seed(cfg.seed)
    _, _, _, snaps = _train(Xs, Pmat, Bb, K, cfg, int(best_ep.max()), gen, snap_epochs=best_ep)
    recon, fore = [], []
    for r in range(R):
        p = snaps[r]
        with torch.no_grad():
            z, coef = _forward(p, (Xs[r:r + 1] @ Pmat))
            rec = (coef @ Bb.T)[0].numpy()
            H = z[0].numpy().astype(np.float64)
            b, A = fit_var1(H, ridge)
            Hf = torch.tensor(forecast_var1(H[-1], b, A, steps), dtype=torch.float32)[None]
            fo = (_decode(p, Hf) @ Bb.T)[0].numpy()
        recon.append(rec * scs[r] + mus[r]); fore.append(fo * scs[r] + mus[r])
    out = dict(recon=recon, fore=fore, best_epoch=best_ep.numpy())
    if return_hist: out["hist"] = hist
    return out


def evaluate_cell(datasets, K, B, fae_cfg, L=5):
    """List of (u, Xclean, Xnoisy). Returns list of per-replicate dicts."""
    u = datasets[0][0]
    Xtr = [d[2][:-L] for d in datasets]
    res = fae_fit_forecast(Xtr, u, K, B, L, fae_cfg)
    rows = []
    for r, (uu, Xc, Xn) in enumerate(datasets):
        frec, ffor = fpca_B(Xn[:-L], u, K, B, L)
        rows.append(dict(K=K, B=B, rep=r,
                         FPCA_recon=rel_mse(frec, Xc[:-L]), FAE_recon=rel_mse(res["recon"][r], Xc[:-L]),
                         FPCA_fore=rel_mse(ffor, Xc[-L:]), FAE_fore=rel_mse(res["fore"][r], Xc[-L:]),
                         best_epoch=int(res["best_epoch"][r])))
    return rows