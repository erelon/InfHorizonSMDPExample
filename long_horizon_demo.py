"""Long-horizon demo (brief + amendment 1): one machinery per story.

Figure 1 -- offset + delivery, sine world (GAE lambda = 0.95, stale score, 48 clipped minibatch steps).
Figure 2 -- width collapse + the controller: one bump grows, then relocates out of reach (amendment 3;
            lambda = 0, one full-batch step per iteration), plus the swap-world epilogue (amendment 2).

Usage
  FIG=1 python long_horizon_demo.py                # run every missing (arm, seed) of figure 1, then plot
  FIG=1 ARM=2 SEED=3 python long_horizon_demo.py   # one run -> runs/fig1/arm2_seed3.npz
  FIG=2 EPI=1 SEED=0 python long_horizon_demo.py  # one epilogue run (swap world, boost arm only)
  FIG=2 PLOT=1 python long_horizon_demo.py         # plot + summary + acceptance from the .npz files only
  FORCE=1 ...                                      # re-run even if the .npz exists

Pure numpy + matplotlib. Seeds 0-4 are shared by all arms (common random numbers): the action noise
and minibatch permutations of iteration t are identical in every arm of a figure.
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = range(5)

# shared machinery constants
N = 256
ALPHA = 0.02
GAMMA = 1.0
BETA_V = 1.0
LS_MIN, LS_MAX = -4.0, 2.5
S_SLOW_MEMORY = 400.0        # amendment 1: 800 -> 400 (both figures)
BOOST_EFOLD = 600.0
BOOST_MAX = np.log(8.0)
STARVE_FRAC = 0.3
GUARD_WINDOW = 200
RECOVERY_WINDOW = 200

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(120)
_GH_W = _GH_W / _GH_W.sum()


def gh_expect(f, mu, sigma):
    """E[f(a)], a ~ N(mu, sigma^2), by Gauss-Hermite quadrature. mu, sigma broadcast."""
    mu, sigma = np.asarray(mu, float), np.asarray(sigma, float)
    return f(mu[..., None] + sigma[..., None] * _GH_X) @ _GH_W


def log_sigma_rising(ls_hist, t):
    w = ls_hist[max(0, t - GUARD_WINDOW + 1):t + 1]
    if len(w) < 2:
        return False
    x = np.arange(len(w)) - (len(w) - 1) / 2.0
    return float(x @ (w - w.mean())) > 0.0


def rolling_mean(x, w):
    c = np.cumsum(np.insert(np.nan_to_num(np.asarray(x, float)), 0, 0.0))
    out = np.full(len(x), np.nan)
    out[w - 1:] = (c[w:] - c[:-w]) / w
    return out


def rolling_median(x, w):
    out = np.full(len(x), np.nan)
    out[w - 1:] = np.median(np.lib.stride_tricks.sliding_window_view(x, w), axis=1)
    return out


def recovery(perf_seed, t_event):
    """Amendment 1 metric: iterations after t_event until the trailing 200-iter mean perf exceeds
    0.9 x its own mean over the 1000 iterations before t_event. inf = never.
    The trailing window must lie entirely after t_event (a window straddling the event would contain
    pre-event iterations and "recover" at t_event + 1 by construction)."""
    target = 0.9 * perf_seed[t_event - 1000:t_event].mean()
    rm = rolling_mean(perf_seed, RECOVERY_WINDOW)
    first = t_event + RECOVERY_WINDOW          # rm[first] averages [t_event + 1, t_event + 200]
    hit = np.where(rm[first:] > target)[0]
    return float(hit[0] + RECOVERY_WINDOW) if len(hit) else np.inf


def fmt_t(x):
    return "never" if np.isinf(x) else str(int(x))


class Checks:
    def __init__(self):
        self.lines = []
        self.n_fail = 0

    def __call__(self, tag, name, ok, detail):
        ok = bool(ok)
        self.n_fail += not ok
        self.lines.append(f"[{'PASS' if ok else 'FAIL'}] [{tag}] {name}: {detail}")


# ====================================================================== FIGURE 1: sine world
class Fig1:
    ITERS = 12000
    T_B, T_HOLD = 4000, 9000
    M = 32
    EPOCHS = 6
    LR = 0.025
    LAMBDA = 0.95
    P = {1: 0.05, 2: 0.2, 3: 0.2}
    ARMS = (1, 2, 3)
    NAMES = {1: "no fixes", 2: "frozen dose", 3: "tracked dose"}
    COLORS = {1: "#b2182b", 2: "#e08214", 3: "#2166ac"}
    MU0, SIGMA0 = 0.0, 0.35
    RUN_DIR = os.path.join(HERE, "runs", "fig1")
    FIG_PATH = os.path.join(HERE, "long_horizon_fig1.png")
    TABLE_PATH = os.path.join(HERE, "summary_fig1.txt")
    KEYS = ("b", "b_pred", "b_centred", "phi", "stdA", "s_slow", "sigma", "mu", "cH", "pressure", "perf", "cos")

    _k = np.arange(N)
    L = np.triu(LAMBDA ** np.clip(_k[None, :] - _k[:, None], 0, None))   # A = L @ delta
    C_N = L.sum(axis=1).mean()      # exact finite-batch offset factor (18.46; asymptote 1/(1-lambda) = 20)

    @classmethod
    def schedule(cls):
        t = np.arange(cls.ITERS, dtype=float)
        m = np.clip(1.0 + 4.0 * (t - cls.T_B) / (cls.T_HOLD - cls.T_B), 1.0, 5.0)
        g = 0.8 * np.sin(2 * np.pi * t / 3000.0)
        ramp = np.where(t < 800, 0.003 * t ** 2 / 1600.0, 0.003 * (t - 400.0))
        return m, g, ramp

    @staticmethod
    def reward(a, g, m, ramp):
        # amendment 1 (option B): the ramp is no longer scaled by m
        return m * (1.0 - np.minimum((a - g) ** 2, 4.0)) + ramp

    @classmethod
    def run(cls, arm, seed):
        m_t, g_t, ramp_t = cls.schedule()
        rng = np.random.default_rng(seed)
        log = {k: np.zeros(cls.ITERS) for k in cls.KEYS}
        mu, ls, phi = cls.MU0, np.log(cls.SIGMA0), 0.0
        num = den = 0.0
        s_slow = std0 = 0.0
        n_mb = N // cls.M
        for t in range(cls.ITERS):
            m, g, ramp = m_t[t], g_t[t], ramp_t[t]
            z = rng.standard_normal(N)
            perms = np.stack([rng.permutation(N) for _ in range(cls.EPOCHS)]).reshape(cls.EPOCHS * n_mb, cls.M)
            sigma = np.exp(ls)
            a = mu + sigma * z
            r = cls.reward(a, g, m, ramp)
            if t == 0:
                num, den = r.mean(), 1.0          # gain EMA seeded from batch 1
            eta = num / den
            eps_pred = gh_expect(lambda x: cls.reward(x, g, m, ramp), mu, sigma) - eta

            delta = r - eta + (GAMMA - 1.0) * phi
            A = cls.L @ delta
            b, stdA = A.mean(), A.std()
            if t == 0:
                std0 = s_slow = stdA
            else:
                s_slow += (stdA - s_slow) / S_SLOW_MEMORY
            cH = cls.P[arm] * (s_slow if arm == 3 else std0)

            phi = phi + BETA_V * (np.mean(A + phi) - phi)       # scalar critic: integrator of mean(A)

            A_used = A - A.mean() if arm >= 2 else A
            Ap, zp = A_used[perms], z[perms]                    # stale score: z from pre-update params
            g_mu = (Ap * zp).mean(axis=1) / sigma
            g_ls = (Ap * (zp ** 2 - 1.0)).mean(axis=1) + cH
            scale = np.minimum(1.0, 1.0 / np.maximum(np.hypot(g_mu, g_ls), 1e-12))
            mu0, ls0 = mu, ls
            for smu, sls in zip(cls.LR * g_mu * scale, cls.LR * g_ls * scale):
                mu += smu
                ls = min(max(ls + sls, LS_MIN), LS_MAX)

            d_upd = np.array([mu - mu0, ls - ls0])
            d_true = np.array([-2.0 * m * (mu0 - g), -2.0 * m * sigma ** 2])
            cos = d_upd @ d_true / (np.linalg.norm(d_upd) * np.linalg.norm(d_true) + 1e-300)

            if t > 0:
                num = (1 - ALPHA) * num + ALPHA * r.mean()
                den = (1 - ALPHA) * den + ALPHA

            for k, v in (("b", b), ("b_pred", (eps_pred + (GAMMA - 1.0) * phi) * cls.C_N),
                         ("b_centred", A_used.mean()), ("phi", phi), ("stdA", stdA), ("s_slow", s_slow),
                         ("sigma", sigma), ("mu", mu0), ("cH", cH), ("pressure", cH / stdA),
                         ("perf", 1.0 - np.mean(np.minimum((a - g) ** 2, 4.0))), ("cos", cos)):
                log[k][t] = v
        return log

    # ---------------------------------------------------------------- analysis
    @classmethod
    def balance(cls, cH):
        m, _, _ = cls.schedule()
        return np.sqrt(cH / (2.0 * m))

    @classmethod
    def summary(cls, data):
        phases = {"A": (0, cls.T_B), "B ramp": (cls.T_B, cls.T_HOLD), "B hold": (cls.T_HOLD, cls.ITERS)}
        w = 13
        hdr = f"{'arm':<14}" + "".join(f"{'perf ' + p:<{w}}" for p in phases) + \
              "".join(f"{'press ' + p:<{w + 1}}" for p in phases) + "".join(f"{'sigma ' + p:<{w + 1}}" for p in phases)
        lines = [hdr, "-" * len(hdr)]
        for arm in cls.ARMS:
            d = data[arm]
            row = f"{cls.NAMES[arm]:<14}"
            row += "".join(f"{np.mean(d['perf'][:, lo:hi]):<{w}.3f}" for lo, hi in phases.values())
            row += "".join(f"{np.median(d['pressure'][:, lo:hi]):<{w + 1}.3f}" for lo, hi in phases.values())
            row += "".join(f"{np.median(d['sigma'][:, lo:hi]):<{w + 1}.3f}" for lo, hi in phases.values())
            lines.append(row)
        lines += ["",
                  "perf = 1 - mean(min((a-g)^2,4)) (= (mean r - ramp)/m: current-phase units); "
                  "press = median realised pressure c_H/std(A) (searched 0.2).",
                  "Windows: A [0,4000)  B ramp [4000,9000) m: 1->5  B hold [9000,12000) m = 5.  "
                  "No event in figure 1, so no recovery metric."]
        return "\n".join(lines)

    @classmethod
    def acceptance(cls, data):
        c = Checks()
        med = lambda x: np.median(x, axis=0)
        d1, d2, d3 = (data[a] for a in cls.ARMS)
        TB, TH, TE = cls.T_B, cls.T_HOLD, cls.ITERS

        p1 = rolling_mean(med(d1["perf"]), 100)
        dead = np.where(p1 < -1)[0]
        c("orig", "no fixes dead by t ~ 2000 (perf < -1)", len(dead) and dead[0] <= 2500,
          f"first t where 100-iter mean of median perf < -1: {dead[0] if len(dead) else 'never'}")
        e = d1["b"][:, 200:TB] - d1["b_pred"][:, 200:TB]
        inband = np.mean(np.abs(e) <= 2 * e.std(axis=1, keepdims=True))
        bias = abs(e.mean()) / np.mean(np.abs(d1["b"][:, 200:TB]))
        c("orig", "no fixes: b on its prediction (within noise band)", inband >= 0.9 and bias < 0.05,
          f"phase A: {100 * inband:.1f}% of iterations within the 2-sd band; bias = {100 * bias:.2f}% of mean |b|")
        c("orig", "no fixes: phi unbounded", np.all(np.abs(d1["phi"][:, -1]) > 1e4),
          f"|phi(end)| per seed = {np.abs(d1['phi'][:, -1]).round(0).astype(int).tolist()}")
        for arm in cls.ARMS:
            d = data[arm]
            rel = np.median(np.abs(d["phi"][:, -1] / np.cumsum(d["b_pred"], axis=1)[:, -1] - 1))
            c("orig", f"{cls.NAMES[arm]}: phi on its integrator line", rel < 0.05,
              f"median |phi / cumsum(b_pred) - 1| at end = {100 * rel:.2f}%")
        for arm in (2, 3):
            e = data[arm]["b_centred"]
            c("orig", f"{cls.NAMES[arm]}: after centring the policy sees mean 0 exactly", np.max(np.abs(e)) < 1e-9,
              f"max |mean(A_used)| = {np.max(np.abs(e)):.1e}")

        pr2 = np.median(d2["pressure"][:, TH - 200:TH])
        c("adapted", "frozen dose: realised pressure at end of ramp <= 0.06 (was: 'at C'; no C in fig 1)",
          pr2 <= 0.06, f"median over [8800,9000) = {pr2:.4f} ({0.2 / pr2:.1f}x under-delivery)")
        pr3 = rolling_mean(med(d3["pressure"]), 200)
        mask = np.zeros(TE, bool)
        mask[200:TE] = True
        mask[TB:TB + 400] = False
        dev_ab = np.nanmax(np.abs(pr3[mask] / 0.2 - 1))
        dev_hold = np.nanmax(np.abs(pr3[TH + 200:TE] / 0.2 - 1))
        c("revised", "tracked dose: pressure within 30% of 0.2 over A-B except [4000,4400)", dev_ab <= 0.30,
          f"worst 200-iter-mean deviation = {100 * dev_ab:.1f}%")
        c("revised", "tracked dose: pressure within 15% of 0.2 during the m = 5 hold", dev_hold <= 0.15,
          f"worst 200-iter-mean deviation in [9200,12000) = {100 * dev_hold:.1f}%")
        for arm in (2, 3):
            d = data[arm]
            pred = cls.balance(med(d["cH"]))
            parts, ok = [], True
            for name, (lo, hi) in (("A", (2000, TB)), ("ramp end", (8000, TH)), ("hold", (TH + 500, TE))):
                ratio = np.median(med(d["sigma"])[lo:hi] / pred[lo:hi])
                parts.append(f"{name} {ratio:.2f}")
                ok &= abs(ratio - 1) <= 0.25
            c("orig", f"{cls.NAMES[arm]}: sigma plateau within 25% of balance", ok,
              "measured/predicted: " + ", ".join(parts))
        c.lines.append("Moved to figure 2 by amendment 1: all recovery-after-C and boost criteria.")
        return c

    # ---------------------------------------------------------------- figure
    @classmethod
    def figure(cls, data, plt):
        t = np.arange(cls.ITERS)
        med = lambda x: np.median(x, axis=0)
        titles = ["1. size of the offset |b|\n(50-iter running median;\ndashed: prediction)",
                  "2. critic phi\n(dashed: cumsum of\npredicted b)",
                  "3. realised pressure\nc_H / std(A)",
                  "4. policy width sigma\n(dashed: balance\nsqrt(c_H / 2m))",
                  "5. tracking performance\n1 - mean(min((a-g)^2, 4))",
                  "6. direction quality\ncos(update, true grad)"]
        fig, axes = plt.subplots(len(titles), 3, figsize=(18, 21), sharex=True)
        for j, arm in enumerate(cls.ARMS):
            d, col = data[arm], cls.COLORS[arm]
            axes[0, j].set_title(cls.NAMES[arm], fontsize=15, fontweight="bold", color=col, pad=16)

            ax = axes[0, j]
            ax.plot(t, rolling_median(med(np.abs(d["b"])), 50), color=col, lw=1.6, label="|b| measured")
            ax.plot(t, rolling_median(med(np.abs(d["b_pred"])), 50), color="k", lw=1.0, ls="--",
                    label="|b| predicted = eps * C_N")
            ax.set_yscale("log")
            if arm >= 2:
                ax.text(0.98, 0.05, f"after centring the policy sees\nmean(A) = 0 exactly "
                                    f"(max {np.max(np.abs(d['b_centred'])):.0e})",
                        transform=ax.transAxes, ha="right", fontsize=8.5, fontweight="bold",
                        bbox=dict(fc="white", ec="k", lw=0.8))
            else:
                ax.text(0.98, 0.05, "no centring: the policy sees\nthe full offset",
                        transform=ax.transAxes, ha="right", fontsize=8.5, bbox=dict(fc="white", ec=col, lw=0.8))
            ax.legend(fontsize=8, loc="upper left")

            ax = axes[1, j]
            ax.plot(t, med(d["phi"]), color=col, lw=1.6, label="phi (median)")
            ax.plot(t, med(np.cumsum(d["b_pred"], axis=1)), color="k", lw=1.0, ls="--", label="predicted integrator")
            ax.set_yscale("symlog", linthresh=10.0)
            ax.text(0.98, 0.05, "drift cancels in the residual: harmless\n(gamma = 1: phi never enters delta)",
                    transform=ax.transAxes, ha="right", fontsize=8)
            ax.legend(fontsize=8, loc="upper left")

            ax = axes[2, j]
            ax.plot(t, med(d["pressure"]), color=col, lw=0.5, alpha=0.3)
            ax.plot(t, rolling_mean(med(d["pressure"]), 50), color=col, lw=1.6, label="realised (50-iter mean)")
            ax.axhline(0.2, color="k", lw=1.0, ls="--", label="searched p = 0.2")
            if arm == 3:
                ax.axhspan(0.2 * 0.85, 0.2 * 1.15, color="0.85", zorder=0, label="+-15% (hold criterion)")
            ax.axhline(0.06, color="0.5", lw=0.7, ls="-.", label="0.06 = 3x under-delivery")
            ax.set_yscale("log")
            ax.set_ylim(3e-3, 1.0)
            ax.legend(fontsize=8, loc="lower left")

            ax = axes[3, j]
            ax.plot(t, med(d["sigma"]), color=col, lw=1.4, label="sigma (median)")
            ax.plot(t, cls.balance(med(d["cH"])), color="k", lw=1.0, ls="--",
                    alpha=1.0 if arm >= 2 else 0.5,
                    label="balance prediction" + ("" if arm >= 2 else " (valid only while alive)"))
            ax.set_yscale("log")
            ax.set_ylim(1e-2, 3)
            ax.legend(fontsize=8, loc="upper left")
            if arm == 2:
                ax.text(0.98, 0.04, "measured noise-floor effect: once c_H is tiny, (z^2-1) gradient noise\n"
                                    "random-walk-widens sigma ~1.3x above the deterministic balance\n"
                                    "(side run at half LR: excess 1.32x -> 1.13x). Stated, not tuned.",
                        transform=ax.transAxes, ha="right", fontsize=7.5, bbox=dict(fc="white", ec=col, lw=0.8))

            ax = axes[4, j]
            for s in range(d["perf"].shape[0]):
                ax.plot(t, d["perf"][s], color=col, lw=0.3, alpha=0.15)
            ax.plot(t, rolling_mean(med(d["perf"]), 50), color=col, lw=1.6, label="median perf (50-iter mean)")
            ax.axhline(1.0, color="k", lw=1.0, ls="--", label="ideal 1.0")
            ax.axhline(-3.0, color="0.5", lw=0.7, ls="-.", label="floor -3")
            ax.set_ylim(-3.3, 1.2)
            ax.legend(fontsize=8, loc="lower left")

            ax = axes[5, j]
            ax.plot(t, med(d["cos"]), color=col, lw=0.4, alpha=0.3)
            ax.plot(t, rolling_mean(med(d["cos"]), 50), color=col, lw=1.6, label="cos (50-iter mean)")
            ax.axhline(1.0, color="k", lw=1.0, ls="--", label="ideal 1")
            ax.axhline(0.0, color="0.3", lw=0.6)
            ax.set_ylim(-1.05, 1.05)
            ax.legend(fontsize=8, loc="lower left")
            ax.set_xlabel("iteration")

            for i in range(len(titles)):
                for x in (cls.T_B, cls.T_HOLD):
                    axes[i, j].axvline(x, color="0.5", lw=0.8, ls=":")
                if j == 0:
                    axes[i, j].set_ylabel(titles[i], fontsize=10.5)
            for x, lab in ((cls.T_B / 2, "A: stable"), ((cls.T_B + cls.T_HOLD) / 2, "B: reward scale x1 -> x5"),
                           ((cls.T_HOLD + cls.ITERS) / 2, "B: hold x5")):
                axes[0, j].text(x, 1.01, lab, transform=axes[0, j].get_xaxis_transform(), ha="center",
                                fontsize=8, color="0.35", va="bottom")
        fig.suptitle("Figure 1 - offset and dose delivery (sine-tracking world, median of 5 seeds; dashed = theory)",
                     fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(cls.FIG_PATH, dpi=110)
        return cls.FIG_PATH


# ====================================================================== FIGURE 2: bump worlds
class Fig2:
    """Amendment 3. world = "relocate" (main figure: one bump that grows, then vanishes and reappears out of
    reach) or "swap" (epilogue, kept exactly as delivered in amendment 2: two bumps swap heights at the
    midpoint, boost arm only, 0.25x floor)."""
    STEP = 0.05
    GRAD_CAP = 5.0
    P = 0.8
    WIDTH = 0.3
    SETTLE = 2000                 # settled reference: the starvation peak accumulates only for t >= SETTLE
    MU0, SIGMA0 = -0.7, 0.3
    NAMES = {1: "frozen dose", 2: "tracked, no floor", 3: "tracked + warmup floor", 4: "tracked + floor + boost"}
    COLORS = {1: "#e08214", 2: "#b2182b", 3: "#2166ac", 4: "#1b7837"}
    KEYS = ("sigma", "mu", "perf", "stdA", "s_slow", "peak", "cH", "pressure", "boost", "starving", "rising",
            "floor", "phi")
    RECOVERY_FRAC = 0.8           # amendment 3: threshold = 0.8 x the post-event ideal

    M = S_SLOW_MEMORY             # slow-EMA memory (iterations)
    B_MAX = 8.0                   # boost cap
    BOOST_FLOOR = True            # does the boost arm carry the warmup floor?

    def __init__(self, world="relocate", c_new=None, m=None, t_ref=None, b_max=None, run_dir=None):
        self.world = world
        t = np.arange(20000 if world == "relocate" else 8000, dtype=float)
        self.ITERS = len(t)
        if world == "relocate":
            # amendment 5: phase A [0, 6000) bump at -0.7; at t = 6000 it reappears at C_NEW; reward noise eta;
            # the boost rides on the UNDER-dosed (no-floor) arm; M, t_ref, B_max, c solved from I1-I4
            self.ARMS = (1, 2, 3, 4)
            self.NAMES = {**self.NAMES, 4: "tracked, no floor + boost"}
            self.T_EVENT = 6000
            self.BOUNDS = (self.T_EVENT,)
            self.FLOOR_FRAC = 1.0                     # floor at the FULL warmup (batch-1) scale
            self.BOOST_FLOOR = False
            self.NOISE = 0.03
            self.C_NEW = 1.5 if c_new is None else c_new
            self.M = 250.0 if m is None else m
            self.SETTLE = 500 if t_ref is None else t_ref
            self.B_MAX = 2.0 if b_max is None else b_max
            self.RUN_DIR = run_dir or os.path.join(HERE, "runs", "fig2")
            self.h1 = np.ones_like(t)
            self.c1 = np.where(t < self.T_EVENT, -0.7, self.C_NEW)
            self.h2 = np.zeros_like(t)
            self.k = np.full_like(t, 0.02)
        else:
            self.ARMS = (4,)
            self.NOISE = 0.0                          # epilogue unchanged, as delivered
            self.T_EVENT = 4000
            self.BOUNDS = (self.T_EVENT,)
            self.FLOOR_FRAC = 0.25                    # as delivered in amendment 2
            self.RUN_DIR = os.path.join(HERE, "runs", "fig2_epilogue")
            self.h1 = np.where(t < 4000, 1.0, 0.6)
            self.h2 = np.where(t < 4000, 0.6, 1.3)
            self.c1 = np.full_like(t, -0.7)
            self.k = np.full_like(t, 0.1)
        self.FIG_PATH = os.path.join(HERE, "long_horizon_fig2.png")
        self.TABLE_PATH = os.path.join(HERE, "summary_fig2.txt")

    @classmethod
    def reward(cls, a, h1, h2, k, c1):
        w2 = 2 * cls.WIDTH ** 2
        return h1 * np.exp(-(a - c1) ** 2 / w2) + h2 * np.exp(-(a - 0.7) ** 2 / w2) - k * a ** 2

    def env(self, i):
        return self.h1[i], self.h2[i], self.k[i], self.c1[i]

    def run(self, arm, seed):
        rng = np.random.default_rng(seed)
        log = {k: np.zeros(self.ITERS) for k in self.KEYS}
        mu, ls, phi = self.MU0, np.log(self.SIGMA0), 0.0
        num = den = 0.0
        s_slow = std0 = 0.0
        peak = np.nan
        log_boost = 0.0
        ls_hist = np.zeros(self.ITERS)
        beta = 1.0 / BOOST_EFOLD
        for t in range(self.ITERS):
            z = rng.standard_normal(N)
            sigma = np.exp(ls)
            ls_hist[t] = ls
            a = mu + sigma * z
            r = self.reward(a, *self.env(t))
            if self.NOISE > 0:                          # amendment 4: exogenous i.i.d. reward noise
                r = r + self.NOISE * rng.standard_normal(N)
            if t == 0:
                num, den = r.mean(), 1.0
            eta = num / den
            A = r - eta + (GAMMA - 1.0) * phi          # lambda = 0: per-sample advantage, no batch shape
            stdA = A.std()
            if t == 0:
                std0 = s_slow = stdA
            else:
                s_slow += (stdA - s_slow) / self.M
            if t >= self.SETTLE:                        # settled reference
                peak = s_slow if np.isnan(peak) else max(peak, s_slow)
                starving = s_slow < STARVE_FRAC * peak
            else:
                starving = False
            rising = log_sigma_rising(ls_hist, t)
            floor = False
            if arm == 1:
                cH = self.P * std0
            elif arm == 2 or (arm == 4 and not self.BOOST_FLOOR):
                cH = self.P * s_slow
            else:
                floor = s_slow < self.FLOOR_FRAC * std0
                cH = self.P * max(s_slow, self.FLOOR_FRAC * std0)
            if arm == 4:
                if t < self.SETTLE:
                    log_boost = 0.0
                elif starving and not rising:
                    log_boost += beta
                elif not starving:
                    log_boost -= beta
                # starving and rising: HOLD
                log_boost = min(max(log_boost, 0.0), np.log(self.B_MAX))
                cH *= np.exp(log_boost)

            phi = phi + BETA_V * (np.mean(A + phi) - phi)
            Au = A - A.mean()                           # centring ON in all arms
            grad = np.array([np.mean(Au * z) / sigma, np.mean(Au * (z ** 2 - 1.0)) + cH])   # score at current params
            step = self.STEP * grad / max(1.0, np.linalg.norm(grad) / self.GRAD_CAP)
            mu0 = mu
            mu += step[0]
            ls = min(max(ls + step[1], LS_MIN), LS_MAX)

            if t > 0:
                num = (1 - ALPHA) * num + ALPHA * r.mean()
                den = (1 - ALPHA) * den + ALPHA

            for k, v in (("sigma", sigma), ("mu", mu0), ("perf", r.mean()), ("stdA", stdA), ("s_slow", s_slow),
                         ("peak", peak), ("cH", cH), ("pressure", cH / stdA),
                         ("boost", np.exp(log_boost) if arm == 4 else np.nan), ("starving", starving),
                         ("rising", rising), ("floor", floor), ("phi", phi)):
                log[k][t] = v
        return log

    # ---------------------------------------------------------------- theory
    _SIG = np.logspace(-2.5, 0.7, 900)
    _A = np.linspace(-1.5, 3.5, 5001)

    def optimum(self, idx):
        """Best action (argmax of r) and its reward at iterations idx."""
        idx = np.asarray(idx)
        r = self.reward(self._A[None, :], *(v[idx, None] for v in (self.h1, self.h2, self.k, self.c1)))
        j = np.argmax(r, axis=1)
        return self._A[j], r[np.arange(len(idx)), j]

    def contraction(self, mu, sigma, i):
        """kappa = -sigma dE[r]/dsigma at (mu, sigma): the contracting force on log sigma."""
        f = lambda x: self.reward(x, *self.env(i))
        return -(gh_expect(f, mu, sigma * 1.001) - gh_expect(f, mu, sigma / 1.001)) / (2 * np.log(1.001))

    def sigma_star(self, cH, idx):
        """Balance kappa(sigma*) = c_H at the current optimum; first crossing = the stable width."""
        mu_opt, _ = self.optimum(idx)
        s = self._SIG
        out = np.full(len(idx), np.nan)
        for n, i in enumerate(idx):
            W = self.contraction(np.full_like(s, mu_opt[n]), s, i)
            above = W >= cH[n]
            if above.any():
                out[n] = s[np.argmax(above)]
        return out

    def held_width_reward(self, sigma):
        idx = np.arange(self.ITERS)
        mu_opt, _ = self.optimum(idx)
        x = mu_opt[:, None] + sigma[:, None] * _GH_X
        return self.reward(x, *(v[:, None] for v in (self.h1, self.h2, self.k, self.c1))) @ _GH_W

    # ---------------------------------------------------------------- analysis
    def phases(self):
        if self.world == "relocate":
            return {"A": (0, self.T_EVENT), "D0-2k": (self.T_EVENT, self.T_EVENT + 2000),
                    "D2k-end": (self.T_EVENT + 2000, self.ITERS)}
        return {"before swap": (0, self.T_EVENT), "after swap": (self.T_EVENT, self.ITERS)}

    CEILING = (1.0 * 0.3 / (2 * 0.02)) ** (1 / 3)      # acquisition ceiling (h w / 2k)^(1/3)

    def closed_forms(self, data):
        """Noise-corrected predictions (amendments 4-5)."""
        eta, p, k = self.NOISE, self.P, 0.02
        cH = np.median(data[1]["cH"])                                    # frozen dose (= floor scale)
        R2 = 2 * (1 / (2 * self.WIDTH ** 2) + k)                           # |R''| on the bump
        x_nf = p * eta / np.sqrt(1 - p ** 2 / 2)                           # live-tracked fixed point
        ref4 = np.nanmedian(data[4]["peak"][:, self.SETTLE])
        thr4 = STARVE_FRAC * ref4
        x_reg = np.sqrt(2) * np.sqrt(max(thr4 ** 2 - eta ** 2, 0.0))
        return dict(cH=cH, ref=np.sqrt(eta ** 2 + (cH / np.sqrt(2)) ** 2), press=cH / np.sqrt(eta ** 2 + cH ** 2 / 2),
                    sigma2=np.sqrt(x_nf / R2), ref4=ref4, thr4=thr4,
                    sigma_reg_bump=np.sqrt(x_reg / R2), boost_reg=x_reg / (p * thr4),
                    sigma_reg_flat=np.sqrt(np.sqrt(max(thr4 ** 2 - eta ** 2, 0.0)) / (np.sqrt(2) * k)),
                    sigma_flat_frozen=np.sqrt(cH / (2 * k)),
                    sigma_flat_nofloor=(p ** 2 * eta ** 2 / (4 * k ** 2 - 2 * k ** 2 * p ** 2)) ** 0.25)

    def recovery(self, perf_seed):
        """Relocate world (amendment 3): iterations after the event until the trailing 200-iter mean reward,
        window entirely post-event, exceeds 0.8 x the post-event ideal. Swap world: amendment-1 metric."""
        if self.world == "swap":
            return recovery(perf_seed, self.T_EVENT)
        target = self.RECOVERY_FRAC * self.optimum([self.ITERS - 1])[1][0]
        rm = rolling_mean(perf_seed, RECOVERY_WINDOW)
        hit = np.where(rm[self.T_EVENT + RECOVERY_WINDOW:] > target)[0]
        return float(hit[0] + RECOVERY_WINDOW) if len(hit) else np.inf

    def stats(self, data):
        out = {}
        T = self.T_EVENT
        for arm in self.ARMS:
            d = data[arm]
            rec = np.array([self.recovery(p) for p in d["perf"]])
            o = {"rec": rec, "rec_med": np.median(rec)}
            if arm == 4:
                fire, bmax, dec = [], [], []
                for s in range(d["boost"].shape[0]):
                    f = np.where(d["starving"][s][T:] > 0)[0]
                    fire.append(float(f[0]) if len(f) else np.inf)
                    t_rec = self.ITERS if np.isinf(rec[s]) else int(T + rec[s])
                    bmax.append(np.nanmax(d["boost"][s][T:t_rec]) if t_rec > T else 1.0)
                    if np.isinf(rec[s]):
                        dec.append(np.inf)
                    else:
                        below = np.where(d["boost"][s][t_rec:] < 1.3)[0]
                        dec.append(float(below[0]) if len(below) else np.inf)
                o.update(first_fire=np.array(fire), boost_max=np.array(bmax), decay=np.array(dec))
            out[arm] = o
        return out

    def summary(self, data, st):
        ph = self.phases()
        w = 12
        hdr = f"{'arm':<26}" + "".join(f"{'rew ' + p:<{w}}" for p in ph) + "".join(f"{'sigma ' + p:<{w}}" for p in ph) + \
              "".join(f"{'press ' + p:<{w}}" for p in ph) + "".join(f"{'floor% ' + p:<{w}}" for p in ph) + "recovery (per seed)"
        lines = [hdr, "-" * len(hdr)]
        for arm in self.ARMS:
            d = data[arm]
            row = f"{self.NAMES[arm]:<26}"
            row += "".join(f"{np.mean(d['perf'][:, lo:hi]):<{w}.3f}" for lo, hi in ph.values())
            row += "".join(f"{np.median(d['sigma'][:, lo:hi]):<{w}.3f}" for lo, hi in ph.values())
            row += "".join(f"{np.median(d['pressure'][:, lo:hi]):<{w}.3f}" for lo, hi in ph.values())
            row += "".join(f"{100 * np.mean(d['floor'][:, lo:hi]):<{w}.0f}" for lo, hi in ph.values())
            row += f"med {fmt_t(st[arm]['rec_med'])} [{','.join(fmt_t(x) for x in st[arm]['rec'])}]"
            lines.append(row)
        s4, d4, T = st[4], data[4], self.T_EVENT
        lines += ["", f"boost arm: starving iterations in [2000, {T}) per seed = "
                      f"{d4['starving'][:, self.SETTLE:T].sum(axis=1).astype(int).tolist()}; first fire after "
                      f"t = {T}: {[fmt_t(x) for x in s4['first_fire']]}; boost max between event and recovery: "
                      f"{np.round(s4['boost_max'], 2).tolist()}"]
        if self.world == "relocate":
            cf = self.closed_forms(data)
            ideal = self.optimum([self.ITERS - 1])[1][0]
            lines.append(f"constants (solved from I1-I4, see fig2_preregistration_amendment5.md): M = {self.M:g}, "
                         f"t_ref = {self.SETTLE}, B_max = {self.B_MAX:g}, c = {self.C_NEW:g}, no skirt")
            lines.append(f"boost arm reference {cf['ref4']:.4f}, threshold {cf['thr4']:.4f}, noise eta {self.NOISE}; "
                         f"regulated band on the bump: sigma {cf['sigma_reg_bump']:.4f}, boost {cf['boost_reg']:.3f}; "
                         f"flat widths: frozen {cf['sigma_flat_frozen']:.2f}, no-floor {cf['sigma_flat_nofloor']:.3f}, "
                         f"boost (trigger-limited) {cf['sigma_reg_flat']:.3f}; acquisition ceiling {self.CEILING:.3f}")
            lines.append(f"Recovery = iterations after t = {self.T_EVENT} until the trailing 200-iter mean reward (window "
                         f"entirely post-event) exceeds {self.RECOVERY_FRAC} x post-event ideal {ideal:.3f} = "
                         f"{self.RECOVERY_FRAC * ideal:.3f}.")
        return "\n".join(lines)

    def acceptance(self, data, st):
        c = Checks()
        if self.world == "swap":
            d = data[4]
            n = d["starving"][:, self.SETTLE:].sum(axis=1)
            c("new2", "epilogue: trigger silent after settling (trapped policy looks healthy)", np.all(n == 0),
              f"starving iterations in [2000, 8000) per seed = {n.astype(int).tolist()}")
            c("new2", "epilogue: reward stays stuck after the swap", np.all(np.isinf(st[4]["rec"])),
              f"recovery per seed = {[fmt_t(x) for x in st[4]['rec']]}")
            return c
        T, TR = self.T_EVENT, self.SETTLE
        cf = self.closed_forms(data)
        d1, d2, d3, d4 = (data[a] for a in self.ARMS)
        refs = d4["peak"][:, TR]
        c("new5", "I1: boost-arm reference taken while healthy (>= 0.15)", np.all(refs >= 0.15),
          f"reference per seed = {np.round(refs, 4).tolist()}")
        thr = STARVE_FRAC * refs
        sticky = []
        for s in range(len(refs)):
            e = d2["s_slow"][s, TR:T]
            below = e < thr[s]
            sticky.append(bool(below.any() and below[np.argmax(below):].all()))
        c("new5", "I2: unaided no-floor collapse holds its EMA below the threshold until the event", all(sticky),
          f"per seed = {sticky}; settled no-floor std(A) {np.median(d2['stdA'][:, 4000:T]):.4f} vs threshold "
          f"{np.median(thr):.4f}")
        sA = {a: np.median(data[a]["sigma"][:, 4000:T]) for a in self.ARMS}
        n1, n3 = int(d1["starving"][:, TR:T].sum()), int(d3["starving"][:, TR:T].sum())
        c("new5", "A: frozen and warmup-floor arms identical and silent",
          abs(sA[1] / sA[3] - 1) <= 0.02 and n1 == 0 and n3 == 0,
          f"median sigma {sA[1]:.4f} / {sA[3]:.4f}; starving iterations {n1} / {n3}")
        fire = np.array([TR + int(np.argmax(d4["starving"][s, TR:])) for s in range(d4["starving"].shape[0])])
        c("new5", "A: boost arm fires at the predicted t = 1923 +- 300", abs(np.median(fire) - 1923) <= 300,
          f"first fire per seed = {fire.tolist()} (median {np.median(fire):.0f})")
        b_reg = np.median(d4["boost"][:, 4000:T])
        c("new5", "A: boost arm regulated in the predicted band (I4: sigma and boost within 25%)",
          abs(sA[4] / cf["sigma_reg_bump"] - 1) <= 0.25 and abs(b_reg / cf["boost_reg"] - 1) <= 0.25,
          f"sigma {sA[4]:.4f} vs {cf['sigma_reg_bump']:.4f}; boost {b_reg:.3f} vs {cf['boost_reg']:.3f}")
        c("new4", "A: unaided no-floor arm at the noise-rescued width", abs(sA[2] / cf["sigma2"] - 1) <= 0.25,
          f"{sA[2]:.4f} vs {cf['sigma2']:.4f}")
        crash = {a: np.mean(data[a]["perf"][:, T:T + 100]) for a in self.ARMS}
        c("new5", "event: reward crashes in every arm (< 0.2 over the first 100 iters)",
          all(v < 0.2 for v in crash.values()), ", ".join(f"{self.NAMES[a]} {v:.3f}" for a, v in crash.items()))
        r = {a: st[a]["rec_med"] for a in self.ARMS}
        if np.isfinite(r[4]):
            ratio = r[2] / r[4]
        else:
            ratio = np.nan
        c("new5", "recovery: boost arm >= 3x faster than the unaided no-floor arm (PREDICTED TO FAIL: 1.4-2.0)",
          np.isfinite(r[4]) and ratio >= 3,
          f"no-floor {fmt_t(r[2])} [{', '.join(fmt_t(x) for x in st[2]['rec'])}] vs boost {fmt_t(r[4])} "
          f"[{', '.join(fmt_t(x) for x in st[4]['rec'])}] -> ratio {ratio:.2f}")
        c("new5", "recovery: speed ratio inside the pre-registered 1.4-2.0 prediction",
          bool(np.isfinite(ratio) and 1.4 <= ratio <= 2.0), f"ratio {ratio:.2f}")
        fin = [x for a in (1, 3) for x in st[a]["rec"] if np.isfinite(x)]
        c("new5", "recovery: boost arm faster than any well-dosed arm that recovers at all",
          bool(np.isfinite(r[4]) and all(r[4] < x for x in fin)),
          "; ".join(f"{self.NAMES[a]} {[fmt_t(x) for x in st[a]['rec']]}" for a in (1, 3)))
        stall = {a: np.median(data[a]["mu"][:, -2000:]) for a in (1, 3)}
        c("new5", "recovery: well-dosed arms stall at the ceiling (median late mu < c - 0.3)",
          all(v < self.C_NEW - 0.3 for v in stall.values()),
          ", ".join(f"{self.NAMES[a]} mu {v:.2f}" for a, v in stall.items()) + f" (c = {self.C_NEW})")
        smax = {a: np.max(data[a]["sigma"]) for a in self.ARMS}
        c("orig", "sigma <= 3 throughout, every arm", max(smax.values()) <= 3,
          ", ".join(f"{self.NAMES[a]} {v:.2f}" for a, v in smax.items()))
        s_late, b_late = np.median(d4["sigma"][:, -4000:]), np.median(d4["boost"][:, -4000:])
        c("new5", "I4: after re-acquisition the boost arm returns to the regulated band (predicted: regulation)",
          bool(np.isfinite(r[4]) and abs(s_late / cf["sigma_reg_bump"] - 1) <= 0.25
               and abs(b_late / cf["boost_reg"] - 1) <= 0.25),
          f"last 4000 iters: sigma {s_late:.4f} vs {cf['sigma_reg_bump']:.4f}; boost {b_late:.3f} vs {cf['boost_reg']:.3f}")
        return c

    def best_mu(self, sigma):
        mus = np.linspace(-1, 3.5, 4501)
        f = lambda x: self.reward(x, *self.env(self.ITERS - 1))
        return mus[np.argmax(gh_expect(f, mus, np.full_like(mus, sigma)))]

    def bonus_ratio(self, d):
        """c_H / kappa(sigma) at the measured (median) state: 1 = balance, p/sqrt(2) = the no-fixed-point law."""
        mu, sg, cH = (np.median(d[k], axis=0) for k in ("mu", "sigma", "cH"))
        out = np.empty(self.ITERS)
        for lo in range(0, self.ITERS, 1000):
            idx = np.arange(lo, min(lo + 1000, self.ITERS))
            f = lambda x: self.reward(x, *(v[idx, None] for v in (self.h1, self.h2, self.k, self.c1)))
            x_hi = mu[idx, None] + sg[idx, None] * 1.001 * _GH_X
            x_lo = mu[idx, None] + sg[idx, None] / 1.001 * _GH_X
            kappa = -((f(x_hi) - f(x_lo)) @ _GH_W) / (2 * np.log(1.001))
            out[idx] = cH[idx] / kappa
        return out

    # ---------------------------------------------------------------- figure
    def figure(self, data, st, epi, epi_data, epi_st, plt):
        t = np.arange(self.ITERS)
        med = lambda x: np.nanmedian(x, axis=0)
        ncol = len(self.ARMS)
        nrow = 8
        fig = plt.figure(figsize=(6 * ncol, 33))
        gs = fig.add_gridspec(nrow + 2, ncol, height_ratios=[1] * (nrow - 1) + [0.45, 0.12, 1.0])
        axes = np.empty((nrow, ncol), dtype=object)
        for i in range(nrow):
            for j in range(ncol):
                axes[i, j] = fig.add_subplot(gs[i, j], sharex=axes[0, 0] if (i or j) else None)
        titles = ["1. policy width sigma\n(dashed: balance\nkappa(sigma*) = c_H)",
                  "2. policy centre mu\n(dashed: bump centre;\ndotted: park point 0)",
                  "3. realised pressure\nc_H / std(A)",
                  "4. bonus / contraction\nc_H / kappa(sigma)\n(1 = balance)",
                  "5. trigger logic: std(A),\nslow EMA, settled peak,\n0.3 x peak",
                  "6. boost and\nstarvation flag",
                  "7. mean reward\n(dashed: ideal;\ndotted: at held width)",
                  "8. phase means and\nrecovery"]
        sub = np.arange(0, self.ITERS, 40)
        cf = self.closed_forms(data)
        inset_s = np.logspace(np.log10(0.05), np.log10(3), 60)
        inset_mu = np.array([self.best_mu(s) for s in inset_s])
        ideal = self.optimum(t)[1]
        target = self.RECOVERY_FRAC * ideal[-1]
        for j, arm in enumerate(self.ARMS):
            d, col = data[arm], self.COLORS[arm]
            axes[0, j].set_title(self.NAMES[arm], fontsize=15, fontweight="bold", color=col, pad=16)

            ax = axes[0, j]
            ax.plot(t, med(d["sigma"]), color=col, lw=1.6, label="sigma (median)")
            cH_s = rolling_median(med(d["cH"]), 25)
            pre = sub < self.T_EVENT
            ax.plot(sub[pre], self.sigma_star(cH_s[sub][pre], sub[pre]), color="k", lw=1.0, ls="--",
                    label="balance on the bump")
            ax.axhline(np.exp(LS_MIN), color="0.5", lw=0.7, ls="-.", label="log-sigma clamp")
            ax.axhline(self.CEILING, color="purple", lw=1.0, ls="--",
                       label=f"acquisition ceiling (hw/2k)^(1/3) = {self.CEILING:.2f}")
            if arm == 2:
                ax.hlines(cf["sigma2"], 0, self.T_EVENT, color="#b2182b", lw=1.2, ls=":",
                          label=f"noise-rescued fixed point {cf['sigma2']:.3f}")
            ax.set_yscale("log")
            ax.set_ylim(1e-2, 5)
            ax.legend(fontsize=8, loc="upper left")

            ax = axes[1, j]
            for s in range(d["mu"].shape[0]):
                ax.plot(t, d["mu"][s], color=col, lw=0.4, alpha=0.3)
            ax.plot(t, med(d["mu"]), color=col, lw=1.6, label="mu (median; faint = seeds)")
            ax.plot(t, self.c1, color="k", lw=0.9, ls="--", label="bump centre")
            ax.axhline(0.0, color="0.5", lw=0.8, ls=":")
            ax.set_ylim(-1.5, 3.5)
            ax.legend(fontsize=8, loc="upper left")
            ins = ax.inset_axes([0.62, 0.08, 0.35, 0.42])
            ins.plot(inset_s, inset_mu, color="k", lw=1.0)
            ins.axhline(self.C_NEW, color="k", lw=0.6, ls="--")
            ins.axvline(self.CEILING, color="purple", lw=0.8, ls="--")
            s_late = np.median(d["sigma"][:, -2000:])
            ins.plot([s_late], [np.median(d["mu"][:, -2000:])], "o", color=col, ms=5)
            ins.set_xscale("log")
            ins.set_xlim(0.05, 3)
            ins.set_ylim(-0.2, self.C_NEW + 0.4)
            ins.set_title("best mu vs sigma (after event); dot = this arm, late", fontsize=6.5)
            ins.tick_params(labelsize=6)

            ax = axes[2, j]
            ax.plot(t, med(d["pressure"]), color=col, lw=0.5, alpha=0.3)
            ax.plot(t, rolling_mean(med(d["pressure"]), 50), color=col, lw=1.6, label="realised (50-iter mean)")
            ax.axhline(self.P, color="k", lw=1.0, ls="--", label=f"searched p = {self.P:g}")
            ax.axhline(np.sqrt(2), color="0.4", lw=0.8, ls=":", label="sqrt(2): the deterministic-limit pin")
            if arm != 2:
                ax.hlines(cf["press"], 0, self.T_EVENT, color="k", lw=1.0, ls="-.",
                          label=f"noise-corrected settled pressure {cf['press']:.2f}")
                ax.hlines(cf["cH"] / self.NOISE, self.T_EVENT, self.ITERS, color="0.3", lw=0.8, ls="--",
                          label=f"c_H / eta = {cf['cH'] / self.NOISE:.1f} (signal term dead)")
            if arm >= 3:
                fb = d["floor"].mean(axis=0)
                ax.fill_between(t, 1e-3, np.where(fb > 0.5, 1e4, 1e-3), color="purple", alpha=0.12, step="mid",
                                label="warmup floor binding")
            ax.set_yscale("log")
            ax.set_ylim(0.1, 1e3)
            ax.legend(fontsize=8, loc="upper left")

            ax = axes[3, j]
            ax.plot(t, self.bonus_ratio(d), color=col, lw=1.4, label="c_H / kappa (median state)")
            if arm == 2:
                d_inst = dict(d)
                d_inst["cH"] = self.P * d["stdA"]
                ax.plot(t, self.bonus_ratio(d_inst), color="0.3", lw=0.9,
                        label="p * std(A)_now / kappa (no EMA lag)")
            ax.axhline(1.0, color="k", lw=1.0, ls="--", label="1 = balance")
            ax.axhline(self.P / np.sqrt(2), color="#b2182b", lw=1.0, ls=":", label="p/sqrt(2) = 0.566 (no fixed point)")
            ax.set_yscale("log")
            ax.set_ylim(0.1, 1e3)
            ax.legend(fontsize=8, loc="upper left")

            ax = axes[4, j]
            ax.plot(t, med(d["stdA"]), color=col, lw=0.5, alpha=0.35, label="std(A) per batch")
            ax.plot(t, med(d["s_slow"]), color=col, lw=1.6, label=f"slow EMA (memory {self.M:g})")
            ax.plot(t, med(d["peak"]), color="k", lw=1.0, label=f"settled peak (t >= {self.SETTLE})")
            ax.plot(t, STARVE_FRAC * med(d["peak"]), color="k", lw=1.0, ls="--", label="trigger: 0.3 x peak")
            ax.axvline(self.SETTLE, color="0.6", lw=0.8, ls="-.")
            ax.axhline(self.NOISE, color="purple", lw=0.8, ls=":", label=f"noise floor eta = {self.NOISE}")
            ax.set_yscale("log")
            ax.legend(fontsize=8, loc="lower left")

            ax = axes[5, j]
            if arm == 4:
                ax.plot(t, med(d["boost"]), color=col, lw=1.6, label="boost (median)")
                ax.fill_between(t, 0, 8 * d["starving"].mean(axis=0), color="orange", alpha=0.3, step="mid",
                                label="starving (fraction of seeds x 8)")
                ax.axhline(1.0, color="k", lw=0.8, ls="--", label="off-state: 1")
                ax.axhline(1.3, color="0.5", lw=0.6, ls="-.", label="1.3")
                ax.set_ylim(0, 8.5)
            else:
                ax.fill_between(t, 0, d["starving"].mean(axis=0), color="orange", alpha=0.3, step="mid",
                                label="what the starvation signal would say (unused)")
                ax.set_ylim(0, 1.05)
                ax.text(0.5, 0.5, "no boost in this arm", ha="center", va="center", transform=ax.transAxes,
                        fontsize=11, color="0.4")
            ax.legend(fontsize=8, loc="upper left")

            ax = axes[6, j]
            for s in range(d["perf"].shape[0]):
                ax.plot(t, rolling_mean(d["perf"][s], 50), color=col, lw=0.4, alpha=0.3)
            ax.plot(t, rolling_mean(med(d["perf"]), 50), color=col, lw=1.6, label="median reward (50-iter mean)")
            ax.plot(t, ideal, color="k", lw=1.0, ls="--", label="ideal")
            ax.plot(t, self.held_width_reward(med(d["sigma"])), color="k", lw=0.9, ls=":",
                    label="predicted on the bump at held width")
            ax.axhline(target, color="0.5", lw=0.7, ls="-.", xmin=self.T_EVENT / self.ITERS,
                       label=f"recovery threshold {target:.3f}")
            ax.set_ylim(-0.3, 1.15)
            ax.legend(fontsize=8, loc="upper left")

            ax = axes[7, j]
            ph = self.phases()
            txt = "reward  " + "  ".join(f"{p}:{np.mean(d['perf'][:, lo:hi]):.3f}" for p, (lo, hi) in ph.items())
            txt += "\nsigma   " + "  ".join(f"{p}:{np.median(d['sigma'][:, lo:hi]):.3f}" for p, (lo, hi) in ph.items())
            txt += f"\nrecovery: {fmt_t(st[arm]['rec_med'])} [{', '.join(fmt_t(x) for x in st[arm]['rec'])}]"
            if arm == 4:
                txt += f"\nboost < 1.3 after recovery: {', '.join(fmt_t(x) for x in st[4]['decay'])}"
            ax.text(0.02, 0.5, txt, transform=ax.transAxes, fontsize=9, va="center", family="monospace")
            ax.set_yticks([])
            ax.set_xlabel("iteration")

            for i in range(nrow):
                for x in self.BOUNDS:
                    axes[i, j].axvline(x, color="0.4", lw=0.8, ls=":")
                if j == 0:
                    axes[i, j].set_ylabel(titles[i], fontsize=10.5)
            for x, lab in ((3000, "A: settle / spiral"), (13000, f"D: bump moved to +{self.C_NEW:g}")):
                axes[0, j].text(x, 1.01, lab, transform=axes[0, j].get_xaxis_transform(), ha="center", fontsize=8,
                                color="0.35", va="bottom")

        # epilogue (amendment 2, kept as delivered): the scope boundary
        de, te = epi_data[4], np.arange(epi.ITERS)
        ax = fig.add_subplot(gs[nrow + 1, 0])
        ax.plot(te, med(de["stdA"]), color="0.5", lw=0.5, alpha=0.5, label="std(A)")
        ax.plot(te, med(de["s_slow"]), color=self.COLORS[4], lw=1.6, label="slow EMA")
        ax.plot(te, med(de["peak"]), color="k", lw=1.0, label="settled peak")
        ax.plot(te, STARVE_FRAC * med(de["peak"]), color="k", lw=1.0, ls="--", label="0.3 x peak")
        ax.set_yscale("log")
        ax.set_title("Epilogue: swap world - trigger inputs", fontsize=11)
        ax.legend(fontsize=8)
        ax = fig.add_subplot(gs[nrow + 1, 1])
        ax.plot(te, med(de["boost"]), color=self.COLORS[4], lw=1.6, label="boost")
        ax.fill_between(te, 0, 8 * de["starving"].mean(axis=0), color="orange", alpha=0.3, step="mid",
                        label="starving (x 8)")
        ax.set_ylim(0, 8.5)
        ax.set_title("boost and starvation flag (silent)", fontsize=11)
        ax.legend(fontsize=8)
        ax = fig.add_subplot(gs[nrow + 1, 2])
        ax.plot(te, rolling_mean(med(de["perf"]), 50), color=self.COLORS[4], lw=1.6, label="median reward")
        ax.plot(te, epi.optimum(te)[1], color="k", lw=1.0, ls="--", label="ideal")
        ax.set_ylim(0.3, 1.4)
        ax.set_title("reward: stuck on the worse bump", fontsize=11)
        ax.legend(fontsize=8)
        for a_ in fig.axes[-3:]:
            a_.axvline(epi.T_EVENT, color="0.4", lw=0.8, ls=":")
            a_.axvline(epi.SETTLE, color="0.6", lw=0.8, ls="-.")
            a_.set_xlabel("iteration")
        ax = fig.add_subplot(gs[nrow + 1, 3])
        ax.axis("off")
        ax.text(0.0, 0.5, "A policy trapped at a locally optimal behaviour is healthy in every width "
                          "observable; the starvation trigger correctly stays silent — this failure family is "
                          "not width-collapse and is not entropy-treatable (it belongs to the optimism/reset "
                          "mechanisms of the parent study).", wrap=True, fontsize=11, va="center", style="italic")
        fig.suptitle(f"Figure 2 - the boost on the under-dosed arm (bump relocates -0.7 -> +{self.C_NEW:g}; eta = "
                     f"{self.NOISE}; p = {self.P:g}; M = {self.M:g}, t_ref = {self.SETTLE}, B_max = {self.B_MAX:g}; "
                     "median of 5 seeds; dashed = theory)", fontsize=16)
        fig.tight_layout(rect=(0, 0, 1, 0.985))
        fig.savefig(self.FIG_PATH, dpi=100)
        return self.FIG_PATH


# ====================================================================== driver
def npz_path(fig, arm, seed):
    return os.path.join(fig.RUN_DIR, f"arm{arm}_seed{seed}.npz")


def run_and_save(fig, arm, seed, force):
    path = npz_path(fig, arm, seed)
    if os.path.exists(path) and not force:
        return
    os.makedirs(fig.RUN_DIR, exist_ok=True)
    t0 = time.time()
    np.savez_compressed(path, **fig.run(arm, seed))
    print(f"{fig.NAMES[arm]} seed {seed}: {time.time() - t0:.1f}s -> {os.path.relpath(path, HERE)}", flush=True)


def load_all(fig):
    out = {}
    for arm in fig.ARMS:
        runs = [np.load(npz_path(fig, arm, s)) for s in SEEDS]
        out[arm] = {k: np.stack([r[k] for r in runs]) for k in fig.KEYS}
    return out


def plot_and_report(fig, which):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load_all(fig)
    if which == 1:
        table, checks = fig.summary(data), fig.acceptance(data)
        path = fig.figure(data, plt)
    else:
        epi = Fig2("swap")
        epi_data = load_all(epi)
        st, epi_st = fig.stats(data), epi.stats(epi_data)
        table = fig.summary(data, st) + "\n\nEPILOGUE (swap world, boost arm only, 8000 iters, swap at 4000)\n" + \
            epi.summary(epi_data, epi_st)
        checks = fig.acceptance(data, st)
        checks.lines += epi.acceptance(epi_data, epi_st).lines
        checks.n_fail = sum(l.startswith("[FAIL]") for l in checks.lines)
        path = fig.figure(data, st, epi, epi_data, epi_st, plt)
    text = (f"FIGURE {which} summary\n{table}\n\nAcceptance criteria "
            "(tags: orig = brief as written, adapted = brief criterion re-anchored to the amended schedule, "
            "revised = changed by amendment 1, new = introduced by amendment 1, new2 = introduced by amendment 2, new3 = introduced by amendment 3, new4 = introduced by amendment 4, new5 = introduced by amendment 5)\n"
            + "\n".join(checks.lines) + f"\n{checks.n_fail} failing.\n")
    print(text)
    with open(fig.TABLE_PATH, "w") as f:
        f.write(text)
    print(f"figure -> {os.path.relpath(path, HERE)}; table -> {os.path.relpath(fig.TABLE_PATH, HERE)}")


if __name__ == "__main__":
    which = int(os.environ.get("FIG", "1"))
    fig = Fig1 if which == 1 else Fig2("swap" if os.environ.get("EPI") == "1" else "relocate")
    if os.environ.get("PLOT") == "1":
        plot_and_report(fig, which)
        sys.exit(0)
    arms = [int(os.environ["ARM"])] if "ARM" in os.environ else list(fig.ARMS)
    seeds = [int(os.environ["SEED"])] if "SEED" in os.environ else list(SEEDS)
    targets = [fig] if (which == 1 or "ARM" in os.environ or "SEED" in os.environ) else [fig, Fig2("swap")]
    for f in targets:
        for arm in (arms if f is fig else list(f.ARMS)):
            for seed in seeds:
                run_and_save(f, arm, seed, os.environ.get("FORCE") == "1")
    if "ARM" not in os.environ and "SEED" not in os.environ and os.environ.get("EPI") != "1":
        plot_and_report(fig, which)
