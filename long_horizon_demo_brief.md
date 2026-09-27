# Long-horizon demo brief — every component of the fix-set, working, on one sine-tracking task

You are building a self-contained demonstration in a fresh repo. No other code
or context is needed; this file is the full specification. Read it end to end
before writing anything — the "traps we already hit" section will save you
hours.

## 1. What this demonstrates, in plain language

We train policies with policy gradients under two commitments most codebases
avoid: **no discounting (gamma = 1)** and **semi-Markov structure** (not used in
this demo — keep tau = 1; it has its own figure elsewhere). Under gamma = 1 two
failure modes appear:

1. **The offset.** The algorithm subtracts an estimate of the average reward
   ("the gain") from every reward. That estimate always lags a moving target,
   the lag enters every advantage as a uniform offset, the advantage estimator
   amplifies it by 1/(1-lambda), and at gamma = 1 the critic CANNOT absorb it
   (it just integrates it). Under gradient clipping the offset dominates the
   update direction and destroys the policy. Fix: **subtract the batch mean of
   the advantage** before the policy update ("centring") — exact, one line.
2. **Exploration starvation.** Exploiting a reward peak contracts the policy's
   width sigma (this is a theorem, not an accident: the width force equals
   sigma^2 times the local curvature of the advantage). On a continuing task
   with no resets, nothing re-widens it, so the policy goes deterministic and
   can never adapt again. Fix: an entropy bonus with coefficient c_H — but the
   RIGHT c_H is environment- and time-dependent, which is the interesting part:
   - **Calibration:** c_H = p * std(advantage), so p ("pressure") is
     dimensionless. But if std is measured once and frozen, the delivered
     pressure drifts as std changes over training (this silently broke our real
     40M study).
   - **Tracking:** use a slow running estimate of std instead of a frozen one —
     delivers the searched pressure forever.
   - **Boost:** when the task changes and the policy is already contracted,
     even the correct constant pressure is too slow. A bounded multiplicative
     boost, raised only while a RELATIVE starvation signal fires (the advantage
     scale falling below a fraction of its own running peak) and disabled while
     entropy is already rising, supplies the temporary extra width, then decays.

All of this was validated in the real system (150 paired 40M MuJoCo runs). Your
job: show every component doing its work on a toy so simple that every panel
can carry the THEORETICAL PREDICTION as an overlay. One long run, phases, four
arms, one big figure.

## 2. The environment (already tuned — do not redesign)

Continuing 1-D tracking task, batch of N i.i.d. actions per iteration:

- Policy: a ~ Normal(mu, sigma^2). Two learned scalars: mu, log_sigma
  (clip log_sigma to [-4, 2.5]).
- Critic: ONE scalar phi (this makes the offset theory exact, not approximate).
- Target: g(t) = g_c(t) + 0.8 * sin(2*pi*t / 3000)   [the sine of the title]
- Reward per sample: r = m(t) * (1 - min((a - g(t))^2, 4) + ramp(t))
  - the penalty is BOUNDED at 4 (a wrecked policy sits on a flat floor at -3
    instead of feeding chaotic feedback — keeps plots legible)
  - ramp(t) = 0.003 * t^2 / 1600 for t < 800, else 0.003 * (t - 400): a reward
    level that rises forever = a gain that never stops improving = a PERMANENT
    estimator lag eps ~= 0.15 (this fuels failure mode 1; the soft onset lets
    the discounted/centred arms survive the start)
  - m(t) and g_c(t) implement the phases (Section 3)
- Gain estimate: eta = num/den, two EMAs with gain ALPHA = 0.02, SEEDED from
  batch 1 (num = mean r, den = 1). Seeding isolates the sustained lag.
- Residual: delta = r - eta + (gamma - 1) * phi     [gamma = 1 in all arms here]
- Advantage: GAE over the batch treated as a sequence,
  A_k = sum_{j>=k} (gamma*lambda)^(j-k) delta_j, LAMBDA = 0.95.
- Critic pass: phi <- mean(A + phi)  (full regression, BETA_V = 1.0 — the
  scalar critic is literally an integrator of mean(A); that is the point).
- Centring (arms 2-4): A_used = A - mean(A). Arm 1: A_used = A.
- Policy update: EPOCHS = 6 passes over minibatches of M = 32 from the batch of
  N = 256; per minibatch, gradient
    g_mu = mean(A_used * z / sigma),  g_ls = mean(A_used * (z^2 - 1)) + c_H
  with z = (a - mu)/sigma computed ONCE per iteration from the pre-update
  (mu, sigma) and reused across epochs (this staleness IS the damage channel of
  failure mode 1 — do not "fix" it); clip the 2-vector to unit norm; step with
  LR = 0.025.
- Log every iteration: b = mean(A) pre-centring, its prediction
  (eps + (gamma-1)*phi)/(1 - gamma*lambda), phi, std(A), sigma, realised
  pressure c_H/std(A), the boost value, the starvation flag, tracking
  performance perf = 1 - mean(min((a-g)^2, 4)) (ramp-free, scale-free).

## 3. The long horizon: one run, four phases (total ITERS = 16000)

| phase | iterations | what changes | which mechanism it stresses |
|---|---|---|---|
| A: stable | 0 - 4000 | m = 1, g_c = 0 | baseline health; frozen p is fine here |
| B: scale drift | 4000 - 9000 | m(t) ramps linearly 1 -> 5 | std(A) grows ~5x, so a FROZEN c_H's realised pressure decays ~5x (the delivery failure); the tracked floor holds it |
| C: abrupt change | at t = 9000 | g_c jumps 0 -> 3.0 (sine continues around the new centre); m stays 5 | a contracted policy is blind to the new target (the old one sits in the flat penalty region); constant pressure recovers slowly or not at all; the BOOST must fire |
| D: recovery | 9000 - 16000 | nothing further | the boost must DECAY back to the floor after recovery (a controller, not a latch) |

Phase boundaries are drawn as vertical lines on every panel.

## 4. The four arms (5 seeds each, plot elementwise medians)

| arm | centring | entropy mechanism | expected story |
|---|---|---|---|
| 1: uncentred | OFF | tiny fixed floor, p = 0.05 frozen at batch-1 scale | destroyed in phase A by the offset alone (b tracks its prediction, phi integrates, direction cosine dies, perf hits the -3 floor). The reminder of failure mode 1. |
| 2: frozen | ON | calibrated p = 0.2, c_H frozen at batch-1 std | healthy in A; realised pressure decays ~5x through B; sigma over-contracts; at C it is stuck or recovers very slowly. The delivery failure. |
| 3: tracked | ON | p = 0.2, c_H = p * slow-EMA(std(A)) (memory ~800 iters) | realised pressure held at 0.2 through B; at C recovers faster than arm 2 but still slowly (constant pressure vs a contracted policy facing a jump). |
| 4: tracked + boost | ON | arm 3 + boost: log_boost += +beta while (starving and entropy not rising) else -beta; clip to [0, log 8]; starving := s_slow < 0.3 * running_peak(s_slow); s_slow = the same slow EMA; entropy-rising guard = positive slope of log sigma over the last 200 iters; beta such that the boost e-folds in ~600 iters | identical to arm 3 until C; at C the starvation trigger fires, boost climbs, sigma spikes, the policy re-acquires the target, boost decays back to ~1 in D. The full shipped mechanism. |

Notes: the searched p = 0.2 was derived for this landscape from the balance
condition kappa(sigma*) = c_H (quadratic: kappa = 2 sigma^2 per unit reward
scale) targeting sigma* ~ 0.25; keep it. Arm 1 keeps its tiny floor so the
width channel is identical to what arm 1 had in our earlier figure and its
death is attributable to the offset alone. Init: mu_0 = 0 (= g(0), start AT the
optimum), sigma_0 = 0.35, phi_0 = 0.

## 5. The figure (one file, ~7 rows x 4 columns, shared x)

Rows, each with an "ideal" reference line and, where we have one, a PREDICTION
overlay (dashed):
1. offset b, with prediction (eps + (gamma-1)phi)/(1-gamma*lambda); centred
   arms additionally show the bold after-centring line at 0 ("what the policy
   sees: 0 exactly").
2. critic phi with its predicted integrator cumsum(mean-b prediction) — label:
   "drift cancels in the residual: harmless".
3. REALISED PRESSURE c_H / std(A): the searched value 0.2 as the ideal line.
   This is the demo's money row: arm 2's curve decays through B, arms 3-4 sit
   on the line, arm 4 spikes above it during C and returns.
4. sigma (log scale), with the balance-predicted equilibrium per phase drawn
   as horizontal dashed segments (recompute sigma* per phase from the
   scale/curvature; on this quadratic: sigma* = sqrt(c_H_realised / (2 m(t)))).
5. boost value and the starvation flag (arm 4 only; other columns empty with a
   note) — must show: silent in A/B, fires at C, regulates, decays in D.
6. tracking performance perf, ideal line at 1.0; phase lines; the C-recovery
   contrast between arms 2/3/4 is the outcome claim.
7. direction quality cos(update, true gradient) where the true gradient is
   [-2 m (mu - g), -2 m sigma^2] (bounded-penalty region only; plot raw).

Also print a summary table: per arm — perf mean in each phase, recovery time
after C (iterations until perf returns to its pre-jump mean), realised-pressure
mean per phase, boost peak and decay time.

## 6. Acceptance criteria (pre-registered; check before delivering)

- Arm 1 dead by t ~ 2000 (perf < -1), b on its prediction, phi unbounded.
- Arm 2: realised pressure at C is <= 0.06 (>= 3x under-delivery); recovery
  time after C at least 3x arm 4's, or no recovery by t = 16000.
- Arm 3: realised pressure within 20% of 0.2 throughout A-B; recovery slower
  than arm 4 by at least 2x.
- Arm 4: recovery to pre-jump perf within ~1500 iters of C; boost > 2 during
  recovery and back below 1.3 by t = 14000; sigma never exceeds 3 (the guard);
  phases A-B identical to arm 3 within noise (the trigger must NOT fire before
  C — if it does, your s_slow/peak memory is too short).
- Predictions overlay their measurements: b within the noise band; sigma
  plateaus within ~25% of the balance prediction; phi on its integrator line.
If a criterion fails, do not tune silently: report what you saw first.

## 7. Traps we already hit (each cost us a debugging cycle — respect them)

1. **Pro-cyclicality:** never calibrate a scale AFTER a collapse can happen.
   Batch-1 freeze for arm 2's frozen scale; the slow EMA for arms 3-4 must be
   seeded at batch 1 too.
2. **Units/clocks:** every rate (EMA memory, boost e-fold, guard window) is
   given above in ITERATIONS because this env has one batch per iteration and
   tau = 1. If you ever change batch size or add sojourn times, convert all of
   them to environment steps first. This bug has bitten this project seven
   times.
3. **Soft onset:** the ramp's quadratic onset (first 800 iters) is what lets
   the centred arms survive the start. Removing it kills them in the
   transient and invalidates the comparison.
4. **Bounded penalty:** without min(.,4) the destroyed arm enters a chaotic
   eta-feedback regime and the plots become unreadable.
5. **Stale score:** z is computed once per iteration on purpose (epoch reuse is
   the damage channel). Recomputing it per minibatch neutralises failure mode 1
   and breaks arm 1's story.
6. **Trigger false-fire is not automatically a bug:** if the trigger fires
   somewhere unexpected but outcomes are unharmed, report it as measured
   tolerance, exactly like that; do not widen f to hide it.

## 8. Operational constraints (how to actually run this)

- Pure numpy + matplotlib, single script, deterministic seeds 0-4.
- Estimated cost: 4 arms x 5 seeds x 16000 iters x 48 minibatch steps —
  roughly 10-25 minutes total. DO NOT run it as one monolithic foreground
  command if your execution environment kills long calls: structure the script
  so each (arm, seed) run saves its logged arrays to an .npz, guard the figure
  code under `if __name__ == "__main__"`, select work with an env var
  (e.g. ARM=2 SEED=3 python demo.py, then FIGS=1 python demo.py to plot from
  the .npz files). Background jobs do not survive between tool calls in some
  environments — chunk foreground instead.
- Deliverables: `long_horizon_demo.py`, the figure PNG, the printed summary
  table, and a plain-language README explaining every row of the figure to a
  reader who knows no RL jargon (state what each curve should do and why; open
  with a five-sentence plain summary; no internal arm names in the README —
  call them "no fixes", "frozen dose", "tracked dose", "tracked + boost").

## 9. What NOT to do

- Do not add mechanisms beyond the four arms (no z-scoring, no sigma targets,
  no meta-gradients — each was tested in the real system and failed or was
  out-scoped).
- Do not change the searched p mid-run in any arm; the point of arm 4 is that
  adaptation comes from the boost, not from retuning.
- Do not compare arms across different phase schedules or seeds; all arms see
  identical g(t), m(t), ramp(t), and share seeds (common random numbers).
- Do not smooth away the boost spike or the pressure decay in plots; they ARE
  the result.
