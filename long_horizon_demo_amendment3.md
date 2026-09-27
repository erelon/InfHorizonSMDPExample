# Amendment 3 to the long-horizon demo — figure 2, second redesign

Both figure-2 failures trace to my design errors, which your theory checks
caught: (1) the stiffening event raised the floor curvature where the bump's
own curvature (h/w^2 = 11.1) dominates — a 9% perturbation sold as 6x; (2) the
0.25x floor was an arbitrary weakening — the parent system's shipped rule
floors the tracked scale at the FULL warmup value, c_H = p * max(EMA_slow,
s_warmup). Your p/sqrt(2) derivation is accepted and PROMOTED: it is the
cleanest statement of the pro-cyclicality trap in this project's history and
becomes a prediction overlay on the figure (credited in the README as derived
during this exercise). Figure 1 is closed (the LR-halving side-check
supporting the noise-floor annotation is accepted as confirmation).

## The named result your report contains

**Live-scale calibration has no fixed point on a noise-free quadratic peak.**
With std(A) = C*sigma^2/sqrt(2) and contraction kappa = C*sigma^2, the
bonus-to-contraction ratio is p/sqrt(2) at EVERY width; for p < sqrt(2),
contraction wins at all scales and sigma falls to the clamp regardless of the
landscape. Real systems escape only because std(A) carries exogenous variance
that the policy's own width does not generate — which is exactly why the
shipped rule freezes/floors the warmup scale. The no-floor arm is therefore
not expected to match the others in phase A; it is the exhibit of this law,
with the constant p/sqrt(2) = 0.57 drawn on its bonus/contraction panel.

## Figure 2, third design: the crisis is RELOCATION-INTO-REACH-FAILURE

Environment (k reduced globally so the quadratic anchor never guts a bump):
- r(a) = h1(t)*exp(-(a - c1(t))^2/(2*0.3^2)) - 0.02*a^2, single bump.
- Phase A [0, 3000): h1 = 1.0, c1 = -0.7. Settle and lock.
- Phase B [3000, 6000): h1 ramps 1.0 -> 2.5, c1 = -0.7. std(A) rises ~2.5x:
  the DELIVERY exhibit inside this world — the frozen arm's realised pressure
  decays ~2.5x; the tracked arms hold it.
- Event at t = 6000: the bump VANISHES and reappears at c1 = +2.8 with
  h1 = 1.0 (peak reward there: 1 - 0.02*2.8^2 = 0.843). The policy is left on
  the near-flat quadratic; it will drift and park at the flat maximum a = 0 —
  this parking is anticipated and harmless (the k-pull you flagged), and all
  reach distances below are measured from the park point.
- Phase D [6000, 16000): recovery and boost decay.

Arms (5 seeds, medians; centring ON; lambda = 0; current-parameter full-batch
step as before; p = 0.8):
1. frozen dose: c_H = p * std(A at batch 1).
2. tracked, NO floor: c_H = p * EMA_400(std(A)). The p/sqrt(2) exhibit; it
   spirals to the clamp in phase A and never recovers after the event.
3. tracked + floor AT THE WARMUP SCALE: c_H = p * max(EMA_400, batch-1 std).
   In this world std only falls below the warmup scale after lock-on and after
   the event, so this arm equals arm 1 except during phase B, where the EMA
   exceeds the floor and delivery is held. This is the shipped rule.
4. arm 3 + boost (hold guard, settled reference from t = 2000, f = 0.3,
   cap 8, e-fold 600).

Why the event separates the arms (pre-registered mechanics):
- At the park point the local curvature is 2k = 0.04, so the flat-land width
  force is ~c_H against kappa = 0.04*sigma^2; escape requires growing sigma
  from ~0.1 to the reach width ~1.1 (2.8 / ~2.5), and the growth rate of
  log sigma is proportional to c_H. Expected escape ordering, by 1/c_H:
  arm 2 never (c_H ~ 0 on the flat), arms 1 = 3 slow (c_H ~ 0.23), arm 4 fast
  (boosted c_H up to ~1.8). Recovery-time ratio arm1/arm4 >= 3 is the
  criterion; report the measured ratio either way.
- The trigger: after the event std(A) collapses (~0.001 at the park against a
  settled reference ~0.25 from late A / scaled B — use the reference as
  specified, max tracked from t = 2000), so starving fires within 300 iters on
  arm 4 and the boost ratchets under the hold guard. After re-acquisition,
  std recovers above 0.3x the reference and the boost decays.

Pre-registered expectations (replacing amendment 2's):
- Phase A: arms 1, 3, 4 identical within noise; trigger silent on arm 4 after
  t = 2000; arm 2 falls to the clamp with its bonus/contraction panel pinned
  at p/sqrt(2) (the overlay).
- Phase B: arms 1 vs 3 separate in REALISED PRESSURE (frozen decays toward
  ~0.32 of searched; tracked holds within 20%); widths follow their balance
  predictions (recompute with h1(t)).
- Event: reward crashes to the flat level in every arm (nobody is locally
  content — the crisis is unambiguous). Arm 4: starving within 300 iters,
  boost >= 3 during escape, sigma <= 3 throughout (if exceeded, report — the
  approach-phase contraction should cap it, and whether it does is data).
- Recovery: trailing-200 window entirely post-event, threshold 0.8x the
  POST-EVENT ideal (0.8 * 0.843 = 0.674). Arm 4 recovers; arm 4 at least 3x
  faster than arms 1/3 (or they never do); arm 2 never; boost decays below
  1.3 within 3000 iters of arm 4's recovery.
- Keep the epilogue panel from amendment 2 (swap world, silent trigger, the
  taxonomy caption) exactly as delivered — it passed and it stays.

## Metric and bookkeeping

- The recovery metric above replaces all previous versions; your
  window-entirely-post-event reading is adopted as the definition.
- The README's boxed notes now number three: the batch-shape/accidental-
  exploration finding (report 1), the trigger's scope boundary (report 2 /
  epilogue), and the p/sqrt(2) no-fixed-point law (report 3). Together with
  the figures they are the exercise's actual product: the toy has now
  reproduced, from its own failures, the three design lessons the parent
  project paid 40M-step experiments to learn — one machinery per failure
  mode, detectors have scope, and never calibrate a scale to a signal the
  policy's own state generates.
- Process unchanged: report before tuning; state which criteria this
  amendment revised; chunked execution; plain-language README, five-sentence
  summary first.
