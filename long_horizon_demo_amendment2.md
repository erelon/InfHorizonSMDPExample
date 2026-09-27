# Amendment 2 to the long-horizon demo — figure 2 redesign after your report

Your three causes are accepted as structural, your recovery-window fix is
accepted as a correct reading of the metric, and your p = 0.4 diagnostic was
the right thing to run and the right thing not to adopt. Figure 1 is ACCEPTED
as delivered (12/13; disposition of the one failure below). Do not run the
hold-only diagnostic — it is superseded: this amendment adopts hold as the
correct semantics AND removes the reason it would have misfired.

## The finding your report contains, promoted to a result

The post-swap state in the two-bump world is a policy exploiting a locally
optimal behaviour — locally healthy in every width observable. That is not
width collapse; it is entrapment at a worse optimum, a different failure
family that the parent project proved is NOT entropy-treatable (it yields to
optimism/reset mechanisms instead). The relative starvation trigger is a
WIDTH-COLLAPSE detector; it fired at lock-on and stayed silent about the trap
because both healthy exploitation and entrapment look identical to a
variance-relative signal. Your figure 2 therefore did not fail its claim — it
located the trigger's scope boundary, which is a result. It stays in the
figure as an epilogue (below).

## Spec corrections (mine, not yours)

1. **Guard semantics = HOLD.** While `starving and log-sigma rising`: hold the
   boost (no growth, no decay). Decay (-beta) only when `not starving`. The
   amendment-1 "else decay" wording was wrong; hold is the shipped reading in
   the parent system. Your cause 3 (symmetric coin-flip, zero drift) is
   thereby removed: against a noisy slope the boost ratchets on the not-rising
   iterations and holds otherwise.
2. **Settled reference.** The running peak of s_slow starts accumulating only
   after a declared settling window: t >= 2000. Before that, starving is
   forced false and the boost is pinned at 1. Justification: a reference must
   be taken from the regime it is meant to police (the same principle as the
   batch-1 scale freeze — never calibrate across a transient). This removes
   your cause 1's lock-on half: after settling, the peak equals the healthy
   locked-on level, and only a genuine collapse below it can fire the trigger.
3. **The floor becomes an exhibit.** Your cause 2 (tracked scale is
   pro-cyclical after lock-on; the floor binds) is correct and is exactly why
   the parent system's floor exists. It gets its own arm below so the figure
   shows it instead of suffering it.

## Figure 2, redesigned: the crisis is REQUIREMENT MOTION, not relocation

Environment: the two-bump world of amendment 1, unchanged EXCEPT the
non-stationary event. The bumps never swap in the main phases. Instead the
curvature floor stiffens and relaxes:

- k(t) = 0.10 for t < 6000; ramps linearly 0.10 -> 0.60 over [6000, 7000];
  holds 0.60 over [7000, 11000]; relaxes linearly back to 0.10 over
  [11000, 12000]; 0.10 to the end (T = 16000).
- Stiffening raises the contraction kappa everywhere, crushing sigma below the
  working band under any fixed dose (the balance sigma* falls ~sqrt(6)) and
  collapsing std(A) far below the settled reference — a genuine width-collapse
  event, the type the controller was built and validated for.

Phases: A settle [0, 6000) | C stiffen [6000, 11000) | D relax + decay
[11000, 16000). Draw all boundaries on every panel.

Arms (5 seeds, medians; centring ON everywhere; lambda = 0, current-parameter
full-batch step as in amendment 1; p = 0.8):
1. **frozen dose**: c_H = p * std(A at batch 1).
2. **tracked, no floor**: c_H = p * slow-EMA(std(A)) (memory 400). Expected to
   be the WORST arm in C — the pro-cyclicality exhibit.
3. **tracked + floor**: as 2, floored at 0.25x the batch-1 scale.
4. **tracked + floor + boost**: as 3, plus the boost with HOLD guard, settled
   reference, f = 0.3, cap 8, e-fold 600 iters.

Rows: sigma (log) with per-phase balance predictions per arm; realised
pressure; std(A), its slow EMA, the settled peak, and the 0.3x threshold (one
panel that makes the trigger's logic visible); boost + starving flag;
reward with predicted-at-held-width overlays; a text panel of phase means and
recovery times (your corrected windowed metric, window entirely inside the
phase).

Pre-registered expectations:
- Phase A: all four arms identical within noise; trigger silent after t = 2000
  (if it fires in A, the settling window or the memory is wrong — report, do
  not widen f).
- Phase C: frozen and floor arms crushed to their new (low) balance widths and
  degraded reward; no-floor arm strictly worse than frozen; boost arm fires
  within 500 iters of the collapse crossing the threshold, ratchets under the
  hold guard, restores sigma to within 25% of the healthy-band prediction
  kappa(sigma*) = c_H_boosted, and recovers reward to >= 0.9x its phase-A mean
  at least 3x faster than any other arm (or while others never do).
- Phase D: boost decays below 1.3 within 3000 iters of the relax completing;
  all arms reconverge; the boost arm ends indistinguishable from arm 3.
- Sigma never exceeds 3 in any arm (the guard's purpose).

## Epilogue panel (small, after the main figure): the scope boundary

One extra short run of arm 4 only, in the ORIGINAL swap world (amendment 1's
h-swap at its midpoint), 8000 iters: show std(A), the settled peak, the silent
trigger, and the stuck reward. Caption it exactly as: "A policy trapped at a
locally optimal behaviour is healthy in every width observable; the starvation
trigger correctly stays silent — this failure family is not width-collapse and
is not entropy-treatable (it belongs to the optimism/reset mechanisms of the
parent study)." This is the taxonomy, exhibited; it is the honest disposition
of your negative result and must not be cut.

## Figure 1 disposition

Accepted as delivered. The one failing criterion (frozen-dose sigma 1.31-1.32x
balance once m is large) is reclassified: annotate it on the figure as a
measured noise-floor effect — when c_H is tiny, the width's dynamics are
dominated by the (z^2 - 1) gradient noise against the log-sigma clamp, which
random-walk-widens sigma above the deterministic balance. State it; do not
tune it. (Optional 1-line verification if cheap: halve LR on that arm only in
a side run and confirm the excess shrinks; report either way.)

## Unchanged

Everything else in the brief and amendment 1 stands: constants, seeds, CRN,
chunked execution, npz checkpoints, plain-language README (five-sentence
summary first, no internal arm names), the boxed batch-shape note from your
first report, and the rule you have followed twice now and should keep
following: report before tuning.
