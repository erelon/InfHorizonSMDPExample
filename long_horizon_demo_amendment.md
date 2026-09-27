# Amendment 1 to the long-horizon demo brief — decision after your report

Your diagnosis is accepted in full and is correct: the GAE batch-shape gives the
centred advantage a residual spread ~3.6*eps, the jump makes eps huge, and that
spread IS exploration — so phase C cannot starve anyone in the sine world. That
is not a tuning problem; it is the two failure modes interacting. The original
project hit the same wall and resolved it the same way we will now: ONE
MACHINERY PER STORY. Do not pursue your option A (lambda = 0 everywhere kills
the offset story's physics) and do not ship option C. We take your option B AND
split the starvation/boost story into a second figure built on the machinery
where starvation is structural.

## Figure 1 (revise the existing run): offset + delivery, sine world

Keep: GAE lambda = 0.95, stale score, all four... no — THREE arms only
(no fixes / frozen dose / tracked dose). Drop the boost arm and row 5 from this
figure entirely; the boost belongs to figure 2.

Changes:
1. **Your option B**: r = m(t)*(1 - min((a-g)^2, 4)) + ramp(t). The ramp is no
   longer scaled by m, so eps stays ~0.15 and std(A) tracks the reward scale
   (~5x through B), which is what the delivery story needs.
2. **Drop phases C/D from this figure's claims.** Run to t = 12000 with phases
   A [0,4000) and B [4000,12000) only (extend B so the tracked arm's EMA has
   room to demonstrate steady delivery after the ramp ends: hold m = 5 for
   [9000,12000)). No jump in this figure.
3. **Tracking memory**: shorten the slow-EMA memory to 400 iterations (the 800
   memory against a 5x ramp over 5000 iters is what produced your 53%
   deviation). Revised criterion: tracked realised pressure within 30% of 0.2
   over all of A-B except the first 400 iterations of the ramp, and within 15%
   during the m = 5 hold.
4. **Row 1 readability**: plot the 50-iteration running median of |b| (measured
   and predicted) on a log axis instead of the raw signed series; keep the bold
   after-centring zero line on a small linear inset or as a stated annotation.
5. All other rows, constants, seeds, and the arm-1 acceptance criteria stand.
   Recompute the frozen arm's balance-prediction row with the new reward form.

## Figure 2 (new): starvation + the controller, two-bump world, long horizon

This is the machinery from the project's earlier starvation demo, where
starvation is structural (narrow reward bumps over a weak curvature floor:
a contracted policy on a bump has literally zero gradient toward the other
one), extended to a long horizon with the controller arms.

Environment (per iteration, batch N = 256; tau = 1):
- r(a) = h1*exp(-(a+0.7)^2/(2*0.3^2)) + h2*exp(-(a-0.7)^2/(2*0.3^2)) - 0.10*a^2
- Phase A [0, 6000): h = (1.0, 0.6). Long on purpose: sigma must fully settle
  at its equilibrium so the swap meets a genuinely contracted policy.
- Phase C at t = 6000: swap to h = (0.6, 1.3). Phase D [6000, 16000): recovery.
- Policy/critic/eta exactly as figure 1 EXCEPT: lambda = 0 (per-sample
  advantage — no batch shape, by design and by this amendment's premise), and
  the score is recomputed at CURRENT parameters with ONE full-batch update per
  iteration, step = 0.05 * grad / max(1, |grad|/5). This is the width-dynamics
  integration the earlier demo validated; it is the machinery in which the
  earlier run showed the no-boost arm stuck at the weak bump FOREVER.
- Init mu = -0.7, sigma = 0.3. Centring ON in all arms.

Arms (5 seeds, medians):
- frozen dose: c_H = p * std(A at batch 1), p = 0.8.
- tracked dose: c_H = p * slow-EMA(std(A)) (memory 400), p = 0.8, floored at
  0.25x the batch-1 scale (a tracked scale in this world is pro-cyclical after
  lock-on — std(A) falls as sigma contracts — so the floor is load-bearing;
  log when the floor binds, it is part of the story).
- tracked + boost: the PD controller on top: starving := s_slow < 0.3 *
  running_peak(s_slow); guard: no boost growth while the 200-iter slope of
  log sigma is positive; log-domain boost, e-fold 600 iters, cap 8.

Rows: sigma (log, with the two bumps' balance predictions per phase), mu against
the bump centres, reward with the per-phase ideal (0.951 / 1.251) and the
predicted-at-held-width dashed line, realised pressure, boost + starving flag,
plus a small text panel with recovery times.

Pre-registered expectations:
- Phase A: all three arms identical within noise (trigger silent — if it fires
  during initial lock-on, your s_slow seed or memory is wrong).
- After the swap: the frozen and tracked arms may escape eventually (the swap
  weakens bump 1's curvature below c_H, so sigma grows), but slowly — their
  time-to-recover is set by the fixed pressure; the boost arm must recover at
  least 3x faster and its boost must decay below 1.3 within 4000 iterations of
  recovery. If the no-boost arms escape nearly as fast as the boost arm at
  p = 0.8, reduce p to 0.4 in ALL arms (one shared change, report it): the
  differentiation lives in the ratio of escape speeds, and the earlier demo
  measured escape to fail entirely at p <= 0.6 with a different floor — expect
  the working point between 0.4 and 0.8.
- The starving flag must fire within 500 iterations of the swap in the boost
  arm (lock-on makes std(A) tiny relative to its own peak; that is the trigger
  working as designed).

## Recovery metric (both figures, replaces the broken one)

recovery(t_event) := first t > t_event such that the trailing 200-iter mean of
perf exceeds 0.9 * (its own mean over the 1000 iterations before t_event),
with perf measured in the CURRENT phase's units (figure 1: divide by m(t);
figure 2: subtract nothing, the scale is fixed). "Never" = not by end of run.
Your diagnosis of the old metric (phase-D mean sits below late-B, threshold
unreachable) is accepted; this replaces it.

## Deliverables and process

Same as the brief (script per figure or one script with FIG env var, .npz
checkpoints, chunked foreground, README in plain language, five-sentence plain
summary first). Re-run the acceptance table with the revised criteria; the
report should state plainly which criteria were REVISED by this amendment and
which are original, so the record shows what moved and why. Your batch-shape
finding goes in the README as a boxed note: it is a real result of this
exercise — the offset's residue acts as accidental exploration, which is why
the two failure modes need separate machineries to be seen cleanly.
