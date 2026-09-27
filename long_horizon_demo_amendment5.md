# Amendment 5 to the long-horizon demo — figure 2: the boost joins the arm that can starve

Your amendment-4 run is accepted as shipped, and its two findings join the
boxed notes: (5) the ACQUISITION CEILING — through width sigma a bump of
height h, scale w only out-pulls a bowl of curvature 2k while
sigma < (h*w/2k)^(1/3); excess width blocks acquisition, a second cost channel
of over-width that exists even on non-terminating tasks; (6) DETECTORS HAVE
TIMESCALES — a trigger smoothed over memory M cannot confirm a starvation that
the system itself repairs in < M; sticky starvation exists only where the dose
cannot self-recover.

Note 6 exposes the design error that has been mine for three rounds: the boost
was attached to the WELL-DOSED arm, whose collapses are self-healing
transients in every toy we have built — so it had nothing to detect. In the
parent system the boost's validated rescue (Ant) was of an UNDER-DELIVERED
dose whose sigma-death was a stable equilibrium. The toy's faithful analogue
is therefore boost-on-the-under-dosed-arm, and your "unpredicted result" (the
no-floor arm alone recovering, slowly, while the well-dosed arms stall at the
ceiling) is the spine of the corrected figure, not an anomaly.

## Arms (figure 2, final structure; everything not mentioned is amendment 4's)

1. frozen dose (p = 0.8): the WELL-DOSED reference. Expected post-event: flat
   balance ~2.38 > ceiling 1.96 -> stalls near mu ~ 0.9 (your measurement);
   any late tail-escapes reported as such.
2. tracked, no floor: the under-dosed arm, unaided. Spirals to the
   noise-rescued width in phase A (corrected p/sqrt(2) overlay, as delivered);
   post-event recovers SLOWLY (~2600 iters measured) because its small dose
   keeps sigma under the ceiling.
3. tracked + warmup floor: well-dosed twin of arm 1 in this world (floor
   binds); stalls with it. Kept because the frozen-vs-floored equivalence
   under falling std is itself a shipped-rule fact.
4. **tracked, no floor, + boost**: the story arm. Same spiral exposure as
   arm 2; the trigger detects the sticky under-dose and the boost lends width.

## Pre-registered INEQUALITIES (you solve the constants; I fix only bounds)

You have out-derived my arithmetic five times; this amendment therefore fixes
the design by inequalities and licenses you to choose the constants that
satisfy them, within the stated bounds, reporting every chosen value and the
check that fixed it. Constants you may set: reference start t_ref in
[200, 1500]; EMA memory M in [150, 400]; B_max in [2, 5]; bump position c in
[1.2, 2.2]; optionally a broad skirt (second Gaussian, same centre, height
<= 0.4, scale in [0.8, 1.5]) if and only if inequality (I3) is otherwise
infeasible. eta = 0.03, p = 0.8, k = 0.02, w = 0.3, h = 1.0, T = 20000,
event at t = 6000 are fixed.

- (I1) REFERENCE CAPTURES HEALTH: the running-max reference on arm 4 must be
  taken while the arm is still healthy — t_ref before the spiral completes —
  so that reference ~ the healthy-band std, not the collapsed one. Verify on
  a single seed before the grid: reference >= 0.15.
- (I2) STICKINESS BEATS THE DETECTOR: the un-boosted spiral/collapse must hold
  std(A) below 0.3 * reference for longer than the EMA fall time (~2M). Verify
  with arm 2's trace.
- (I3) CAPTURE BEFORE OVERSHOOT: the boosted flat balance
  sqrt(B_max * c_H_base / 2k) must stay below the acquisition ceiling
  (h_eff * w_eff / 2k)^(1/3) of the (possibly skirted) bump, AND best-mu(sigma)
  at the boosted width must reach within 0.3 of the bump centre (your report-3
  computation, run as preflight).
- (I4) DECAY OR REGULATION, STATED IN ADVANCE: after re-acquisition, if
  on-bump std clears the threshold, predict boost decay below 1.3 within 3000
  iters; if your preflight arithmetic instead predicts a regulated limit cycle
  (decay -> spiral -> re-fire), pre-register the band and score against the
  band. Either outcome is a pass if predicted; an unpredicted one is a fail.

## Expectations (scored after the inequalities are satisfied on paper)

- Phase A: arms 1 and 3 identical and silent. Arms 2 and 4 spiral together
  until the trigger fires on arm 4 (timestamp predicted from I1/I2 arithmetic,
  tolerance +-300); from there arm 4 is boost-sustained in a predicted width
  band — the boost acting as an ADAPTIVE FLOOR, which is the mechanism's
  phase-A exhibit, replacing "silent in A" for this arm.
- Event: reward crashes in all arms. Arm 4 recovers to the 0.8x-post-ideal
  threshold at least 3x faster than arm 2 (the same under-dose, unaided) and
  faster than any well-dosed arm that recovers at all; arms 1/3 stall at the
  ceiling (report late escapes); sigma <= 3 everywhere.
- Panels add: the ceiling (h_eff*w_eff/2k)^(1/3) as a horizontal line on the
  sigma row, and best-mu(sigma) as a small inset — the two curves that decide
  who can come home.
- Epilogue unchanged. Boxed notes now number six; the README states that
  notes 2-6 came from failed pre-registered runs and that the final design's
  constants were solved from inequalities I1-I4 rather than guessed.

## Scope guard

This amendment does not touch figure 1, the epilogue, or the shipped
amendment-4 material (archive it beside amendment 1's). If I1-I4 cannot be
jointly satisfied within the bounds — including with the skirt — do not run
the grid: report the infeasibility with the binding pair of inequalities, and
we ship the amendment-4 package with note 6 as the final word on why a
one-state toy cannot host this controller. That outcome is acceptable; a
tuned illusion is not.
