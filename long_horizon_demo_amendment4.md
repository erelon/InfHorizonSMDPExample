# Amendment 4 to the long-horizon demo — figure 2, final design

Your std(A) = c_H/sqrt(2) identity is verified, accepted, and promoted to the
figure's organizing result: in a deterministic per-sample-advantage world,
every settled width equilibrium pins the realised pressure at sqrt(2), so a
variance-relative observable is SELF-REFERENTIAL — the coefficient generates
the signal meant to calibrate it. This single identity explains all three of
your amendment-3 findings (the pressure pin, the trigger blindness after
transients, the impossibility of delivery drift at any bump equilibrium). The
consequence you asked me to decide is adopted: exogenous reward variance is a
NECESSARY CONDITION for variance-relative control, proven by the identity, and
figure 2 adds the minimal amount. This is the last redesign: the change is
derived, every prediction below is closed-form, and if this run fails its
criteria we ship the passing material plus the boxed theorems as the product.

## The one new ingredient

i.i.d. Gaussian reward noise per sample: r <- r + eta * xi, xi ~ N(0,1),
**eta = 0.03**, present in ALL arms and phases of figure 2 (figure 1 is closed
and unchanged). Everything else in the machinery stays as amendment 3 built it.

Noise-corrected closed forms (draw each as its panel's prediction):
- Advantage spread: std(A) = sqrt(eta^2 + S^2), S = |R''| sigma^2 / sqrt(2).
- Settled-state pressure: c_H / sqrt(eta^2 + (c_H/sqrt 2)^2) — draw the
  sqrt(2) line on the pressure panel as "the deterministic-limit pin"; the
  measured curve sits just below it while locked (signal-dominated) and jumps
  to ~c_H/eta when the signal term dies. The pressure observable becomes
  informative exactly when exploration currency dies — that sentence goes in
  the README.
- Live-tracked fixed point (the corrected p/sqrt(2) law): with x = |R''| sigma^2,
  balance at x = p*eta / sqrt(1 - p^2/2); at p = 0.8, eta = 0.03, C = 11.1
  on-bump: sigma* = sqrt(0.97*eta/C) ~= 0.051. The no-floor arm no longer dies
  at the clamp; it settles at the noise-rescued width, far below healthy —
  overlay this prediction AND the raw p/sqrt(2) = 0.566 line AND the
  EMA-lagged measurement, per your suggestion (both-lines).

## Figure 2 phases and event (phase B is REMOVED — the identity forbids
## delivery drift at any settled bump equilibrium; figure 1 owns delivery)

- Phase A [0, 5000): single bump h = 1.0 at c = -0.7, floor -0.02*a^2.
  Settle; settled reference from t = 2000.
- Event at t = 5000: bump vanishes, reappears at **c = +2.0**, h = 1.0
  (peak reward 1 - 0.02*4 = 0.92). Geometry chosen against your excess-width
  trap: the fixed-dose flat balance is sigma_flat = sqrt(c_H / (2k)) ~= 2.4,
  and at 2.0 = 0.83*sigma_flat the smeared bump overlaps the policy —
  acquisition pulls mu, and NARROWING IS DONE BY THE BUMP's own contraction
  after arrival, never demanded of the (one-sided) boost. PRE-FLIGHT CHECK
  (before running seeds): compute best-mu(sigma) as you did in report 3 and
  verify it reaches within 0.3 of +2.0 at sigma = sigma_flat and at the
  boosted width; if not, move the bump inward and report the value used.
- Phase D [5000, 14000): recovery and boost decay. T = 14000.

Arms unchanged from amendment 3 (frozen / tracked-no-floor / tracked+warmup-
floor / +boost; p = 0.8; hold guard; settled reference; f = 0.3; cap 8).

## Pre-registered expectations (all noise-corrected)

- Phase A: arms 1, 3, 4 identical; trigger silent after t = 2000; arm 2
  settles at sigma ~= 0.051 (the corrected-law overlay), NOT the clamp.
- Event: std(A) falls to ~eta = 0.03 against a settled reference
  ~sqrt(eta^2 + (c_H/sqrt2)^2) ~= 0.166; threshold 0.3*0.166 = 0.050 > eta,
  so starving fires on arm 4 within 300 iterations. (This inequality,
  eta < 0.3 * reference, is the design's load-bearing arithmetic — print the
  three numbers in the summary.)
- Escape: ordering by 1/c_H — arm 4 fastest (boosted c_H up to ~1.8), arms
  1 = 3 slower, arm 2 slowest (its c_H ~ p*eta ~ 0.024 on the flat) but now
  FINITE (the noise floor rescues it eventually; report its time, do not
  suppress it — it is the corrected law's other prediction). Criterion:
  arm 4 at least 3x faster than arms 1/3 to the recovery threshold.
- Recovery threshold: trailing-200 window entirely post-event,
  0.8 * post-event ideal = 0.8 * 0.92 = 0.736.
- Boost decays below 1.3 within 3000 iters of arm 4's recovery (after
  re-acquisition std returns to ~0.166 > 0.050, so not-starving holds).
- sigma <= 3 in all arms (report if exceeded rather than clamp-tune).
- Epilogue panel: unchanged, as delivered (it passes; it stays).

## README bookkeeping

The boxed notes now number four, and together they are the exercise's thesis:
1. The offset's residue is accidental exploration (one machinery per failure
   mode).
2. Variance-relative triggers detect width collapse, not entrapment at a worse
   optimum (detectors have scope; entrapment belongs to the optimism/reset
   family).
3. Live-scale calibration has no fixed point on a noise-free peak
   (p/sqrt(2)); exogenous variance of at least p*eta/sqrt(1-p^2/2) in
   curvature units is what rescues it.
4. At any settled equilibrium of a deterministic width channel,
   std(A) = c_H/sqrt(2): the observable is self-referential, so exogenous
   advantage variance is a NECESSARY condition for variance-relative control —
   the property of real environments that this mechanism reads.
State plainly in the README that notes 2-4 were derived by the implementing
agent from failed pre-registered runs, and that the final figure's noise term
exists because of note 4, not despite it. Process unchanged: report before
tuning; mark revised vs original criteria; chunked execution; five-sentence
plain summary first.
