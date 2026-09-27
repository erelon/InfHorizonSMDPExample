# Figure 2 (amendment 5) — constants solved from I1-I4, and predictions, written BEFORE the 5-seed grid

Fixed by the amendment: eta = 0.03, p = 0.8, k = 0.02, w = 0.3, h = 1.0, T = 20000, event at t = 6000.
The derivations use the single-seed (seed 0) no-floor trace only. No boost-arm outcome was run before this file.

## Chosen constants

| constant | value | check that fixed it |
|---|---|---|
| EMA memory M | 250 | Middle of [150, 400]. I2 holds for any M here, because the no-floor collapse is a stable equilibrium (permanent). |
| reference start t_ref | 500 | I1 vs I3 is the binding pair. I1 needs ref >= 0.15; ref = s_slow(500) = 0.154 (seed 0). A later t_ref breaks I1; an earlier one raises the threshold, which raises the trigger-regulated flat width and breaks I3. |
| B_max | 2 | I3: best-mu at the boosted flat width must be within 0.3 of c. Among B_max in [2, 5], only B_max = 2 qualifies (width 1.10). |
| bump position c | 1.5 | I3: best-mu(1.095) = 1.25 and best-mu(1.116) = 1.24 (0.25 / 0.26 from c). c = 1.6 is marginal (0.27 / 0.28); c >= 1.8 fails. |
| skirt | none | I3 is feasible without it. |

### I3 note: the stated formula understates the boosted width
The formula sqrt(B_max * c_H_base / 2k), with c_H_base = p*eta, gives 1.095. But the no-floor arm's base dose
grows with width: std(A) = sqrt(eta^2 + 2 k^2 sigma^4) on the flat. Whenever B > sqrt(2)/p = 1.77 (B_max = 2 is
above that), the flat therefore has no balance. The width then stops only when std(A) reaches the threshold and
the boost starts decaying: sigma_reg = sqrt(sqrt(thr^2 - eta^2) / (sqrt(2) k)) = 1.116. I3 is checked at both
widths, and both pass. Both are below the acquisition ceiling (h w / 2k)^(1/3) = 1.957.

## Load-bearing numbers (seed 0)
- reference 0.1543, threshold 0.0463
- settled no-floor std(A) 0.0364 < threshold (margin 1.27x)
- EMA crosses the threshold at t = 1923

## Pre-registered predictions

1. Phase A, frozen dose and warmup floor: identical, with the starving flag silent after t_ref.
2. Phase A, boost arm: fires at t = 1923 +- 300, per-seed median.
3. Phase A, boost arm: from then on it is REGULATED, not decaying (I4). Closed form: sigma* = 0.0669,
   boost* = x / (p thr) = 1.35, with x = sqrt(2) sqrt(thr^2 - eta^2). Scored as the median over [4000, 6000)
   within +-25% of each.
4. Event: reward crashes in every arm.
5. After the event:
   - The flat balance of the frozen / floor dose is 2.38, above the ceiling, so they stall near mu ~ 0.9-1.0
     (late escapes are reported).
   - The unaided no-floor arm balances on the flat at sigma = 0.853 (live-tracked with noise:
     0.64(eta^2 + 2k^2 s^4) = 4k^2 s^4), where best-mu = 1.36. It therefore recovers slowly, as in amendment 4.
   - The boost arm's flat width is trigger-limited at about 1.12 (best-mu 1.24), so it also recovers.
6. **Speed ratio, no-floor / boost: predicted 1.4-2.0, so the pre-registered ">= 3x" is predicted to FAIL.**
   - At the event, the boost arm's dose advantage is boost* x s_slow ratio = 1.35 x (0.046 / 0.036) ~ 1.7.
   - The hold guard freezes the boost once log-sigma starts rising, about 100-200 iterations after the event.
     So the boost can grow by at most ~e^(200/600) = 1.4x before it is held, giving a ceiling of ~2.4x at best.
   - Once the width rises, std(A) exceeds the threshold, so the boost decays rather than grows.
7. After re-acquisition (I4): the boost arm returns to the phase-A regulated band (boost ~1.35, sigma ~0.067) and
   does NOT settle below 1.3. This is scored as the median over the last 4000 iterations within +-25% of the band.
8. sigma <= 3 in every arm. The largest width is the frozen / floor flat balance, ~2.4.
