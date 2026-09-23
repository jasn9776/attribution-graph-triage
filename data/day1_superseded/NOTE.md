# Superseded Day 1 outputs

Computed with the metric implementation corrected on 22-09-2026 (preregistration D3.2:
logit nodes were located structurally and could be mis-ordered).

**Wrong in these files:** rep, comp, pos_gini, err_final. Final-position error reads 0.0
throughout; it is in fact usually the largest error position (D3.10).

**Still valid:** attr_s, metric_s, features, headroom, peak_gb — these are unaffected by
the metric bug and are the evidence for the timing and feature-cap decisions.

Both runs used max_feature_nodes = 24576; the final setting is 32768.

The live reference graph is data/reference_france.json, written during the Day 3 run.