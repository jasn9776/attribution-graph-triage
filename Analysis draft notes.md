**Rating streams agree moderately and disagree systematically at the 0/1 boundary.** 
Human–LLM kappa 0.68 and 0.67; LLM–LLM 0.82. The count of 2s is stable across streams (6, 5, 7), while 1s vary 
from 6 to 17: the disagreement is concentrated on whether a legible but generic feature at the read-out position, 
with no cross-position edge, counts as a named intermediate (grade 1) or as restating the prompt (grade 0). The 
rubric under-specifies this case; it is reported rather than resolved after the fact.
TODO: Check how consistent the 2's are accross the streams
**Entity-category criterion articulated mid-run.** The criteria did not say whether «dates» features in an entity graph count as restating the prompt. The rater recorded the decision at presentation 48 that they do not. The duplicate pairs show how it applied: p0198 was rated 1 at both presentations 22 and 48, so the rule was already being applied before it was written down; p0186 was rated 0 at presentation 20 and 1 at presentation 45, and is the single non-identical duplicate pair in the human stream. The intra-rater kappa of 0.889 (9 of 10 pairs identical) is therefore attributable to this one documented criterion change rather than to random inconsistency. Ratings were not revised.

Save the numbers that aren't in any file

Several results came from cells you ran ad hoc, and they exist only in the notebook (Day6 01-10-2026) output:


{
  "matched_frames": {
    "n_pairs": 27,
    "mean_diff": 0.0323,
    "median_diff": 0.0196,
    "positive": 27,
    "wilcoxon_p": 1.49e-08,
    "three_frame_means": [
      0.7147,
      0.7378,
      0.7703
    ],
    "friedman_p": 6.14e-06,
    "n_items_three": 12
  },
  "direct_path": {
    "overall_mean": 0.372,
    "corr_replacement": 0.556,
    "corr_rating": 0.033,
    "by_category": {
      "induction": 0.616,
      "factual_recall": 0.615,
      "code_completion": 0.415,
      "obfuscated": 0.384,
      "entity_known_unknown": 0.297,
      "multi_hop": 0.257,
      "syntactic_agree
