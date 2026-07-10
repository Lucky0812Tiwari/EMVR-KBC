"""
emvr_eval_example.py — example usage of evidence.py's post-hoc metrics
(Section XIII-B/C of the EMVR-KBC proposal): Filtering Recall and
EV-Weight correlation.

Run this AFTER you have:
  (a) a vanilla LeSR run's learned_weights.pt for some relation, and
  (b) an EMVR-KBC run's checked_rules + emvr_stats["rows"] + learned_weights.pt
      for the same relation.

This script does not require re-running any training; it only reads saved
artifacts and applies evidence.py's metric functions.
"""

import json
import torch

from evidence import filtering_recall, ev_weight_correlation, grounding_flops_saved


def load_rule_texts(rules_json_path):
    with open(rules_json_path) as f:
        rules = [json.loads(line) for line in f]
    return [r[0] for r in rules]  # rule_text is index 0


def main(vanilla_weights_path, vanilla_rules_path,
         emvr_weights_path, emvr_rules_path, theta_hi=0.5):
    # --- Filtering Recall (Eq. 18) ---
    vanilla_weights = torch.load(vanilla_weights_path)          # tensor, len == n_rules(+1)
    vanilla_rule_texts = load_rule_texts(vanilla_rules_path)     # same order as weights[:-1]
    high_weight_texts = [
        text for text, w in zip(vanilla_rule_texts, vanilla_weights[:-1])
        if w.item() > theta_hi
    ]

    emvr_rule_texts = load_rule_texts(emvr_rules_path)  # already the verified/kept set

    recall = filtering_recall(emvr_rule_texts, high_weight_texts)
    print("Filtering Recall (theta_hi={}): {}".format(theta_hi, recall))

    # --- EV-Weight correlation (Sec. XIII-C.1) ---
    emvr_weights = torch.load(emvr_weights_path)
    # ev_scores must be recovered from the saved emvr_stats rows at training
    # time (see IMPLEMENTATION_GUIDE.md §3d) and saved alongside the run,
    # e.g. json.dump([row["ev_i"] for row in emvr_stats["rows"] if row["verified"]], f)
    with open(emvr_rules_path.replace("_rules.json", "_ev_scores.json")) as f:
        ev_scores = json.load(f)

    corr = ev_weight_correlation(ev_scores, emvr_weights[:-1].tolist())
    print("EV-Weight correlation: pearson={}, spearman={}".format(
        corr["pearson"], corr["spearman"]))

    # --- Illustrative FLOPs saved (Eq. 15-17) ---
    flops = grounding_flops_saved(
        n_candidates=len(vanilla_rule_texts),
        n_verified=len(emvr_rule_texts),
        n_entities=14541,  # example: FB15K-237 entity count, replace as needed
    )
    print("Grounding FLOPs saved: {:.1%}".format(flops["pct_saved"]))


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--vanilla_weights", required=True)
    p.add_argument("--vanilla_rules", required=True)
    p.add_argument("--emvr_weights", required=True)
    p.add_argument("--emvr_rules", required=True)
    p.add_argument("--theta_hi", type=float, default=0.5)
    args = p.parse_args()
    main(args.vanilla_weights, args.vanilla_rules,
         args.emvr_weights, args.emvr_rules, args.theta_hi)
