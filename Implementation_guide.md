# EMVR-KBC → LeSR Integration Guide

Based on inspecting `hyanique/LeSR` (files: `lesr.py`, `reasoner.py`, `data.py`,
`utils.py`, `kge.py`, `extractor.py`, `proposer.py`). Confirms the proposal's
premise: `lesr.py` calls `prepare_rules()` (reasoner.py) to get `checked_rules`,
then later calls `ground_rules_over_kg_semantic()` (also reasoner.py) which does
the expensive O(|E|²) sparse-tensor grounding for **every** candidate rule,
with no quality gate in between. `ReasonerModel.raw_weights` (reasoner.py,
`nn.Parameter(torch.zeros(num_rj))`) is randomly initialized before training.

## 1. New file needed

| File | Purpose |
|---|---|
| `evidence.py` | **New module.** Everything EMVR-KBC adds: adjacency-based BFS evidence sampling (Eq. 1), optional external ConceptNet similarity (Eq. 4), adaptive threshold + verification (Eq. 6-8), warm-start logits for `ReasonerModel` (Eq. 10), and post-hoc metrics (Filtering Recall, EV–Weight correlation). No existing file is modified in its class/function definitions — `evidence.py` only imports `prepare_rule_map_relations` and `RELATION_ID2Text_MAPPING_MODE` from `reasoner.py` and `remove_wikidata_prefix` from `data.py`, both reused unchanged. |

## 2. Extra requirements

Add to the existing dependency list (`README.md` / your `requirements.txt`):

```
faiss-cpu>=1.7.4      # optional: only needed if you enable external evidence
networkx>=3.2          # optional: not strictly required by evidence.py's own
                        # BFS (pure dict-based), but handy if you later swap
                        # in networkx for the chain traversal
```

Everything else (`sentence-transformers`, `torch`, `scikit-learn`, etc.) is
already in LeSR's dependency list and is reused as-is.

## 3. Two integration points in `lesr.py` (no other files touched)

### 3a. Argparse — add EMVR flags

Insert after the existing `--good_rule_criteria` argument (around line 55):

```python
    parser.add_argument('--use_emvr', action='store_true',
        help='enable EMVR-KBC evidence verification before grounding')
    parser.add_argument('--emvr_sample_size', type=int, default=500,
        help='S in Eq.(1): max sampled entity pairs per rule for KBSample')
    parser.add_argument('--emvr_beta', type=float, default=0.7,
        help='beta in Eq.(5): weight on internal KB evidence vs external sim')
    parser.add_argument('--emvr_tau_min', type=float, default=0.1)
    parser.add_argument('--emvr_tau_max', type=float, default=0.5)
    parser.add_argument('--emvr_external_density_gate', type=float, default=0.001,
        help='only compute ExternalSim when density_r is below this')
    parser.add_argument('--emvr_conceptnet_patterns', type=str, default=None,
        help='optional path to a .json list of ConceptNet relation-chain '
             'pattern strings for ExternalSim; omit to disable external evidence')
    parser.add_argument('--use_emvr_warmstart', action='store_true',
        help='warm-start ReasonerModel.raw_weights from EV_i instead of zeros')
```

### 3b. Import — add `evidence` module functions

Change the existing `from reasoner import ...` block's neighbourhood
(top of `lesr.py`) by adding one new import line right after it:

```python
from reasoner import prepare_rules, get_model_and_tokenizer, get_phrase_embedding, custom_similarity, find_similar_phrases, convert_arr_to_sparse_coo,  load_grounding_results, save_grounding_results, ground_rules_over_kg_semantic, keep_good_rules, print_grounding_stats, get_unique_eval_query_ids, ReasonerModel, ReasonerModelPlus, train_loop, get_triplets_and_scores, compute_metrics,test_loop,test_loop_plus

from evidence import (build_adjacency, verify_rules_for_relation, apply_warmstart,
                       ExternalEvidenceIndex, filtering_recall, ev_weight_correlation,
                       grounding_flops_saved)
```

### 3c. Verification stage — insert BEFORE grounding

This is the exact gap identified in the proposal (Section IV-C, "inserts...
between LeSR's unchanged LLM proposer and its unchanged matrix-grounding Rule
Reasoner"). In `lesr.py`, the grounding loop currently reads (unmodified
excerpt, `else:` branch when `checked_rules_fname` doesn't exist yet):

```python
        else:
            all_rules_fname = os.path.join(reasoner_dir, "{}_rules_all.json".format(relation_wk_idx))
            checked_rules = load_nested_list(all_rules_fname)
            logging.info("load {} checked rules from disk for relation {}={}".format(len(checked_rules), relation_wk_idx, relation_text))
            train_sparse = convert_arr_to_sparse_coo(train_arr, n_entities, n_relations)
            train_aligned_masks, train_chained_masks, train_ground_results,train_ground_rconds = ground_rules_over_kg_semantic(train_sparse, checked_rules, relation2id_dict, similarity_matrix,verbose=False)
            checked_rules, train_aligned_masks, train_chained_masks, train_ground_results,train_ground_rconds = keep_good_rules(...)
```

Insert the EMVR-KBC verification stage right between `train_sparse = ...`
and the `ground_rules_over_kg_semantic(...)` call:

```python
        else:
            all_rules_fname = os.path.join(reasoner_dir, "{}_rules_all.json".format(relation_wk_idx))
            checked_rules = load_nested_list(all_rules_fname)
            logging.info("load {} checked rules from disk for relation {}={}".format(len(checked_rules), relation_wk_idx, relation_text))
            train_sparse = convert_arr_to_sparse_coo(train_arr, n_entities, n_relations)

            # ---------------- EMVR-KBC: evidence verification (NEW) ----------------
            ev_scores_for_warmstart = None
            if args.use_emvr:
                fwd_adj, bwd_adj = build_adjacency(train_arr)
                external_index = None
                if args.emvr_conceptnet_patterns:
                    import json
                    with open(args.emvr_conceptnet_patterns) as f:
                        patterns = json.load(f)
                    external_index = ExternalEvidenceIndex(pattern_strings=patterns)
                checked_rules, ev_scores_for_warmstart, emvr_stats = verify_rules_for_relation(
                    checked_rules, relation2id_dict, train_arr, n_entities,
                    target_relation_id=relation_wk_idx, fwd=fwd_adj, bwd=bwd_adj,
                    sample_size=args.emvr_sample_size, beta=args.emvr_beta,
                    tau_min=args.emvr_tau_min, tau_max=args.emvr_tau_max,
                    external_density_gate=args.emvr_external_density_gate,
                    external_index=external_index, verbose=args.debug)
                print("[EMVR-KBC] relation {}={}: {}/{} rules verified, "
                      "filtering_rate={:.1%}, tau_r={:.3f}".format(
                          relation_wk_idx, relation_text, emvr_stats["n_verified"],
                          emvr_stats["n_candidates"], emvr_stats["filtering_rate"],
                          emvr_stats["tau_r"]))
                if len(checked_rules) == 0:
                    print("no rules survived EMVR-KBC verification for relation={}, skip".format(relation))
                    relation_wo_logicrules.append(relation)
                    continue
            # -------------------------------------------------------------------------

            train_aligned_masks, train_chained_masks, train_ground_results,train_ground_rconds = ground_rules_over_kg_semantic(train_sparse, checked_rules, relation2id_dict, similarity_matrix,verbose=False)
            checked_rules, train_aligned_masks, train_chained_masks, train_ground_results,train_ground_rconds = keep_good_rules(checked_rules, train_aligned_masks, train_chained_masks, train_ground_results,train_ground_rconds,verbose=args.debug, criteria=args.good_rule_criteria)
```

Notes:
- `ground_rules_over_kg_semantic()` itself is called **unchanged** — it just
  now receives a shorter `checked_rules` list.
- `keep_good_rules()` still runs afterward exactly as in LeSR (it can further
  drop rules that failed to chain on the *full* KG even after verification —
  that's fine and expected; it doesn't affect `ev_scores_for_warmstart`
  correctness because we regenerate the aligned EV list at warm-start time
  from `checked_rules`' final rule_text set — see 3d).
- Since `checked_rules` and the associated masks are re-used for `test_sparse`
  and `valid_sparse` grounding calls a few lines below, those automatically
  benefit from the same filtered rule set — no separate change needed there.

### 3d. Warm-start — insert at reasoner construction

Currently (`lesr.py`, inside the `else:` branch that builds a fresh model):

```python
                else:
                    print("init and train the reasoner")
                    n_rules = n_rules +1
                    if args.use_reasoner_plus:
                        model = ReasonerModelPlus(n_rules) 
                    else:
                        model = ReasonerModel(n_rules) 
                    if not args.run_reasoner_models_on_cpu:
                        model.to("cuda")
                    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
```

Insert the warm-start call right after `model.to("cuda")` (or after model
construction if running on CPU):

```python
                else:
                    print("init and train the reasoner")
                    n_rules = n_rules +1
                    if args.use_reasoner_plus:
                        model = ReasonerModelPlus(n_rules) 
                    else:
                        model = ReasonerModel(n_rules) 
                    if not args.run_reasoner_models_on_cpu:
                        model.to("cuda")

                    # ---------------- EMVR-KBC: warm-start init (NEW) ----------------
                    if args.use_emvr and args.use_emvr_warmstart:
                        # checked_rules here is the post-verification, post-keep_good_rules
                        # list (its rule_text is unique and stable); recover each verified
                        # rule's EV_i from emvr_stats["rows"] by rule_text lookup so the
                        # warm-start vector aligns 1:1 with the final `n_rules - 1` logical
                        # rules the model was constructed with.
                        ev_by_text = {row["rule_text"]: row["ev_i"] for row in emvr_stats["rows"]}
                        aligned_ev = [ev_by_text.get(cr[0], 0.0) for cr in checked_rules]
                        model = apply_warmstart(model, aligned_ev)
                        print("[EMVR-KBC] warm-started {} logical-rule weights from EV_i "
                              "(loss function L1 unchanged)".format(len(aligned_ev)))
                    # -------------------------------------------------------------------

                    optimizer = optim.AdamW(model.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
```

`ReasonerModel`/`ReasonerModelPlus` class definitions in `reasoner.py` are
**not edited** — `apply_warmstart()` only assigns `.data` on the already
existing `nn.Parameter`, which is exactly the zero-risk property argued in
Section VII-A of the proposal (gradient descent on the unmodified loss then
proceeds normally from this initialization).

## 4. Recommended order of implementation

1. Drop `evidence.py` into the LeSR repo root (same folder as `lesr.py`).
2. Add the 8 argparse flags (§3a) — everything defaults to `False`/off, so
   `run_example.sh` keeps working unmodified until you opt in with `--use_emvr`.
3. Add the import line (§3b).
4. Add the verification-stage insertion (§3c). Run once with
   `--use_emvr --debug` on one small relation and confirm the printed
   `n_verified/n_candidates` and `tau_r` look sane (compare to the worked
   example in the proposal, Section IX: density_r=0.125 → tau_r≈0.15).
5. Add the warm-start insertion (§3d), run with `--use_emvr --use_emvr_warmstart`.
6. Compare metrics (MRR/Hit@k from `compute_metrics()`, unchanged) between a
   `--use_emvr` run and a vanilla run on the same dataset — this is your A1
   vs. B0 ablation from Table in Section XIII-D.
7. Optionally wire up `filtering_recall()` and `ev_weight_correlation()` as a
   small standalone analysis script (see `emvr_eval_example.py`) once you have
   both a vanilla LeSR `learned_weights.pt` and an EMVR-KBC run to compare.
8. Only after 4-7 work end-to-end on one dataset, enable external evidence
   (`--emvr_conceptnet_patterns path/to/patterns.json`) for the sparsest
   datasets (CN100, WN18RR) per Table II of the proposal.

## 5. What is intentionally *not* touched

- `reasoner.py`'s grounding math (`tensor_logic_*_semantic`, `ground_rules_over_kg_semantic`,
  `keep_good_rules`) — identical to LeSR.
- `ReasonerModel` / `ReasonerModelPlus` class bodies and `train_loop()`'s loss
  (`-log(sum(w_i * s_ij))`) — identical to LeSR (Eq. 9/11 in the proposal).
- `extractor.py`, `proposer.py`, `kge.py`, `data.py` — untouched (Subgraph
  Extractor, LLM Proposer, RotatE embedding baseline are all unchanged per
  the proposal's architecture diagram).
