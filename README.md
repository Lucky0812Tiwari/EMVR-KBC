# LLM-enhanced Symbolic Reasoning for KBC + EMVR-KBC

## Introduction

This repository provides resources for the paper "Large Language Model-Enhanced
Symbolic Reasoning for Knowledge Base Completion" (LeSR).

**This fork/checkout additionally includes EMVR-KBC** (Evidence-guided
Multi-stage Verification and Refinement for KBC), a research extension that
inserts a lightweight evidence-verification stage between LeSR's LLM Proposer
and its Rule Reasoner, filtering out unsupported candidate rules *before* the
expensive `O(|E|²)` grounding step, and warm-starts the Rule Reasoner's
significance weights from that same evidence. LeSR's own grounding math,
loss function, Subgraph Extractor, and LLM Proposer are all unchanged — see
`EMVR-KBC section` below for details.

## Dependencies

- Python 3.9.19
- transformers==4.40.1
- scikit-learn==1.4.2
- scipy==1.13.0
- torch==2.3.0
- transformers==4.40.1
- nltk==3.8.1
- sentence-transformers==3.0.0
- sentencepiece==0.2.0
- openai==1.24.0
- google-generativeai==0.7.2
- groq==0.31.0
- numpy, pandas, PyYAML *(used throughout the code, not listed in the original paper repo)*
- faiss-cpu>=1.7.4 *(new — EMVR-KBC only, optional; needed only if you enable `--emvr_conceptnet_patterns`)*

All of the above are pinned/listed in `requirements.txt`. Install everything
in one step:
```bash
pip install -r requirements.txt
```

## Code Files

The dataset for UMLs/WN18RR/FB15K found [here](https://github.com/DeepGraphLearning/RNNLogic), CN100 could be found [here](https://home.ttic.edu/~kgimpel/commonsense.html) and WD15K could be found [here](https://github.com/THU-KEG/BIMR) with the interpretability annotations. RotatE could be found [here](https://github.com/DeepGraphLearning/KnowledgeGraphEmbedding/) and `kge.py` includes part of the said implementation. The main python file is `lesr.py`. We include the FB15K relation mapping in `fb15k_rels.csv` and example commands in `run_example.sh`. To run LeSR, please create subdirectories `data/`, `log/`, `runs/`, move dataset and kge (if used) to their respective folder, and provide own LLM inference api key. To use other knowledge base data, please edit the commandline parse and add data reading in `data.py`.

| File | Status | Purpose |
|---|---|---|
| `lesr.py` | modified | main pipeline entry point; adds the EMVR-KBC hooks described below |
| `reasoner.py` | unchanged | rule checking, grounding math (`tensor_logic_*_semantic`), `ReasonerModel`/`ReasonerModelPlus`, training loop |
| `extractor.py` | unchanged | Subgraph Extractor |
| `proposer.py` | unchanged | LLM Proposer |
| `data.py` | unchanged | dataset loading/encoding |
| `kge.py` | unchanged | RotatE embedding baseline |
| `utils.py` | unchanged | misc I/O helpers |
| `evidence.py` | **new** | EMVR-KBC: evidence sampling, verification, warm-start, post-hoc metrics |
| `emvr_eval_example.py` | **new** | example script for Filtering Recall / EV–Weight correlation |
| `requirements.txt` | **new** | consolidated dependency list (see above) |

## EMVR-KBC section

### What it changes

EMVR-KBC inserts one new stage into the pipeline, between `prepare_rules()`
and `ground_rules_over_kg_semantic()` in the grounding loop of `lesr.py`:

```
Subgraph Extractor → LLM Proposer → [EMVR-KBC: evidence verification] → Rule Reasoner (grounding + training, unchanged) → Inference
```

For each candidate rule it samples up to `S` body-satisfying entity pairs via
sparse adjacency BFS (cheap, `O(S·d)`, independent of `|E|`), checks how many
also satisfy the rule head, and compares the resulting score `EV_i` against a
relation-density-adaptive threshold `τ_r`. Rules that don't clear `τ_r` are
dropped before they ever reach the `O(|E|²)` grounding step. Optionally, for
very sparse relations, an external similarity signal against a small
ConceptNet-style pattern set can corroborate the internal evidence.
Verified rules can then warm-start `ReasonerModel.raw_weights` directly from
`EV_i` instead of the default zero/random initialization — the training loss
in `reasoner.py::train_loop` is untouched, so this only changes *where*
gradient descent starts from, never the objective itself.

### New command-line flags (all default to off/unchanged behaviour)

| Flag | Default | Meaning |
|---|---|---|
| `--use_emvr` | off | enable EMVR-KBC evidence verification before grounding |
| `--emvr_sample_size` | 500 | max sampled entity pairs per rule (S in Eq. 1) |
| `--emvr_beta` | 0.7 | weight on internal KB evidence vs. external similarity (Eq. 5) |
| `--emvr_tau_min` | 0.1 | verification threshold floor (Eq. 7) |
| `--emvr_tau_max` | 0.5 | verification threshold ceiling (Eq. 7) |
| `--emvr_external_density_gate` | 0.001 | only compute external similarity when relation density is below this |
| `--emvr_conceptnet_patterns` | `None` | path to a JSON list of relation-chain pattern strings; omit to disable external evidence entirely |
| `--use_emvr_warmstart` | off | warm-start `ReasonerModel.raw_weights` from `EV_i` |
| `--emvr_save_stats` | off | persist per-relation evidence rows + EV scores under `<run_dir>/reasoner/`; required for `--use_emvr_warmstart` to find its EV scores |

With `--use_emvr` omitted, the pipeline behaves exactly as vanilla LeSR.

### Example run (mirrors `run_example.sh`, with EMVR-KBC enabled on stage 3)

```bash
dataset_name=UMLs
run_name=umls_emvr_run
api_key=API_KEY_HERE
llm_name=GPT35

# Stage 1: subgraph extraction (unchanged)
python -u lesr.py --run_name ${run_name} --dataset ${dataset_name} --run_extractor --rand_seed 5

# Stage 2: LLM proposer (unchanged)
python -u lesr.py --run_name ${run_name} --dataset ${dataset_name} \
  --run_proposer --llm_name ${llm_name} --llm_max_input_chars 4096 --llm_api_key ${api_key}

# Stage 3: reasoner, with EMVR-KBC verification + warm-start
python -u lesr.py --run_name ${run_name} --dataset ${dataset_name} \
  --run_reasoner --kge_bsize 32 \
  --use_emvr --emvr_save_stats --use_emvr_warmstart \
  --emvr_sample_size 500 --emvr_beta 0.7 \
  --emvr_tau_min 0.1 --emvr_tau_max 0.5
```

### Where to look at results

- **Metrics** (MR, MRR, Hit@1/3/10) print to console during stage 3 and are
  written to `runs/<run_name>/logging.txt`, per-relation and pooled over the
  whole test set — identical `compute_metrics()` output format to vanilla LeSR.
- **EMVR-KBC diagnostics**: `runs/<run_name>/reasoner/{relation_id}_emvr_stats.json`
  (per-rule `KBSample`, `EV_i`, `τ_r`, verified/discarded) and
  `{relation_id}_ev_scores.json` (the EV values used for warm-start), written
  when `--emvr_save_stats` is set.
- **Grounding matrices** (body-support / full-rule masks, analogous to `C_i`/`A_i`
  in the proposal): `runs/<run_name>/grounding/{relation_id}_train_mask_chained.pt`
  and `..._mask_aligned.pt`, loadable with `torch.load(...)`.
- **Post-hoc EMVR-vs-vanilla comparison** (Filtering Recall, EV–Weight
  correlation): see `emvr_eval_example.py`.

### Files not modified

`reasoner.py`, `data.py`, `utils.py`, `extractor.py`, `proposer.py`, `kge.py`
are byte-for-byte the same as upstream LeSR — EMVR-KBC only adds `evidence.py`
and modifies `lesr.py`'s orchestration (argparse flags, two insertion points
in the grounding/reasoning loops). See `IMPLEMENTATION_GUIDE.md` for the
exact before/after code at each insertion point, and `lesr_emvr.patch` for a
reviewable diff.
