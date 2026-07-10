# Running LeSR + EMVR-KBC

This package contains the LeSR files with EMVR-KBC already applied, ready to
drop into a clone of `hyanique/LeSR`.

## Files in this package

| File | What it is |
|---|---|
| `evidence.py` | **New file.** The entire EMVR-KBC contribution (evidence sampling, verification, warm-start, metrics). |
| `lesr.py` | **Modified.** Your original `lesr.py` with the EMVR-KBC hooks applied (see `lesr_emvr.patch` for the exact diff). Everything else in the file is untouched. |
| `lesr_emvr.patch` | Unified diff of `lesr.py` (original repo version → patched version), so you can `git apply` it directly instead of copying the file, or just review exactly what changed. |
| `IMPLEMENTATION_GUIDE.md` | Explanation of *why* each change sits where it sits, mapped to the proposal's equations/sections. |
| `emvr_eval_example.py` | Standalone script for the post-hoc metrics (Filtering Recall, EV–Weight correlation) once you have both a vanilla and an EMVR-KBC run. |

`reasoner.py`, `data.py`, `utils.py`, `extractor.py`, `proposer.py`, `kge.py`
are **not included** because they are not modified — keep your existing
copies of those.

## 1. Set up the repo

```bash
git clone https://github.com/hyanique/LeSR.git
cd LeSR
# install LeSR's own requirements first (however the repo documents that,
# e.g. pip install -r requirements.txt)
pip install faiss-cpu          # new, only needed if you use external evidence
```

## 2. Drop in the EMVR-KBC files

```bash
cp /path/to/downloaded/evidence.py .
```

Then either:

**Option A — apply the patch (recommended, keeps you close to upstream):**
```bash
cp /path/to/downloaded/lesr_emvr.patch .
git apply lesr_emvr.patch
```

**Option B — replace the file directly:**
```bash
cp /path/to/downloaded/lesr.py .
```
(Only do this if you haven't made your own local edits to `lesr.py` — Option B
will overwrite them.)

## 3. Sanity check the patch applied cleanly

```bash
python -c "import lesr"   # should import without errors
python lesr.py --help | grep emvr
```
You should see the new flags: `--use_emvr`, `--emvr_sample_size`,
`--emvr_beta`, `--emvr_tau_min`, `--emvr_tau_max`,
`--emvr_external_density_gate`, `--emvr_conceptnet_patterns`,
`--use_emvr_warmstart`, `--emvr_save_stats`.

## 4. First run — baseline, unchanged behaviour

Confirm nothing broke by running exactly what `run_example.sh` already does,
with no EMVR flags. This should behave identically to vanilla LeSR.

```bash
bash run_example.sh
```

## 5. Second run — EMVR-KBC verification only (no warm-start yet)

Take whatever command `run_example.sh` uses and add:

```bash
python lesr.py \
  --dataset <your_dataset> \
  --run_dir <run_dir> \
  --run_reasoner \
  --use_emvr \
  --emvr_sample_size 500 \
  --emvr_beta 0.7 \
  --emvr_tau_min 0.1 \
  --emvr_tau_max 0.5 \
  --emvr_save_stats \
  --debug \
  <...whatever other flags run_example.sh already passes...>
```

Watch the console for lines like:
```
[EMVR-KBC] relation 12=has_part: 7/19 rules verified, filtering_rate=63.2%, tau_r=0.134
```
Compare `tau_r` against the worked example in the proposal
(density_r=0.125 → tau_r≈0.15) to confirm the adaptive threshold behaves as
expected on your dataset's density range.

With `--emvr_save_stats`, per-relation files appear under
`<run_dir>/reasoner/`:
- `{relation_id}_emvr_stats.json` — full per-rule evidence rows
- `{relation_id}_ev_scores.json` — EV_i for the final verified+kept rule set (used by warm-start)

## 6. Third run — add warm-start

Same command, plus:
```bash
  --use_emvr_warmstart
```
Console should print:
```
[EMVR-KBC] warm-started 7 logical-rule weights from EV_i (loss function L1 unchanged)
```
Training then proceeds exactly as in `train_loop()` (unchanged) — you're only
changing where gradient descent starts from.

## 7. Compare against vanilla LeSR

Run steps 4 and 6 on the **same dataset with the same `--run_dir` cleared
between runs** (or different `--run_dir`s), then compare the printed
MR/MRR/Hit@k from `compute_metrics()` — these functions are untouched, so the
numbers are directly comparable.

## 8. Post-hoc metrics (optional, after you have both runs)

```bash
python emvr_eval_example.py \
  --vanilla_weights <vanilla_run_dir>/reasoner/<rel_id>_learned_weights.pt \
  --vanilla_rules   <vanilla_run_dir>/reasoner/<rel_id>_rules.json \
  --emvr_weights    <emvr_run_dir>/reasoner/<rel_id>_learned_weights.pt \
  --emvr_rules      <emvr_run_dir>/reasoner/<rel_id>_rules.json
```
This prints Filtering Recall (Eq. 18), EV–Weight Pearson/Spearman
correlation, and an illustrative grounding-FLOPs-saved percentage.

## 9. Enable external evidence (optional, sparse datasets only)

Build a small JSON file of ConceptNet-style relation-chain pattern strings,
e.g. `patterns.json`:
```json
["born in -> located in -> country", "parent -> parent -> grandparent", "..."]
```
Then add:
```bash
  --emvr_conceptnet_patterns patterns.json
```
This only activates for relations with `density_r < --emvr_external_density_gate`
(default 0.001) — per Table II of the proposal, that's chiefly CN100 and
WN18RR. If `sentence-transformers`/`faiss-cpu` aren't installed or the file
is missing, EMVR-KBC prints a warning and falls back to internal-only
evidence automatically — it won't crash the run.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'evidence'`** — make sure
  `evidence.py` is in the same directory as `lesr.py` (repo root), not a
  subfolder.
- **`AssertionError` in `apply_warmstart`** — the number of verified rules
  saved to `{relation_id}_ev_scores.json` no longer matches the rule count
  at training time (e.g. you changed `--good_rule_criteria` between the
  grounding run and the reasoning run). Delete the run's `reasoner/` and
  `grounding/` directories and re-run steps 5-6 from scratch so grounding and
  reasoning stay in sync.
- **Every rule gets filtered out (`filtering_rate=100%`)** — check
  `--emvr_tau_min`/`--emvr_tau_max` aren't inverted, and check the printed
  `tau_r` isn't absurdly high for a sparse relation; also verify
  `train_arr`'s triple encoding matches what `prepare_rule_map_relations`
  expects (should be automatic, but worth a `--debug` run on one relation
  first).
