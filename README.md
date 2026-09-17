# Diffusion Layer Integrated Gradients (DLIG)

Code to reproduce the experiments in *Temporally-Resolved Token Attribution Reveals the Generation Dynamics of Diffusion Language Models*. [Paper (arXiv)](https://arxiv.org/abs/XXXX.XXXXX) · [Models (Hugging Face)](https://huggingface.co/collections/darpanaswal/dlig)

## Setup

```bash
pip install -r requirements.txt
```

Set `HUGGINGFACE_API_KEY` in a `.env` file at the repo root if you need to pull checkpoints from the Hub (only `helpers/hf_transfer.py` requires it).

All experiments finetune or evaluate **DiffuGPT-M**, a masked-diffusion adaptation of GPT-2-medium (~355M params). Checkpoints (base + the WiC and ProsQA finetunes) are available on the Hugging Face page above; download and place them under `models/Diffugpt` (base), `models/diffugpt-m-wic`, and `models/diffugpt-m-prosqa` before running any experiment below. See also [Appendix C: Model, Datasets, and Training](#appendix-c-model-datasets-and-training).

```bash
python -m helpers.hf_transfer download \
    --repo_id darpanaswal/DLIG/diffugpt-m-wic \
    --local_dir models/diffugpt-m-wic
```

---

## § 3.1 / Appendix A.1 — Implementation Checks (IG–DLIG Correspondence)

Verify the preconditions the IG–DLIG correspondence relies on: hook transparency, completeness, Riemann-sum convergence, and the bidirectionality/shift gates.

### Completeness grid (Figure 6)

```bash
python -u -m experiments.theorems.verify_completeness_grid \
    --family diffugpt \
    --model_path models/diffugpt-m-prosqa \
    --torch_dtype float32 \
    --generation_steps 12 \
    --check_steps 1 3 5 7 9 11 \
    --m_list 200 1000 \
    --layers 0 2 4 6 8 10 12 14 16 18 20 22 \
    --out_dir runs/verify_completeness
```

### Plot Figure 6

```bash
python -u -m experiments.theorems.plot_completeness_grid \
    --log_dir runs/verify_completeness \
    --out_dir Plots/completeness \
    --m 1000
```

### Partial-forward spot check (Figure 7)

```bash
python -u -m experiments.theorems.spot_check_partial \
    --family diffugpt \
    --model_path models/diffugpt-m-prosqa \
    --torch_dtype bfloat16 \
    --layers 0 6 11 \
    --step 2 --m 12 --score_mode logprob
```

### Plot Figure 7

```bash
python -u -m experiments.theorems.plot_spot_check_partial \
    --log runs/spot_check_partial/spot_check_partial.txt \
    --out Plots/spot_check_partial.png
```

---

## § 4 — Experimental Setup: Task Performance (Table 1, Appendix A.2)

Evaluate DiffuGPT-M's task accuracy/ROUGE on each of the three tasks (writes prediction files under `outputs/<task>/eval_task_preds.jsonl`, reused below):

```bash
python -m helpers.eval_task --task wic \
    --model_path models/diffugpt-m-wic --data data/wic_test_raw.jsonl

python -m helpers.eval_task --task prosqa \
    --model_path models/diffugpt-m-prosqa --data data/prosqa_test.json

python -m helpers.eval_task --task infill \
    --model_path models/Diffugpt --data data/rocstories_test.jsonl
```

Optional sanity check — confirms batched and single-example generation agree on the real checkpoints (underlies every sharded attribution run below):

```bash
python -m helpers.verify_batching --task wic \
    --model_path models/diffugpt-m-wic --data data/wic_test_raw.jsonl --n 6
```

---

## § 5 — Word-Sense Disambiguation (WiC)

### DLIG attribution (self-generated target)

```bash
python -u -m experiments.wic.wic \
    --model_path models/diffugpt-m-wic \
    --wic_jsonl data/wic_test_raw.jsonl \
    --out_file outputs/wic/wic_dlig.jsonl \
    --m 12 --chunk 12 --gen_steps 64 --max_new_tokens 6 \
    --score_mode meancentered \
    --target_steps 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 \
    --layers 0 2 4 6 8 10 12 14 16 18 20 22
```

Shard across GPUs with `--num_shards N --shard_id i` (one process per GPU) and concatenate the shard outputs, as in `args/wic.sh`.

### Commitment timing

```bash
python -u -m experiments.wic.wic_commitment \
    --model_path models/diffugpt-m-wic \
    --wic_jsonl data/wic_test_raw.jsonl \
    --out_file outputs/wic/wic_commitment.jsonl \
    --gen_steps 64 --max_new_tokens 6
```

### Plot Figure 1 (single-example layer panel)

```bash
python -m helpers.analyze_wic --panel \
    --dlig outputs/wic/wic_dlig.jsonl \
    --panel_pick correct_yes \
    --panel_out_file outputs/wic/figs/figure1_panel.png
```

Use `--list_candidates` to browse examples and pick a specific `--panel_idx`.

### Plot Figure 2 (attribution depth & commitment timing; Table 6/7)

```bash
python -m helpers.analyze_wic \
    --dlig outputs/wic/wic_dlig.jsonl \
    --commit outputs/wic/wic_commitment.jsonl \
    --depth_plot --depth_plot_out outputs/wic/figs/figure2a_depth.png \
    --commit_plot --commit_plot_out outputs/wic/figs/figure2b_commit.png \
    --out_dir outputs/wic/figs
```

Run the same command without `--depth_plot`/`--commit_plot` to print the full Table 6 statistics (Mann–Whitney AUC, Hartigan's dip test, BIC) to stdout; add `--empty_report` to regenerate the Table 7 zero-mass footnote.

---

## § 6 — Multi-Hop Graph Reasoning (ProsQA)

### Bucket eval predictions into success / fail / off-manifold

```bash
python -u -m experiments.prosqa.bucket_prosqa \
    --in_file outputs/prosqa/eval_task_preds.jsonl \
    --out_file outputs/prosqa/prosqa_buckets_full.jsonl
```

### Build the fact graph and gold-path/cited-chain labels

```bash
python -u -m experiments.prosqa.prosqa_graph_labels \
    --buckets outputs/prosqa/prosqa_buckets_full.jsonl \
    --out_file outputs/prosqa/prosqa_graph_labels.jsonl
```

### Contrastive DLIG over the reasoning graph

```bash
python -u -m experiments.prosqa.prosqa_contrastive_dlig \
    --graph_labels outputs/prosqa/prosqa_graph_labels.jsonl \
    --out_file outputs/prosqa/prosqa_dlig.jsonl \
    --model_path models/diffugpt-m-prosqa \
    --groups success fail \
    --m 12 --chunk 12 --gen_steps 64 --max_new_tokens 64 \
    --target_steps 1 7 13 19 25 31 37 43 49 54 60 \
    --layers 0 2 4 6 8 10 12 14 16 18 20 22
```

Shard across GPUs with `--num_shards N --shard_id i`, as in `args/prosqa_dlig.sh`.

### Plot Figures 3–4 (Table 8)

```bash
python -u -m helpers.analyze_prosqa \
    --dlig_file outputs/prosqa/prosqa_dlig.jsonl \
    --graph_labels outputs/prosqa/prosqa_graph_labels.jsonl \
    --out_dir outputs/prosqa
```

Writes the Figure 3a–c / Figure 4 panels and `prosqa_dlig_report.txt` (full Table 8 statistics) under `outputs/prosqa/`.

### Table 2 worked examples (Appendix A.3)

Table 2's one-example-per-bucket table is read directly off `outputs/prosqa/prosqa_buckets_full.jsonl`, filtered by `bucket`.

---

## § 7 — Sentence Infilling (ROCStories)

### DLIG attribution (self-generated target, primary)

```bash
python -u -m experiments.infill.attribution_infill \
    --family diffugpt \
    --dataset data/rocstories_test.jsonl \
    --n_samples 1000 --max_side_tokens 120 \
    --m 8 --chunk 12 --gen_steps 64 \
    --score_mode meancentered \
    --target_mode self \
    --target_steps 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 \
    --layers 0 2 4 6 8 10 12 14 16 18 20 22 \
    --out_file outputs/infill_attribution/diffugpt_self.jsonl
```

Shard across GPUs with `--num_shards N --shard_id i` and concatenate the shard outputs, as in `args/attribution_infill.sh`.

### Plot Figure 5a (context-reliance vs. infill-quality correlation)

```bash
python -m helpers.analyze_infill \
    --family diffugpt --target_mode self \
    --input_file outputs/infill_attribution/diffugpt_self.jsonl \
    --mass_per_token --mass_scatter
```

### Plot Figure 5b (signed-distance attribution profile)

```bash
python -m helpers.analyze_infill \
    --family diffugpt --target_mode self \
    --input_file outputs/infill_attribution/diffugpt_self.jsonl \
    --heatmap
```

---

## Extended Analyses (Appendix A.4)

Robustness checks for §7, run at your discretion — these receive secondary emphasis in the paper.

### Normalization robustness (Figures 8–9, Table 3/4)

```bash
python -m helpers.analyze_infill \
    --family diffugpt --target_mode self \
    --input_file outputs/infill_attribution/diffugpt_self.jsonl \
    --split_rouge median --mass_scatter --mass_auc --ratio_r
```

### Fixed-target framing (Figures 10–11, Table 5)

First re-run attribution with the gold sentence held fixed as the target:

```bash
python -u -m experiments.infill.attribution_infill \
    --family diffugpt \
    --dataset data/rocstories_test.jsonl \
    --n_samples 1000 \
    --target_mode gold \
    --m 8 --chunk 12 --gen_steps 64 \
    --score_mode meancentered \
    --target_steps 1 3 5 7 9 11 13 15 17 19 21 23 25 27 29 31 33 35 37 39 41 43 45 47 49 51 53 55 57 59 61 63 \
    --layers 0 2 4 6 8 10 12 14 16 18 20 22 \
    --out_file outputs/infill_attribution/diffugpt_gold.jsonl
```

Then re-run the § 7 analysis commands with `--target_mode gold --input_file outputs/infill_attribution/diffugpt_gold.jsonl`.

---

## Appendix B: Statistical Tables

Full Mann–Whitney AUC, Wilcoxon signed-rank, dip-test, and bootstrap-CI statistics for every experiment are printed to stdout as a byproduct of the `helpers.analyze_*` commands above (WiC → Table 6/7; ProsQA → Table 8; ROCStories infilling → Table 3/4/5). No separate reproduction steps are provided here.

---

## Appendix C: Model, Datasets, and Training

All three tasks finetune or evaluate DiffuGPT-M, a masked-diffusion adaptation of GPT-2-medium (~355M params). WiC uses a 4,928/638/500 train/val/test split (test carved from the official train set, since WiC test labels are private); ProsQA uses the standard 17,886/300/500 split; ROCStories infilling evaluates over 1,000 held-out stories on the base (non-finetuned) checkpoint. Fine-tuning hyperparameters and prompt formats are listed in the paper (Appendix C, Tables 11–13). Download checkpoints from the [Hugging Face page](https://huggingface.co/darpanaswal/DLIG), or finetune your own with [our fork of the DiffuGPT repo](https://github.com/darpanaswal/diffugpt), which provides the exact scripts we used to train these models. Place checkpoints under `models/Diffugpt`, `models/diffugpt-m-wic`, and `models/diffugpt-m-prosqa` (see `utils/config.py` for exact paths) before running any experiment above.

---

## Citation

```bibtex
@article{aswal2026temporally,
  title={Temporally-Resolved Token Attribution Reveals the Generation Dynamics of Diffusion Language Models},
  author={Aswal, Darpan and Hudelot, C{\'e}line},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2026}
}
```
