# MMedAgent-Bench

Evaluation code and dataset-construction scripts for **MMedAgent-Plus**, a multimodal medical tool-use agent. The model training code is in [MMedAgent-Plus](https://github.com/RoxanneWAAANG/MMedAgent-Plus).

> **Status:** partial release while the paper is under review. This release covers tool-calling evaluation, final-answer (LLM-as-a-judge) evaluation, and two example dataset-construction scripts. Task-specific metrics, the remaining construction pipeline, the baseline scripts and the end-to-end execution harness will follow.

## Repository layout

```
build_dataset/
  build_qa.py                  PubMedQA -> single-tool PMC-LLaMA dialogues (stage 1 example)
  single_tool_multiround.py    single-round -> multi-round dialogues (stage 2)
  corpus_pack/                 greeting / closing phrase banks used by stage 2
  dummy_images/                placeholder image for text-only tools
eval/
  tool_calling/
    tool_call_eval.py          run the agent on test dialogues, score tool calls
    generate_eval_metrics.py   precision / recall / F1, per-task, per-modality, confusion
    recalculate_multi_tool_single_round_llava_cap.py
  final_answer/
    final_answer_eval.py       LLM-as-a-judge: faithfulness, completeness, conciseness
  task_metrics/               placeholder, to be released
```

## Installation

```bash
conda create -n mmedagent-bench python=3.11 -y
conda activate mmedagent-bench
pip install -r requirements.txt
```

## Data

Test data is hosted on Hugging Face: ![Roxanne-WANG/MMedAgent-V2](https://huggingface.co/datasets/Roxanne-WANG/MMedAgent-V2).

Images is stored on Google Drive: [instruction_dataset_images](https://drive.google.com/drive/folders/1-AfZ5Ox1nDgs1hNvNyGBWSBX8cwC0xtu?usp=sharing)

```bash
huggingface-cli download Roxanne-WANG/MMedAgent-V2 --repo-type dataset --local-dir ./data
export MMEDAGENT_DATA_ROOT=$(pwd)/data
```

```
$MMEDAGENT_DATA_ROOT/
  test/
    single_tool_multi_round.jsonl   single-tool multi-round
    multi_tool_multiround.jsonl     multi-tool multi-round
    multi_tool_single_round.jsonl   multi-tool single-round
  train/
    ...
```

Image paths inside the JSONL files are relative to `MMEDAGENT_DATA_ROOT`.

## 1. Tool-calling accuracy

Run the agent on each split. The paper evaluates 500 dialogues per split.

```bash
for split in single_tool_multi_round multi_tool_multiround multi_tool_single_round; do
  python eval/tool_calling/tool_call_eval.py \
    --test-jsonl $MMEDAGENT_DATA_ROOT/test/${split}.jsonl \
    --base-dir   $MMEDAGENT_DATA_ROOT \
    --checkpoint <path-or-HF-id-of-MMedAgent-Plus> \
    --max-samples 500 \
    --out-dir    outputs/tool_calling/${split/multiround/multi_round}
done
```

Then compute the metrics:

```bash
python eval/tool_calling/generate_eval_metrics.py \
  --data-root   $MMEDAGENT_DATA_ROOT \
  --results-dir outputs/tool_calling \
  --output-dir  outputs/tool_calling/reports
```

This writes `cross_split_summary.csv` (whole-case accuracy, tool-call accuracy, micro/macro P/R/F1), per-tool, per-task and per-modality tables, and confusion matrices.

In the multi-tool single-round split, the plain-LLaVA workflows (`MedSAM → LLaVA`, `IterNet → LLaVA`) are over-represented. `recalculate_multi_tool_single_round_llava_cap.py` recomputes the metrics with a cap on those cases:

```bash
python eval/tool_calling/recalculate_multi_tool_single_round_llava_cap.py \
  --turns-csv  outputs/tool_calling/reports/multi_tool_single_round/turns.csv \
  --output-dir outputs/tool_calling/reports/multi_tool_single_round_llava_cap
```

## 2. Final-answer quality (LLM-as-a-judge)

For each tool-use episode, the dialogue is replayed up to the tool output and the agent writes its final response. GPT-5.4-mini then scores three criteria on a 1–5 scale, one API call per criterion, each with its own rubric and three scored few-shot examples:

- **Faithfulness**: consistent with the tool output, with no unsupported claims.
- **Completeness**: includes the task-relevant information from the tool output.
- **Conciseness**: no irrelevant or redundant content.

```bash
export OPENAI_API_KEY=...
python eval/final_answer/final_answer_eval.py \
  --test-jsonl  $MMEDAGENT_DATA_ROOT/test/<split>.jsonl \
  --base-dir    $MMEDAGENT_DATA_ROOT \
  --checkpoint  <path-or-HF-id-of-MMedAgent-Plus> \
  --out-dir     outputs/final_answer/<split>
```

For tools whose output is an image, mask or file (MedSAM, IterNet, HealthGPT, UniGradICON, ...), a short acknowledgment counts as a complete answer. Responses that emit another tool call instead of answering ("wrong-context") receive the minimum score without being sent to the judge. Judge responses are cached in `<out-dir>/judge_cache.json`.

## 3. Task-specific metrics

To be released. See [eval/task_metrics/README.md](eval/task_metrics/README.md) for the planned tools and metrics.

## 4. Dataset construction (examples)

The instruction data is assembled from public benchmarks with fixed templates; no per-sample LLM generation is used. This release includes one first-stage builder and the second-stage multi-round expansion:

```bash
cd build_dataset

# Stage 1: source benchmark -> single-tool dialogues
python build_qa.py \
  --input  /path/to/pubmedqa_artificial.json \
  --output tool_instruct/pmc_llama_medqa_dataset.jsonl

# Stage 2: add greeting / closing rounds -> single-tool multi-round dialogues
python single_tool_multiround.py \
  --input "tool_instruct/*.jsonl" \
  --greetings corpus_pack/greetings.txt \
  --banks corpus_pack/phrase_banks.yaml \
  --output full_data/single_tool_multiround.jsonl
```

Both scripts take `--seed` (default 42).
<!-- 
## Citation

```bibtex
@article{mmedagentplus,
  title  = {MMedAgent-Plus: Scaling Multimodal Medical Agents across Modalities, Tasks and Workflows},
  note   = {Under review},
  year   = {2026}
}
```

## License

Code: Apache-2.0 (see [LICENSE](LICENSE)). Each source benchmark keeps its original license; see the Hugging Face dataset card. -->
