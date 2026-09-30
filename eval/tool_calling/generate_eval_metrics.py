#!/usr/bin/env python3
"""
Compute the tool-calling metrics reported in the paper from tool_call_eval.py outputs.

Outputs:
- tool selection precision / recall / F1
- per-task accuracy
- per-modality accuracy
- error type distribution
- tool confusion matrix

Expected input layout (one tool_call_eval.py run per split):
    <results-dir>/single_tool_multi_round/tool_call_evaluation_results.json
    <results-dir>/multi_tool_multi_round/tool_call_evaluation_results.json
    <results-dir>/multi_tool_single_round/tool_call_evaluation_results.json

The matching test files under <data-root>/test/ supply the prompts and metadata used
for the task / modality breakdowns.

Usage:
    python generate_eval_metrics.py \
        --data-root   $MMEDAGENT_DATA_ROOT \
        --results-dir outputs/tool_calling \
        --output-dir  outputs/tool_calling/reports
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.dpi"] = 160
plt.rcParams["font.size"] = 10


NONE_LABEL = "<NONE>"


@dataclass(frozen=True)
class EvalConfig:
    split_name: str  # sub-directory of --results-dir
    test_file: str   # file under <data-root>/test/


CONFIGS: Tuple[EvalConfig, ...] = (
    EvalConfig(split_name="multi_tool_single_round", test_file="multi_tool_single_round.jsonl"),
    EvalConfig(split_name="single_tool_multi_round", test_file="single_tool_multi_round.jsonl"),
    EvalConfig(split_name="multi_tool_multi_round", test_file="multi_tool_multiround.jsonl"),
)


def canonical_tool(tool: str) -> str:
    return " ".join(str(tool).strip().split()).lower()


def ordered_unique(seq: Sequence[str]) -> List[str]:
    seen = set()
    items = []
    for item in seq:
        key = canonical_tool(item)
        if key not in seen:
            seen.add(key)
            items.append(item)
    return items


def normalize_tool_list(tools: Sequence[str]) -> List[str]:
    return ordered_unique([str(tool).strip() for tool in tools if str(tool).strip()])


def normalized_tool_tuple(tools: Sequence[str]) -> Tuple[str, ...]:
    return tuple(canonical_tool(tool) for tool in normalize_tool_list(tools))


def image_signature(image_value) -> str:
    if isinstance(image_value, list):
        items = [Path(str(x)).name.lower() for x in image_value]
        return "||".join(sorted(items))
    return Path(str(image_value)).name.lower()


def conversation_signature(turn_tool_lists: Sequence[Sequence[str]]) -> Tuple[Tuple[str, ...], ...]:
    return tuple(normalized_tool_tuple(turn) for turn in turn_tool_lists)


def last_user_prompt(conversations: Sequence[dict], gpt_idx: int) -> str:
    for idx in range(gpt_idx - 1, -1, -1):
        if conversations[idx].get("from") == "human":
            return conversations[idx].get("value", "")
    return ""


def extract_gt_turns_from_source(conv_obj: dict) -> List[dict]:
    turns = []
    conversations = conv_obj.get("conversations", [])
    for idx, turn in enumerate(conversations):
        if turn.get("from") != "gpt":
            continue
        actions = turn.get("actions") or []
        if not actions:
            continue
        gt_tools = [action.get("API_name", "") for action in actions if action.get("API_name")]
        turns.append(
            {
                "gt_tools": normalize_tool_list(gt_tools),
                "prompt": last_user_prompt(conversations, idx),
            }
        )
    return turns


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def extract_source_records(source_path: Path) -> Dict[Tuple[str, Tuple[Tuple[str, ...], ...]], List[dict]]:
    records = defaultdict(list)
    for obj in load_jsonl(source_path):
        turns = extract_gt_turns_from_source(obj)
        if not turns:
            continue
        key = (image_signature(obj.get("image", "")), conversation_signature([t["gt_tools"] for t in turns]))
        records[key].append(
            {
                "meta": obj.get("meta", {}),
                "turns": turns,
                "image": obj.get("image", ""),
            }
        )
    return records


def classify_error_type(gt_tools: Sequence[str], pred_tools: Sequence[str]) -> str:
    gt = {canonical_tool(t) for t in gt_tools}
    pred = {canonical_tool(t) for t in pred_tools}

    if gt == pred:
        return "correct"
    if not pred and gt:
        return "missing_all_tools"
    if gt and pred and pred.issubset(gt):
        return "missing_some_tools"
    if gt and pred and gt.issubset(pred):
        return "extra_some_tools"
    if not gt and pred:
        return "unexpected_tool_call"
    return "wrong_tools"


def score_prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def parse_prompt_text(prompt: str) -> str:
    return " ".join(prompt.replace("<image>", " ").split()).lower()


def infer_modality(prompt: str, image_value, source_meta: Optional[dict], gt_tools: Sequence[str]) -> str:
    if source_meta and source_meta.get("modality"):
        return source_meta["modality"]

    prompt_l = parse_prompt_text(prompt)
    image_s = image_signature(image_value)
    tools_l = {canonical_tool(tool) for tool in gt_tools}

    if tools_l == {"medsam", "llava"}:
        return "Gross"
    if tools_l == {"iternet", "llava"}:
        return "Retina-Fundus"
    if tools_l == {"ultrasam", "llava-rad"}:
        return "US"
    if "specialistvlms" in tools_l:
        return "Retina-OCT"
    if "cellvit" in tools_l or "cellsam" in tools_l:
        return "Histology"
    if "conch" in tools_l or "dsmil-c16" in tools_l or "dsmil-tcga" in tools_l:
        return "Pathology"

    if "oct" in prompt_l or "oct" in image_s:
        return "Retina-OCT"
    if "fundus" in prompt_l or "retina" in prompt_l or "eye" in prompt_l or "odir" in image_s or "eyepacs" in image_s:
        return "Retina-Fundus"
    if "gross specimen" in prompt_l or "gross" in prompt_l:
        return "Gross"
    if "ultrasound" in prompt_l or "usg" in image_s or "breast-lesions_usg" in image_s:
        return "US"
    if "x-ray" in prompt_l or "radiograph" in prompt_l or "mimic-cxr" in image_s or "chestxray" in image_s:
        return "X-ray"
    if "mri" in prompt_l or "flair" in image_s or "brain" in prompt_l:
        return "MRI"
    if "ct" in prompt_l or "flare_" in image_s or "word_" in image_s:
        return "CT"
    return "Unknown"


def infer_task(prompt: str, gt_tools: Sequence[str]) -> str:
    prompt_l = parse_prompt_text(prompt)
    tools_l = {canonical_tool(tool) for tool in gt_tools}

    if tools_l == {"medsam", "llava"}:
        return "gross_visual_summary"
    if tools_l == {"iternet", "llava"}:
        return "retinal_vessel_analysis"
    if tools_l == {"unigradicon", "llava-rad"}:
        return "longitudinal_comparison"
    if tools_l == {"ultrasam", "llava-rad"}:
        return "ultrasound_segmentation_report"
    if tools_l == {"medsam", "llava-rad"}:
        return "segmentation_report"

    if "rate-ner" in tools_l or "extract and classify the medical entities" in prompt_l or "entity" in prompt_l:
        return "entity_extraction"
    if "chatcad-r" in tools_l or "merck manual" in prompt_l or "trusted source" in prompt_l:
        return "rag_medical_qa"
    if "unigradicon" in tools_l or "compare and explain the interval changes" in prompt_l or "alignment" in prompt_l:
        return "registration_comparison"
    if "healthgpt" in tools_l or "super-resolution" in prompt_l or "enhance" in prompt_l:
        return "image_enhancement"
    if (
        "medsam" in tools_l
        or "ultrasam" in tools_l
        or "iternet" in tools_l
        or "grounding dino" in tools_l
        or "grounding dino + medsam" in tools_l
        or "cellsam" in tools_l
        or "cellvit" in tools_l
        or "segment" in prompt_l
        or "highlight" in prompt_l
        or "detect" in prompt_l
        or "locate" in prompt_l
    ):
        return "segmentation_detection"
    if "llava-rad" in tools_l or "report" in prompt_l or "impression" in prompt_l or "findings" in prompt_l:
        return "report_generation"
    if "pmc-llama" in tools_l or "medical advice" in prompt_l or "question" in prompt_l:
        return "literature_qa"
    if "specialistvlms" in tools_l or "conch" in tools_l or "dsmil" in " ".join(tools_l) or "biomedclip" in tools_l:
        return "specialist_image_analysis"
    return "general_image_understanding"


def update_confusion_pairs(
    confusion: Dict[str, Counter],
    gt_tools: Sequence[str],
    pred_tools: Sequence[str],
    canonical_to_display: Dict[str, str],
) -> None:
    gt_map = {canonical_tool(tool): tool for tool in normalize_tool_list(gt_tools)}
    pred_map = {canonical_tool(tool): tool for tool in normalize_tool_list(pred_tools)}

    for key, value in gt_map.items():
        canonical_to_display.setdefault(key, value)
    for key, value in pred_map.items():
        canonical_to_display.setdefault(key, value)

    gt_keys = set(gt_map)
    pred_keys = set(pred_map)

    common = sorted(gt_keys & pred_keys)
    gt_only = sorted(gt_keys - pred_keys)
    pred_only = sorted(pred_keys - gt_keys)

    for key in common:
        display = canonical_to_display[key]
        confusion[display][display] += 1

    pair_count = min(len(gt_only), len(pred_only))
    for idx in range(pair_count):
        gt_display = canonical_to_display[gt_only[idx]]
        pred_display = canonical_to_display[pred_only[idx]]
        confusion[gt_display][pred_display] += 1

    for key in gt_only[pair_count:]:
        gt_display = canonical_to_display[key]
        confusion[gt_display][NONE_LABEL] += 1

    for key in pred_only[pair_count:]:
        pred_display = canonical_to_display[key]
        confusion[NONE_LABEL][pred_display] += 1


def find_source_match(
    eval_conv: dict,
    source_index: Dict[Tuple[str, Tuple[Tuple[str, ...], ...]], List[dict]],
) -> Optional[dict]:
    key = (
        image_signature(eval_conv.get("image", "")),
        conversation_signature(eval_conv.get("ground_truth", [])),
    )
    matches = source_index.get(key)
    if matches:
        return matches[0]
    return None


def summarize_metric_rows(rows: List[dict], group_col: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=[group_col, "correct_turns", "total_turns", "accuracy"])
    frame = pd.DataFrame(rows)
    summary = (
        frame.groupby(group_col, as_index=False)
        .agg(correct_turns=("is_correct", "sum"), total_turns=("is_correct", "count"))
        .sort_values(["total_turns", group_col], ascending=[False, True])
    )
    summary["accuracy"] = summary["correct_turns"] / summary["total_turns"]
    return summary


def plot_bar_accuracy(df: pd.DataFrame, label_col: str, title: str, output_path: Path, top_n: Optional[int] = None) -> None:
    if df.empty:
        return
    plot_df = df.copy()
    if top_n is not None:
        plot_df = plot_df.head(top_n)
    plot_df = plot_df.sort_values("accuracy", ascending=True)

    fig, ax = plt.subplots(figsize=(10, max(4, 0.45 * len(plot_df))))
    colors = ["#2E8B57" if x >= 0.8 else "#E69F00" if x >= 0.5 else "#D55E00" for x in plot_df["accuracy"]]
    ax.barh(plot_df[label_col], plot_df["accuracy"], color=colors, edgecolor="black", linewidth=0.8)
    for idx, (_, row) in enumerate(plot_df.iterrows()):
        ax.text(
            min(row["accuracy"] + 0.01, 0.99),
            idx,
            f"{row['accuracy']:.3f} ({int(row['correct_turns'])}/{int(row['total_turns'])})",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Exact-Match Accuracy")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_error_distribution(error_df: pd.DataFrame, output_path: Path, title: str) -> None:
    if error_df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4daf4a" if row == "correct" else "#e41a1c" for row in error_df["error_type"]]
    ax.bar(error_df["error_type"], error_df["count"], color=colors, edgecolor="black", linewidth=0.8)
    for idx, row in error_df.iterrows():
        ax.text(idx, row["count"] + max(1, int(error_df["count"].max() * 0.01)), str(int(row["count"])), ha="center")
    ax.set_ylabel("Turn Count")
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=25)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_tool_prf(per_tool_df: pd.DataFrame, output_path: Path, title: str, top_n: int = 15) -> None:
    if per_tool_df.empty:
        return
    plot_df = per_tool_df.sort_values(["support", "f1"], ascending=[False, False]).head(top_n).copy()
    x = np.arange(len(plot_df))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, plot_df["precision"], width, label="Precision", color="#1b9e77")
    ax.bar(x, plot_df["recall"], width, label="Recall", color="#d95f02")
    ax.bar(x + width, plot_df["f1"], width, label="F1", color="#7570b3")
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df["tool"], rotation=35, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(confusion_df: pd.DataFrame, output_path: Path, title: str, normalize: bool = False) -> None:
    if confusion_df.empty:
        return
    data = confusion_df.copy()
    values = data.to_numpy(dtype=float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)

    fig_w = max(8, 0.55 * len(data.columns))
    fig_h = max(7, 0.45 * len(data.index))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, cmap="Blues", aspect="auto")
    ax.set_xticks(np.arange(len(data.columns)))
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_xticklabels(data.columns, rotation=45, ha="right")
    ax.set_yticklabels(data.index)
    ax.set_xlabel("Predicted Tool")
    ax.set_ylabel("Ground Truth Tool")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Rate" if normalize else "Count")

    threshold = np.nanmax(values) * 0.55 if values.size else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            val = values[i, j]
            raw = int(confusion_df.iloc[i, j])
            if raw == 0:
                continue
            text = f"{val:.2f}" if normalize else str(raw)
            ax.text(j, i, text, ha="center", va="center", color="white" if val > threshold else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def build_confusion_frame(confusion: Dict[str, Counter]) -> pd.DataFrame:
    labels = sorted(set(confusion) | {pred for row in confusion.values() for pred in row})
    if not labels:
        return pd.DataFrame()
    if NONE_LABEL in labels:
        labels = [label for label in labels if label != NONE_LABEL] + [NONE_LABEL]
    matrix = pd.DataFrame(0, index=labels, columns=labels, dtype=int)
    for gt, preds in confusion.items():
        for pred, count in preds.items():
            matrix.loc[gt, pred] += count
    return matrix


def write_markdown_summary(
    output_path: Path,
    split_name: str,
    summary_df: pd.DataFrame,
    task_df: pd.DataFrame,
    modality_df: pd.DataFrame,
    error_df: pd.DataFrame,
) -> None:
    lines = [f"# {split_name}", ""]
    if not summary_df.empty:
        lines.extend(["## Summary", "", summary_df.to_markdown(index=False), ""])
    if not task_df.empty:
        lines.extend(["## Per-Task Accuracy", "", task_df.to_markdown(index=False), ""])
    if not modality_df.empty:
        lines.extend(["## Per-Modality Accuracy", "", modality_df.to_markdown(index=False), ""])
    if not error_df.empty:
        lines.extend(["## Error Types", "", error_df.to_markdown(index=False), ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def evaluate_split(data_root: Path, results_dir: Path, config: EvalConfig) -> Dict[str, pd.DataFrame]:
    result_dir = results_dir / config.split_name
    eval_path = result_dir / "tool_call_evaluation_results.json"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing evaluation file: {eval_path}")

    eval_data = load_json(eval_path)

    source_index = {}
    source_path = data_root / "test" / config.test_file
    if source_path.exists():
        source_index = extract_source_records(source_path)
    else:
        print(f"[warn] {source_path} not found; task/modality breakdowns will use fallbacks.")

    tool_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0, "support": 0, "pred_count": 0})
    turn_rows: List[dict] = []
    error_counter = Counter()
    confusion = defaultdict(Counter)
    canonical_to_display: Dict[str, str] = {}
    matched_conversations = 0

    for conv in eval_data.get("conversation_results", []):
        source_record = find_source_match(conv, source_index) if source_index else None
        if source_record:
            matched_conversations += 1

        source_turns = source_record["turns"] if source_record else []
        source_meta = source_record.get("meta", {}) if source_record else {}

        for turn_idx, (gt_tools, pred_tools) in enumerate(zip(conv.get("ground_truth", []), conv.get("predictions", []))):
            gt_norm = normalize_tool_list(gt_tools)
            pred_norm = normalize_tool_list(pred_tools)

            prompt = source_turns[turn_idx]["prompt"] if turn_idx < len(source_turns) else ""
            modality = infer_modality(prompt, conv.get("image", ""), source_meta, gt_norm)
            task = infer_task(prompt, gt_norm)
            error_type = classify_error_type(gt_norm, pred_norm)
            is_correct = int(error_type == "correct")

            turn_rows.append(
                {
                    "conversation_id": conv.get("conversation_id"),
                    "turn_index": turn_idx,
                    "task": task,
                    "modality": modality,
                    "ground_truth": ", ".join(gt_norm),
                    "prediction": ", ".join(pred_norm),
                    "is_correct": is_correct,
                    "error_type": error_type,
                }
            )
            error_counter[error_type] += 1

            gt_set = {canonical_tool(t) for t in gt_norm}
            pred_set = {canonical_tool(t) for t in pred_norm}
            display_gt = {canonical_tool(t): t for t in gt_norm}
            display_pred = {canonical_tool(t): t for t in pred_norm}
            for key, value in display_gt.items():
                canonical_to_display.setdefault(key, value)
            for key, value in display_pred.items():
                canonical_to_display.setdefault(key, value)

            for tool_key in gt_set & pred_set:
                tool_stats[tool_key]["tp"] += 1
            for tool_key in pred_set - gt_set:
                tool_stats[tool_key]["fp"] += 1
            for tool_key in gt_set - pred_set:
                tool_stats[tool_key]["fn"] += 1
            for tool_key in gt_set:
                tool_stats[tool_key]["support"] += 1
            for tool_key in pred_set:
                tool_stats[tool_key]["pred_count"] += 1

            update_confusion_pairs(confusion, gt_norm, pred_norm, canonical_to_display)

    turn_df = pd.DataFrame(turn_rows)
    task_df = summarize_metric_rows(turn_rows, "task")
    modality_df = summarize_metric_rows(turn_rows, "modality")

    per_tool_rows = []
    for tool_key, stats in sorted(tool_stats.items(), key=lambda item: item[0]):
        precision, recall, f1 = score_prf(stats["tp"], stats["fp"], stats["fn"])
        per_tool_rows.append(
            {
                "tool": canonical_to_display.get(tool_key, tool_key),
                "tp": stats["tp"],
                "fp": stats["fp"],
                "fn": stats["fn"],
                "support": stats["support"],
                "pred_count": stats["pred_count"],
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )
    per_tool_df = pd.DataFrame(per_tool_rows).sort_values(["support", "f1"], ascending=[False, False])

    tp_total = int(sum(row["tp"] for row in per_tool_rows))
    fp_total = int(sum(row["fp"] for row in per_tool_rows))
    fn_total = int(sum(row["fn"] for row in per_tool_rows))
    micro_precision, micro_recall, micro_f1 = score_prf(tp_total, fp_total, fn_total)

    macro_precision = float(per_tool_df["precision"].mean()) if not per_tool_df.empty else 0.0
    macro_recall = float(per_tool_df["recall"].mean()) if not per_tool_df.empty else 0.0
    macro_f1 = float(per_tool_df["f1"].mean()) if not per_tool_df.empty else 0.0

    summary_df = pd.DataFrame(
        [
            {
                "split": config.split_name,
                "conversations": int(eval_data.get("total_conversations", 0)),
                "turns": int(len(turn_df)),
                "matched_source_conversations": matched_conversations,
                "whole_case_accuracy": eval_data.get("metrics", {}).get("whole_case_accuracy", 0.0),
                "tool_call_accuracy_existing": eval_data.get("metrics", {}).get("tool_call_accuracy", 0.0),
                "turn_exact_match_accuracy": float(turn_df["is_correct"].mean()) if not turn_df.empty else 0.0,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": micro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            }
        ]
    )

    error_df = (
        pd.DataFrame([{"error_type": key, "count": value} for key, value in error_counter.items()])
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )
    confusion_df = build_confusion_frame(confusion)

    return {
        "summary": summary_df,
        "turns": turn_df,
        "per_tool": per_tool_df,
        "per_task": task_df,
        "per_modality": modality_df,
        "error_types": error_df,
        "confusion": confusion_df,
    }


def save_split_outputs(output_dir: Path, results: Dict[str, pd.DataFrame], split_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name, df in results.items():
        if isinstance(df, pd.DataFrame):
            keep_index = name == "confusion"
            df.to_csv(output_dir / f"{name}.csv", index=keep_index)

    plot_tool_prf(results["per_tool"], figures_dir / "tool_precision_recall_f1.png", f"{split_name}: Tool Selection PRF")
    plot_bar_accuracy(results["per_task"], "task", f"{split_name}: Per-Task Accuracy", figures_dir / "task_accuracy.png")
    plot_bar_accuracy(results["per_modality"], "modality", f"{split_name}: Per-Modality Accuracy", figures_dir / "modality_accuracy.png")
    plot_error_distribution(results["error_types"], figures_dir / "error_type_distribution.png", f"{split_name}: Error Type Distribution")
    plot_confusion_matrix(results["confusion"], figures_dir / "confusion_matrix_counts.png", f"{split_name}: Tool Confusion Matrix", normalize=False)
    plot_confusion_matrix(results["confusion"], figures_dir / "confusion_matrix_normalized.png", f"{split_name}: Tool Confusion Matrix (Row-Normalized)", normalize=True)

    write_markdown_summary(
        output_dir / "report.md",
        split_name,
        results["summary"],
        results["per_task"],
        results["per_modality"],
        results["error_types"],
    )


def plot_cross_split_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    metric_names = ["micro_precision", "micro_recall", "micro_f1", "turn_exact_match_accuracy", "whole_case_accuracy"]
    x = np.arange(len(summary_df))
    width = 0.16

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = ["#1b9e77", "#d95f02", "#7570b3", "#66a61e", "#e7298a"]
    for idx, metric in enumerate(metric_names):
        ax.bar(x + (idx - 2) * width, summary_df[metric], width, label=metric.replace("_", " "))
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["split"], rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_title("Cross-Split Metric Comparison")
    ax.legend(ncol=2, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute tool-calling metrics from tool_call_eval.py outputs.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Dataset root containing test/*.jsonl.")
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="Directory with one tool_call_eval.py output folder per split.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Directory to write tables and figures.")
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    results_dir = args.results_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = []
    combined_turns = []
    combined_tools = []
    combined_errors = []
    combined_confusion = defaultdict(Counter)
    canonical_to_display: Dict[str, str] = {}

    for config in CONFIGS:
        split_results = evaluate_split(data_root, results_dir, config)
        split_output_dir = output_dir / config.split_name
        save_split_outputs(split_output_dir, split_results, config.split_name)

        all_summaries.append(split_results["summary"])

        split_turns = split_results["turns"].copy()
        split_turns["split"] = config.split_name
        combined_turns.append(split_turns)

        split_tools = split_results["per_tool"].copy()
        split_tools["split"] = config.split_name
        combined_tools.append(split_tools)

        split_errors = split_results["error_types"].copy()
        split_errors["split"] = config.split_name
        combined_errors.append(split_errors)

        conf_df = split_results["confusion"]
        for gt in conf_df.index:
            for pred in conf_df.columns:
                count = int(conf_df.loc[gt, pred])
                if count:
                    combined_confusion[gt][pred] += count
                    if gt != NONE_LABEL:
                        canonical_to_display.setdefault(canonical_tool(gt), gt)
                    if pred != NONE_LABEL:
                        canonical_to_display.setdefault(canonical_tool(pred), pred)

    summary_df = pd.concat(all_summaries, ignore_index=True) if all_summaries else pd.DataFrame()
    summary_df.to_csv(output_dir / "cross_split_summary.csv", index=False)

    if combined_turns:
        combined_turn_df = pd.concat(combined_turns, ignore_index=True)
        combined_turn_df.to_csv(output_dir / "all_turns.csv", index=False)
        summarize_metric_rows(combined_turn_df.to_dict("records"), "task").to_csv(output_dir / "combined_per_task.csv", index=False)
        summarize_metric_rows(combined_turn_df.to_dict("records"), "modality").to_csv(output_dir / "combined_per_modality.csv", index=False)

    if combined_tools:
        pd.concat(combined_tools, ignore_index=True).to_csv(output_dir / "all_per_tool_metrics.csv", index=False)
    if combined_errors:
        pd.concat(combined_errors, ignore_index=True).to_csv(output_dir / "all_error_types.csv", index=False)

    combined_conf_df = build_confusion_frame(combined_confusion)
    combined_conf_df.to_csv(output_dir / "combined_confusion.csv", index=True)

    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    plot_cross_split_summary(summary_df, figures_dir / "cross_split_summary.png")
    plot_confusion_matrix(combined_conf_df, figures_dir / "combined_confusion_counts.png", "Combined Tool Confusion Matrix", normalize=False)
    plot_confusion_matrix(combined_conf_df, figures_dir / "combined_confusion_normalized.png", "Combined Tool Confusion Matrix (Row-Normalized)", normalize=True)

    report_lines = [
        "# Tool Calling Evaluation",
        "",
        "Generated splits:",
        "",
        *(f"- {config.split_name}" for config in CONFIGS),
        "",
    ]
    if not summary_df.empty:
        report_lines.extend(["## Cross-Split Summary", "", summary_df.to_markdown(index=False), ""])
    (output_dir / "README.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved enhanced metrics to: {output_dir}")


if __name__ == "__main__":
    main()
