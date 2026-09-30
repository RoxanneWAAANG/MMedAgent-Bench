#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
single_tool_multiround.py

Build multi-round dialogues from single-round tool conversations by optionally
inserting (1) a greeting round at the beginning and (2) a closing round at the end.

Input:  One or more JSONL files where each line is a dict that includes
        a 'conversations' array (single-round dialogue) and arbitrary metadata.
Output: A JSONL file of sessions with augmented multi-round conversations.

Example:
python3 single_tool_multiround.py \
  --input "tool_instruct/*.jsonl" \
  --greetings corpus_pack/greetings.txt \
  --banks corpus_pack/phrase_banks.yaml \
  --output full_data/single_tool_multiround.jsonl
"""

import argparse
import glob
import json
import os
import random
from typing import Dict, List, Any, Sequence

import yaml


# ──────────────────────────────────────────────────────────────────────────────
# I/O utilities
# ──────────────────────────────────────────────────────────────────────────────
def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into a list of dictionaries (skip blank lines)."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(data: Sequence[Dict[str, Any]], path: str) -> None:
    """
    Write a list of dictionaries to JSONL.
    We escape literal newlines to keep each example on one physical line.
    """
    with open(path, "w", encoding="utf-8") as f:
        for row in data:
            s = json.dumps(row, ensure_ascii=False)
            s = s.replace("\n", "\\n").replace("\r", "\\r")
            f.write(s + "\n")


def load_lines(path: str) -> List[str]:
    """Load a UTF-8 text file into a list of non-empty, stripped lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def load_banks(path: str) -> Dict[str, Any]:
    """Load a YAML file that contains phrase banks (e.g., starters/closings)."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Conversation builders
# ──────────────────────────────────────────────────────────────────────────────
def create_greeting_round(greetings: List[str], banks: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create an opening greeting round:
      human: greeting text (from greetings.txt if available, else banks.starters)
      gpt  : short small-talk response (from banks.agent_smalltalk)
    """
    human_greet = random.choice(greetings) if greetings else random.choice(
        banks.get("starters", ["Hello!"])
    )
    agent_smalltalk = random.choice(
        banks.get("agent_smalltalk", ["Hello! How can I help you today?"])
    )

    return [
        {"from": "human", "value": human_greet},
        {
            "from": "gpt",
            "thoughts": "The user greeted me; respond warmly and offer help.",
            "actions": [],
            "value": agent_smalltalk,
        },
    ]


def create_closing_round(banks: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create a closing round:
      human: user closing phrase (banks.user_closing fallback to user_smalltalk)
      gpt  : agent closing phrase (banks.agent_closing fallback to agent_smalltalk)
    """
    user_phrases = banks.get(
        "user_closing", banks.get("user_smalltalk", ["That's all for now, thanks!"])
    )
    agent_phrases = banks.get(
        "agent_closing", banks.get("agent_smalltalk", ["Glad I could help! Have a great day."])
    )

    return [
        {"from": "human", "value": random.choice(user_phrases)},
        {
            "from": "gpt",
            "thoughts": "The user is closing; end politely and offer future help.",
            "actions": [],
            "value": random.choice(agent_phrases),
        },
    ]


def build_session(example: Dict[str, Any], greetings: List[str], banks: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build a complete multi-round session:
      1) Optional greeting round (80% chance)
      2) Original single-round conversation (as-is)
      3) Optional closing round (70% chance)
    """
    original_convo = example.get("conversations", [])
    conversation: List[Dict[str, Any]] = []

    # 1) Greeting (probabilistic)
    if random.random() < 0.8:
        conversation.extend(create_greeting_round(greetings, banks))

    # 2) Original content (single-round tool dialogue)
    conversation.extend(original_convo)

    # 3) Closing (probabilistic)
    if random.random() < 0.7:
        conversation.extend(create_closing_round(banks))

    # Preserve original metadata except for old conversation and image-only keys
    session: Dict[str, Any] = {}
    for k, v in example.items():
        if k not in {"conversations", "image_id", "file_name"}:
            session[k] = v

    session["conversations"] = conversation
    return session


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate multi-round dialogues by adding greeting/closing rounds to single-round tool conversations."
    )
    p.add_argument(
        "--input", nargs="+", required=True, help="Input JSONL file(s) or glob patterns with single-round conversations."
    )
    p.add_argument("--greetings", required=True, help="Text file of greeting phrases (one per line).")
    p.add_argument("--banks", required=True, help="YAML file with phrase banks (starters, smalltalk, closings).")
    p.add_argument("--output", required=True, help="Output JSONL file path.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    # Load inputs
    single_rounds: List[Dict[str, Any]] = []
    for pattern in args.input:
        for file_path in glob.glob(pattern):
            single_rounds.extend(load_jsonl(file_path))

    print(f"Loaded {len(single_rounds)} single-round conversations")

    greetings = load_lines(args.greetings)
    banks = load_banks(args.banks)

    # Generate sessions
    print("Generating multi-round sessions...")
    sessions: List[Dict[str, Any]] = []
    greeting_added = 0
    closing_added = 0

    for ex in single_rounds:
        s = build_session(ex, greetings, banks)
        sessions.append(s)

        # Simple stats: detect greeting/closing by turn structure
        conv = s["conversations"]
        orig_len = len(ex.get("conversations", []))

        # Greeting added if new conversation is longer AND first two turns look like greeting exchange
        if len(conv) > orig_len and len(conv) >= 2 and conv[0].get("from") == "human" and conv[1].get("from") == "gpt":
            greeting_added += 1

        # Closing added if last two turns look like a closing exchange
        if len(conv) >= 2 and conv[-2].get("from") == "human" and conv[-1].get("from") == "gpt":
            closing_added += 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    dump_jsonl(sessions, args.output)

    # Report
    print(f"Saved {len(sessions)} sessions to {args.output}")
    print("\nStatistics")
    print(f"  Original conversations : {len(single_rounds)}")
    print(f"  Generated sessions     : {len(sessions)}")
    print(f"  Added greeting rounds  : {greeting_added} ({greeting_added/len(sessions)*100:.1f}%)")
    print(f"  Added closing rounds   : {closing_added} ({closing_added/len(sessions)*100:.1f}%)")

    # Quick peek
    if sessions:
        sample = sessions[0]
        print("\nSample conversation preview")
        print(f"  Total turns: {len(sample['conversations'])}")
        print(f"  Keys      : {list(sample.keys())}")
        for i, turn in enumerate(sample["conversations"][:3]):
            snip = turn.get("value", "")
            if len(snip) > 64:
                snip = snip[:61] + "..."
            print(f"    Turn {i+1} [{turn.get('from','?')}]: {snip}")


if __name__ == "__main__":
    main()
