"""
Final-answer evaluation for MMedAgent-Plus (paper: "Final Answer Quality").

For every tool-use episode in the test set, the dialogue is replayed up to the
injected tool output, the agent generates its final response, and an OpenAI judge
scores faithfulness, completeness and conciseness (1-5) in three separate calls,
each with its own rubric and few-shot examples. Judge calls are cached by input hash.

Usage:
    export OPENAI_API_KEY=...
    python final_answer_eval.py \
        --test-jsonl $MMEDAGENT_DATA_ROOT/test/sm_sample100.jsonl \
        --base-dir   $MMEDAGENT_DATA_ROOT \
        --checkpoint /path/to/mmedagent-plus-7b \
        --out-dir    outputs/final_answer/sm
"""
import argparse
import os
import re
import json
import time
import hashlib
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from openai import OpenAI

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =============================
# Tool type registry
# Tools that produce artifacts (images/masks/files) rather than textual outputs. The agent cannot display these in text, so a
# short acknowledgment is always the correct response — the judge must not
# penalise for "missed_key_info" in these cases.
# =============================

ARTIFACT_TOOLS = {
    "HealthGPT",
    "HealthGPT-Reconstruction",
    "HealthGPT-SuperResolution",
    "UltraSAM",
    "MedSAM",
    "IterNet",
    "UniGradICON",
    "CellSAM",
    "CellViT",
    "DSMIL",
    "GroundingDino",
    "GroundingDino+MedSAM",
}

def is_artifact_tool(tool_name: str) -> bool:
    return tool_name in ARTIFACT_TOOLS


# =============================
# Utils
# =============================

def sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data

def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


TOOL_OUTPUT_RE = re.compile(
    r"^\s*([A-Za-z0-9_\-\+\.\s]+)\s+(output|result|returned)\s*:\s*",
    re.IGNORECASE
)

ANSWER_REQ_RE = re.compile(
    r"Answer my (first|previous)?\s*(request|question)\s*:\s*(.*)$",
    re.IGNORECASE | re.DOTALL
)

IMAGE_TOKEN_RE = re.compile(r"<image>")

# =============================
# Episode extraction
# =============================

@dataclass
class Episode:
    conv_id: int
    episode_id: int
    tool_name: str          # first tool in the action list (used for per-tool metrics grouping)
    history: List[Dict[str, Any]]   # raw conversation prefix, ending at tool-output human turn
    gold_turn: Dict[str, Any]       # next gpt turn (ground truth)
    gold_value: str
    image_paths: List[str]
    target_request: Optional[str] = None
    is_chain: bool = False          # True when actions form a sequential pipeline (depends_on present)
    terminal_tool_name: str = ""    # last tool in the chain; empty for single-tool or parallel episodes

def normalize_image_field(image_field) -> List[str]:
    if image_field is None:
        return []
    if isinstance(image_field, str):
        return [image_field]
    if isinstance(image_field, list):
        return image_field
    return []

def count_image_tokens(text: str) -> int:
    if not text:
        return 0
    return len(IMAGE_TOKEN_RE.findall(text))

def strip_image_tokens(text: str) -> str:
    if not text:
        return ""
    return IMAGE_TOKEN_RE.sub("", text).strip()

def resolve_image_paths(image_rel_paths: List[str], base_dir: str) -> List[str]:
    out = []
    for p in image_rel_paths:
        if not p:
            continue
        abs_path = p if os.path.isabs(p) else os.path.join(base_dir, p)
        if os.path.exists(abs_path):
            out.append(abs_path)
    return out

def extract_target_request(tool_output_text: str) -> Optional[str]:
    if not tool_output_text:
        return None
    m = ANSWER_REQ_RE.search(tool_output_text)
    if m:
        return m.group(3).strip()
    return None

def detect_chain(actions: List[Dict[str, Any]]) -> tuple:
    """
    Determine whether a multi-tool action list forms a sequential pipeline.

    A chain is defined as: at least one action carries a non-empty `depends_on`
    field, meaning it depends on the output of a prior tool in the same turn.

    Returns:
        (is_chain: bool, terminal_tool_name: str)

    terminal_tool_name is the last tool in dependency order — the one whose
    output is the final product of the pipeline. For parallel multi-tool
    episodes (no depends_on anywhere), terminal_tool_name is empty.

    Detection strategy: build a set of tools that are depended upon by others.
    The terminal tool is the one that appears in the action list but is NOT
    listed in any other action's depends_on — i.e. nothing comes after it.
    """
    if not actions or len(actions) < 2:
        return False, ""

    depended_upon: set = set()
    for action in actions:
        for dep in action.get("depends_on", []):
            depended_upon.add(dep)

    if not depended_upon:
        # No depends_on anywhere — parallel multi-tool, not a chain
        return False, ""

    # Terminal = in the action list but not depended upon by anything else
    all_names = [a.get("API_name", "") for a in actions]
    terminals = [name for name in all_names if name not in depended_upon]
    terminal = terminals[-1] if terminals else all_names[-1]
    return True, terminal


def extract_episodes_from_conversation(conv_data: Dict[str, Any], conv_id: int) -> List[Episode]:
    images = normalize_image_field(conv_data.get("image"))
    conversations = conv_data.get("conversations", [])

    episodes: List[Episode] = []
    ep_id = 0

    for i, turn in enumerate(conversations):
        if turn.get("from") != "gpt":
            continue

        actions = turn.get("actions", [])
        if not actions:
            continue

        # Next human turn should carry tool output
        next_human_idx = i + 1
        while next_human_idx < len(conversations) and conversations[next_human_idx].get("from") != "human":
            next_human_idx += 1
        if next_human_idx >= len(conversations):
            continue

        tool_output_turn = conversations[next_human_idx]
        tool_output_text = (tool_output_turn.get("value") or "").strip()

        # Next gpt turn after tool output is the gold final answer
        next_gpt_idx = next_human_idx + 1
        while next_gpt_idx < len(conversations) and conversations[next_gpt_idx].get("from") != "gpt":
            next_gpt_idx += 1
        if next_gpt_idx >= len(conversations):
            continue

        gold_turn = conversations[next_gpt_idx]
        gold_value = (gold_turn.get("value") or "").strip()

        tool_name = "UnknownTool"
        if isinstance(actions, list) and len(actions) > 0:
            tool_name = actions[0].get("API_name", "UnknownTool")

        is_chain, terminal_tool_name = detect_chain(actions)

        history = conversations[:next_human_idx + 1]
        target_request = extract_target_request(tool_output_text)

        episodes.append(Episode(
            conv_id=conv_id,
            episode_id=ep_id,
            tool_name=tool_name,
            history=history,
            gold_turn=gold_turn,
            gold_value=gold_value,
            image_paths=images,
            target_request=target_request,
            is_chain=is_chain,
            terminal_tool_name=terminal_tool_name,
        ))
        ep_id += 1

    return episodes


# =============================
# Raw history -> Qwen messages
# build_messages_from_history uses a cursor that advances by the number of
# <image> tokens in each human turn, so multi-image conversations are handled
# correctly.
# =============================

SYSTEM_PROMPT = (
    "The tool has already been executed and its output is shown in the conversation.\n"
    "Your task is to produce the final assistant reply to the user.\n\n"
    "CRITICAL RULES — you MUST follow ALL of these:\n"
    "- Do NOT call any tool. Do NOT output actions or tool invocations of any kind.\n"
    "- Do NOT express intent to use a tool (e.g. 'I will use X', 'Let me call X', "
    "'I'll utilize X tool to ...').\n"
    "- Do NOT output JSON action blocks, structured tool-call syntax, or any "
    "actions🚀 field.\n"
    "- The tool has already run. Your ONLY job is to synthesize its output into a "
    "concise, natural-language reply to the user.\n"
    "- If you find yourself about to write a tool name followed by parameters, STOP "
    "and instead write the final answer directly.\n"
    "- Provide a short, direct answer to the user based solely on the tool result shown."
)

def build_messages_from_history(
    history: List[Dict[str, Any]],
    all_image_paths: List[str],
) -> List[Dict[str, Any]]:
    """
    Convert raw conversation history into Qwen processor messages.

    - human turns  -> role=user
    - gpt turns    -> role=assistant (value field only, no thoughts/actions)
    - Each human turn that contains N <image> tokens consumes the next N
      images from the conversation-level image pool via a cursor.
    """
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        }
    ]

    image_cursor = 0

    for turn in history:
        speaker = turn.get("from")
        value = (turn.get("value") or "").strip()

        if speaker == "human":
            content = []
            n_img = count_image_tokens(value)
            for _ in range(n_img):
                if image_cursor < len(all_image_paths):
                    content.append({"type": "image", "image": all_image_paths[image_cursor]})
                    image_cursor += 1

            text = strip_image_tokens(value)
            if text:
                content.append({"type": "text", "text": text})

            if content:
                messages.append({"role": "user", "content": content})

        elif speaker == "gpt":
            # Use only the value field — thoughts and actions are training
            # scaffolding and should not be replayed to the model.
            assistant_text = (turn.get("value") or "").strip()
            if assistant_text:
                messages.append({
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_text}],
                })

    return messages


# =============================
# Parse generated next gpt turn
# =============================

def detect_redundant_tool_call(text: str) -> bool:
    """
    Return True if the model's raw output contains a non-empty actions field,
    indicating the model issued a tool call when it should have synthesised
    an answer (redundant tool call on the second / answer turn).
    """
    parsed = parse_model_output(text)
    actions_raw = parsed.get("actions_raw", "").strip()
    if not actions_raw or actions_raw in ("[]", "[ ]"):
        return False
    # Any non-trivial actions field counts as a redundant tool call
    try:
        import json
        actions = json.loads(actions_raw)
        return isinstance(actions, list) and len(actions) > 0
    except (ValueError, TypeError):
        # If it doesn't parse as JSON but is non-empty, treat as redundant
        return bool(actions_raw)

def parse_model_output(text: str) -> dict:
    """
    Parse the model's structured emoji-format output.

    The model emits:
        "thoughts🤔" <reasoning>
        "actions🚀"  [<action list>]
        "value👉"    <user-facing answer>

    Returns dict with keys 'thoughts', 'actions_raw', 'value' (all default "").
    """
    text = (text or "").strip()
    result = {"thoughts": "", "actions_raw": "", "value": ""}
    if not text:
        return result

    field_re = re.compile(
        r'"thoughts\U0001f914"\s*|"actions\U0001f680"\s*|"value\U0001f449"\s*',
        re.IGNORECASE,
    )
    splits = list(field_re.finditer(text))

    for idx, match in enumerate(splits):
        end = splits[idx + 1].start() if idx + 1 < len(splits) else len(text)
        field_text = text[match.end():end].strip()
        marker = match.group(0).strip().lower()

        if "thoughts" in marker:
            result["thoughts"] = field_text
        elif "actions" in marker:
            result["actions_raw"] = field_text
        elif "value" in marker:
            result["value"] = field_text

    return result


def _strip_structural_markers(text: str) -> str:
    """Remove action JSON noise from free-form text. Last-resort fallback."""
    text = re.sub(r'\{"API_name".*?\}', "", text, flags=re.DOTALL)
    text = re.sub(r"^\s*[\[\]]\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_final_value(text: str) -> str:
    """
    Extract the user-facing answer from the model's raw output.

    Rule: find the literal string '"value\U0001f449"' and return EVERYTHING
    after it. This is simpler and more robust than any regex approach — the
    marker is unique in the model's output format and the entire remaining
    text is the answer, including multi-line content.

    Fallback chain for outputs without the emoji marker:
      1. Plain "value:" style patterns
      2. Full remaining prose after stripping structural noise
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Primary: find the exact emoji marker and take all content after it
    marker = '"value\U0001f449"'
    idx = text.find(marker)
    if idx != -1:
        return text[idx + len(marker):].strip()

    # Fallback 1: plain key-value patterns (no emoji)
    plain_patterns = [
        r'"value"\s*[:]\s*(.*)',
        r'value\s*[:]\s*(.*)',
    ]
    for p in plain_patterns:
        m = re.search(p, text, re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Fallback 2: strip structural noise and return all remaining prose
    return _strip_structural_markers(text) or text




# =============================
# Local model runner (single canonical implementation)
# =============================

class LocalDialogueContinuationModel:
    def __init__(self, model, processor, base_dir: str):
        self.model = model
        self.processor = processor
        self.base_dir = base_dir

    def generate_next_gpt_turn(
        self,
        history: List[Dict[str, Any]],
        image_rel_paths: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> str:
        image_paths = resolve_image_paths(image_rel_paths, self.base_dir)
        messages = build_messages_from_history(history, image_paths)

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, **gen_kwargs)

        response_ids = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        response = self.processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        return response.strip()




# =============================
# Judge — three separate criteria prompts
# =============================

# Shared context block injected into every judge call.
# Gold answer is provided as FORMAT REFERENCE ONLY — the agent's content
# may legitimately differ from gold because tool outputs vary.

_JUDGE_CONTEXT_HEADER = """\
You are evaluating a single scoring criterion for a medical multimodal agent.

The agent's task: after tools have executed and returned output, produce a
final natural-language reply to the user. The agent is an ORCHESTRATOR —
it organizes and phrases the tool's output for the user. It is NOT responsible
for verifying clinical correctness of tool outputs. Never penalise the agent
for a tool returning wrong or low-quality output.

You will be given:
  TOOL_NAME          — the tool that ran
  IS_ARTIFACT_TOOL   — True if the tool produces an image/mask/file (not text)
  IS_CHAIN           — True if multiple tools ran sequentially (pipeline)
  TERMINAL_TOOL      — last tool in the chain (only evaluate its output for chains)
  ORIGINAL REQUEST   — what the user asked
  TOOL OUTPUT        — what the tool returned (last human turn)
  AGENT ANSWER       — what the agent said
  GOLD ANSWER        — the reference answer; its role differs by criterion:
                       • FAITHFULNESS: NOT used — only tool output matters.
                       • COMPLETENESS: reference for what information from the
                         tool output is important to include.
                       • CONCISENESS: reference for the appropriate scope —
                         what is necessary vs. unnecessary to include.

Artifact tool rules (IS_ARTIFACT_TOOL=True):
  The agent cannot see or describe the artifact. A brief acknowledgment that
  the result was generated IS the correct and complete response. Never deduct
  for not describing image content, filename, or details.

Chain rules (IS_CHAIN=True):
  Evaluate only the terminal tool's output. Intermediate outputs are internal
  pipeline steps — the agent is not expected to present them.

Output format (rationale MUST come before the score):
{
  "rationale": "<your step-by-step reasoning for this criterion>",
  "score": <integer>
}
Return ONLY valid JSON with these two fields. No other fields.\
"""
 
# ── Faithfulness ──────────────────────────────────────────────────────────

FAITHFULNESS_JUDGE_INSTRUCTIONS = _JUDGE_CONTEXT_HEADER + """

== CRITERION: FAITHFULNESS (1-5) ==

Definition: Faithfulness measures whether the agent's answer accurately
represents what the tool returned, without inventing or contradicting the
tool output. The agent is only an orchestrator — it phrases the tool result
for the user. Evaluate ONLY against the tool output; the gold answer is NOT
used for this criterion.

Key principles:
- A claim is faithful if it is directly traceable to the tool output.
- Paraphrasing is acceptable as long as the meaning is preserved.
- For artifact tools (IS_ARTIFACT_TOOL=True): a brief acknowledgment that
  the result was generated is perfectly faithful (score 5).
- Do NOT penalise the agent for a tool returning incorrect or limited output.
  Evaluate faithfulness to the tool output AS-IS.

Scoring rubric:
  5 — Every claim is directly traceable to the tool output. Nothing invented,
      nothing contradicted. Perfect representation of what the tool returned.
  4 — Minor imprecision in phrasing (e.g. slight paraphrase that shifts
      emphasis) but no fabricated or contradicted claims.
  3 — Some claims go modestly beyond the tool output, or a portion of the
      tool output is inaccurately paraphrased without outright contradiction.
  2 — Notable fabrication: claims clearly absent from the tool output, or
      significant distortion of what the tool returned.
  1 — Response mostly fabricates or directly contradicts the tool output;
      only superficially touches what the tool returned, or response is empty.

== FEW-SHOT EXAMPLES ==

--- Example A (score 5: perfectly faithful) ---
TOOL_NAME: CONCH | IS_ARTIFACT_TOOL: False | IS_CHAIN: False
ORIGINAL REQUEST: Is this pathology slide consistent with benign or malignant tissue?
TOOL OUTPUT: CONCH output: {benign}
AGENT ANSWER: The tissue classification from the pathology analysis is benign.
GOLD ANSWER: The tissue classification is benign.

{
  "rationale": "The tool returned a single unambiguous classification: benign. The agent's response states exactly that — benign — with no additions, no invented detail, and no contradiction. The phrasing is a natural restatement of the tool output. Every word is grounded in what the tool returned. The gold answer is not consulted for faithfulness. Score 5.",
  "score": 5
}

--- Example B (score 3: partial faithfulness with inaccurate paraphrase) ---
TOOL_NAME: LLaVA-Rad | IS_ARTIFACT_TOOL: False | IS_CHAIN: False
ORIGINAL REQUEST: Describe any abnormalities in this chest X-ray.
TOOL OUTPUT: LLaVA-Rad output: Mild bilateral pleural effusion identified.
AGENT ANSWER: The chest X-ray shows a moderate bilateral pleural effusion with some consolidation noted in the lower lobes.
GOLD ANSWER: There is mild bilateral pleural effusion.

{
  "rationale": "The tool reported 'mild bilateral pleural effusion.' The agent's response makes two errors: (1) it upgrades the severity from 'mild' to 'moderate', which is a meaningful clinical distinction — this is an inaccurate paraphrase that shifts the meaning without outright inventing a new finding; (2) it adds 'consolidation in the lower lobes', which appears nowhere in the tool output and is a fabricated finding. The response is not entirely unfaithful — it correctly identifies bilateral pleural effusion — but the severity distortion and the fabricated consolidation together represent a moderate faithfulness failure. Score 3.",
  "score": 3
}

--- Example C (score 1: direct contradiction and fabrication) ---
TOOL_NAME: CONCH | IS_ARTIFACT_TOOL: False | IS_CHAIN: False
ORIGINAL REQUEST: Is this pathology slide consistent with benign or malignant tissue?
TOOL OUTPUT: CONCH output: {benign}
AGENT ANSWER: Based on the pathological analysis, the slide shows evidence of high-grade dysplasia with nuclear pleomorphism and increased mitotic activity, suggesting malignant transformation.
GOLD ANSWER: The tissue classification is benign.

{
  "rationale": "The tool returned a single unambiguous classification: benign. The agent claims the opposite — malignant transformation — and fabricates histological details (nuclear pleomorphism, increased mitotic activity) that appear nowhere in the tool output. This is the most serious faithfulness failure: not just omitting information, but actively contradicting the tool's finding and inventing clinical evidence to support the wrong conclusion. The surface polish of the prose is irrelevant. Score 1.",
  "score": 1
}
"""
 
# ── Completeness ──────────────────────────────────────────────────────────

COMPLETENESS_JUDGE_INSTRUCTIONS = _JUDGE_CONTEXT_HEADER + """

== CRITERION: COMPLETENESS (1-5) ==

Definition: Completeness measures whether the agent's answer covers all the
important information from the tool output that is relevant to the user's
request. Use the gold answer as a reference to identify which parts of the
tool output are important — the gold shows what an ideal answer highlights.

Key principles:
- The ceiling is set by the tool output: if the tool did not return certain
  information, the agent cannot be penalised for not including it.
- Use the gold answer to identify which aspects of the tool output are important
  to include. If the gold highlights certain findings, check whether the agent
  covered them.
- For artifact tools (IS_ARTIFACT_TOOL=True): acknowledging the artifact was
  generated IS complete — no further detail is expected (score 5).
- For chain episodes: evaluate only the terminal tool's output.
- Do NOT penalise the agent for not going beyond what the tool provided.

Scoring rubric:
  5 — All important information from the tool output is present. Nothing
      the gold answer would highlight is missing.
  4 — Nearly complete. One minor piece of important information is absent,
      but the core findings are all covered.
  3 — Partially complete. A clearly important finding or result that the gold
      answer highlights is absent from the agent's response.
  2 — Incomplete. Multiple important findings are missing, or the single most
      critical result the gold highlights is not mentioned.
  1 — Barely addresses the tool output. Almost all important information that
      the gold would include is absent, or the response is empty/off-topic.

== FEW-SHOT EXAMPLES ==

--- Example A (score 5: all important findings covered) ---
TOOL_NAME: LLaVA-Rad | IS_ARTIFACT_TOOL: False | IS_CHAIN: False
ORIGINAL REQUEST: Interpret this chest X-ray and report all findings.
TOOL OUTPUT: LLaVA-Rad output: Small right pleural effusion present. Mild cardiomegaly. Lungs otherwise clear.
AGENT ANSWER: The chest X-ray reveals a small right pleural effusion and mild cardiomegaly. The remaining lung fields are otherwise clear.
GOLD ANSWER: The chest X-ray shows a small right pleural effusion and mild cardiomegaly. The remaining lung fields are clear.

{
  "rationale": "The tool returned three findings: (1) small right pleural effusion, (2) mild cardiomegaly, (3) lungs otherwise clear. The gold answer highlights all three, with emphasis on the first two as the primary abnormal findings. The agent covers all three findings in the same order of importance that the gold establishes. Nothing the gold would highlight is missing. Score 5.",
  "score": 5
}

--- Example B (score 3: one important finding absent) ---
TOOL_NAME: LLaVA-Rad | IS_ARTIFACT_TOOL: False | IS_CHAIN: False
ORIGINAL REQUEST: Interpret this chest X-ray and report all findings.
TOOL OUTPUT: LLaVA-Rad output: Small right pleural effusion present. Mild cardiomegaly. Lungs otherwise clear.
AGENT ANSWER: There is a small right pleural effusion. The lungs are otherwise clear.
GOLD ANSWER: The chest X-ray shows a small right pleural effusion and mild cardiomegaly. The remaining lung fields are clear.

{
  "rationale": "The tool returned three findings. The gold answer highlights all three, treating cardiomegaly as a co-equal primary finding alongside the pleural effusion. The agent mentions the pleural effusion and lung clearance but omits cardiomegaly entirely. The omitted finding is clearly important — it is explicitly highlighted by the gold answer as one of only two abnormal results. The response is not entirely empty; it covers some relevant content. But the absence of one significant highlighted finding places this at score 3. Score 3.",
  "score": 3
}

--- Example C (score 1: almost all important information absent) ---
TOOL_NAME: LLaVA-Rad | IS_ARTIFACT_TOOL: False | IS_CHAIN: False
ORIGINAL REQUEST: Interpret this chest X-ray and report all findings.
TOOL OUTPUT: LLaVA-Rad output: Small right pleural effusion present. Mild cardiomegaly. Lungs otherwise clear.
AGENT ANSWER: The lungs appear clear.
GOLD ANSWER: The chest X-ray shows a small right pleural effusion and mild cardiomegaly. The remaining lung fields are clear.

{
  "rationale": "The tool returned three findings; the gold highlights two abnormal findings (pleural effusion and cardiomegaly) as the most important results. The agent's response mentions only that lungs appear clear — the least important of the three findings — and omits both primary abnormalities that the gold prioritizes. The agent selects the one finding that requires no clinical attention and ignores the two that are clinically significant. This is a near-total completeness failure. Score 1.",
  "score": 1
}
"""
 
# ── Conciseness ───────────────────────────────────────────────────────────

CONCISENESS_JUDGE_INSTRUCTIONS = _JUDGE_CONTEXT_HEADER + """

== CRITERION: CONCISENESS (1-5) ==

Definition: Conciseness measures whether the agent's answer avoids including
irrelevant, redundant, or unnecessary content relative to what the user asked
and what the tool returned. Use the gold answer as a reference for the
appropriate scope — it shows what is necessary to include and what is not.

Key principles:
- Penalise for content that goes beyond the scope of the tool output or the
  user's request: fabricated context, unsolicited clinical advice, excessive
  disclaimers, verbose preambles, or filler sentences that add no value.
- Penalise for pasting raw tool output verbatim with its prefix label
  (e.g. "CONCH output: {benign}" copied as-is).
- Do NOT penalise for being slightly more or less wordy than gold — minor
  phrasing differences are acceptable.
- Do NOT penalise for content that is relevant and grounded in the tool output,
  even if the gold does not mention it.
- For artifact tools (IS_ARTIFACT_TOOL=True): a brief acknowledgment is ideal;
  padding or over-description of the artifact is unnecessary.

Scoring rubric:
  5 — Response is focused and on-point. No irrelevant content, unnecessary
      padding, or unsolicited additions.
  4 — Mostly concise. One minor instance of unnecessary phrasing or a small
      unsolicited remark that does not distract from the core answer.
  3 — Noticeable padding or one significant irrelevant addition (e.g. an
      unsolicited clinical recommendation, a lengthy disclaimer, raw tool
      prefix label included, or a verbose preamble before the actual answer).
  2 — Significant irrelevant content — multiple unnecessary additions, or the
      response is substantially longer than needed with little extra value.
  1 — Response is dominated by irrelevant, redundant, or off-topic content.
      The core answer is buried, entirely absent, or the response is empty.

== FEW-SHOT EXAMPLES ==

--- Example A (score 5: perfectly focused) ---
TOOL_NAME: CONCH | IS_ARTIFACT_TOOL: False | IS_CHAIN: False
ORIGINAL REQUEST: What does the pathology slide show?
TOOL OUTPUT: CONCH output: {adenocarcinoma}
AGENT ANSWER: The pathology slide is consistent with adenocarcinoma.
GOLD ANSWER: The pathology slide is consistent with adenocarcinoma.

{
  "rationale": "The gold answer is a single sentence stating the classification. The agent's response is exactly one sentence that directly delivers the finding from the tool output. There is no preamble, no disclaimer, no unsolicited recommendation, and no padding of any kind. The scope matches what the tool returned and what the gold confirms is necessary. Score 5.",
  "score": 5
}

--- Example B (score 3: one significant unsolicited addition) ---
TOOL_NAME: CONCH | IS_ARTIFACT_TOOL: False | IS_CHAIN: False
ORIGINAL REQUEST: What does the pathology slide show?
TOOL OUTPUT: CONCH output: {adenocarcinoma}
AGENT ANSWER: The pathology slide shows adenocarcinoma. For an accurate diagnosis, it is recommended to confirm these findings with a board-certified pathologist before making any clinical decisions.
GOLD ANSWER: The pathology slide is consistent with adenocarcinoma.

{
  "rationale": "The gold answer is a single sentence. The agent correctly delivers the finding in the first sentence, but then appends an unsolicited clinical recommendation about consulting a pathologist. This addition was not present in the tool output and was not requested by the user. While the core answer is present and easy to identify, the appended sentence is a meaningful unnecessary addition that goes beyond the appropriate scope. Score 3.",
  "score": 3
}

--- Example C (score 1: core answer buried in irrelevant content) ---
TOOL_NAME: CONCH | IS_ARTIFACT_TOOL: False | IS_CHAIN: False
ORIGINAL REQUEST: What does the pathology slide show?
TOOL OUTPUT: CONCH output: {adenocarcinoma}
AGENT ANSWER: Thank you for sharing this pathology slide. Pathological analysis is a critical component of modern oncology workflows. Based on the AI-assisted review, the slide indicates adenocarcinoma. However, please note that AI tools are not a substitute for expert review. A board-certified pathologist should review these findings. Further molecular testing including EGFR, ALK, ROS1, and PD-L1 status may be required depending on the treatment protocol. Early-stage diagnosis often benefits from multidisciplinary tumor board discussion.
GOLD ANSWER: The pathology slide is consistent with adenocarcinoma.

{
  "rationale": "The gold answer is a single sentence. The agent's response is seven sentences long, of which only one (sentence 3) contains the actual finding from the tool. The remaining six sentences consist of: a conversational opener, a general statement about oncology, two disclaimers about AI limitations, a recommendation for molecular testing (EGFR, ALK, ROS1, PD-L1), and advice about tumor board discussions. None of this appears in the tool output or was requested by the user. The core answer exists but is buried deep in a wall of irrelevant content. Score 1.",
  "score": 1
}
"""



def history_to_text(history: List[Dict[str, Any]]) -> str:
    chunks = []
    for t in history:
        speaker = t.get("from", "").upper()
        value = (t.get("value") or "").strip()
        chunks.append(f"{speaker}:\n{value}")
    return "\n\n".join(chunks)

def get_last_human_turn(history: List[Dict[str, Any]]) -> str:
    for t in reversed(history):
        if t.get("from") == "human":
            return (t.get("value") or "").strip()
    return ""

def build_judge_input(
    history: List[Dict[str, Any]],
    model_final_answer: str,
    tool_name: str,
    target_request: Optional[str],
    gold_value: str = "",
    is_chain: bool = False,
    terminal_tool_name: str = "",
) -> str:
    """Build the shared payload sent to every criterion judge call."""
    artifact_flag = "True" if is_artifact_tool(tool_name) else "False"
    request_line = target_request or "(infer from conversation above)"

    if is_chain and terminal_tool_name:
        chain_line = f"IS_CHAIN: True\nTERMINAL_TOOL: {terminal_tool_name}\n"
    else:
        chain_line = "IS_CHAIN: False\n"

    return (
        f"TOOL_NAME: {tool_name}\n"
        f"IS_ARTIFACT_TOOL: {artifact_flag}\n"
        + chain_line
        + f"\nORIGINAL REQUEST:\n{request_line}\n\n"
        "TOOL OUTPUT (last human turn):\n"
        f"{get_last_human_turn(history)}\n\n"
        "AGENT ANSWER:\n"
        f"{model_final_answer}\n\n"
        "GOLD ANSWER (see criterion instructions for how to use this):\n"
        f"{gold_value or '(not available)'}\n\n"
        "Now evaluate the criterion and produce the JSON."
    )


class GPTJudge:
    def __init__(
        self,
        model_name: str = "gpt-5.4-mini",
        cache_path: Optional[str] = None,
        max_retries: int = 5,
        retry_backoff: float = 1.8,
        api_key: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_path = cache_path
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        self.cache: Dict[str, Any] = {}
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)

    def _save_cache(self):
        if not self.cache_path:
            return
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.cache_path)

    def _call_criterion(
        self,
        instructions: str,
        payload: str,
        criterion_name: str,
    ) -> Dict[str, Any]:
        """
        Make one API call for a single criterion. Returns dict with
        'score' (int 0-5) and 'rationale' (str).
        Each call is cached independently.
        """
        key = sha1_text(self.model_name + "\n" + criterion_name + "\n"
                        + instructions + "\n" + payload)

        if key in self.cache:
            return self.cache[key]

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": instructions},
                        {"role": "user",   "content": payload},
                    ],
                )
                raw = resp.choices[0].message.content.strip()

                parsed = None
                try:
                    parsed = json.loads(raw)
                except Exception:
                    m = re.search(r"\{.*\}", raw, re.DOTALL)
                    if m:
                        parsed = json.loads(m.group(0))

                if not isinstance(parsed, dict) or "score" not in parsed:
                    raise ValueError(
                        f"[{criterion_name}] invalid JSON: {raw[:300]}"
                    )

                result = {
                    "score":    int(parsed.get("score", 0)),
                    "rationale": str(parsed.get("rationale", "")),
                }
                self.cache[key] = result
                self._save_cache()
                return result

            except Exception as e:
                last_err = repr(e)
                print(
                    f"[Judge/{criterion_name} error]"
                    f"[attempt {attempt+1}/{self.max_retries}] {last_err}"
                )
                time.sleep(
                    (self.retry_backoff ** attempt) + random.random() * 0.2
                )

        return {
            "score": 1,
            "rationale": f"Judge failed after retries: {last_err}",
        }

    def judge(
        self,
        history: List[Dict[str, Any]],
        model_final_answer: str,
        tool_name: str = "UnknownTool",
        target_request: Optional[str] = None,
        gold_value: str = "",
        is_chain: bool = False,
        terminal_tool_name: str = "",
    ) -> Dict[str, Any]:
        """
        Run three independent criterion evaluations and combine results.

        Each criterion is evaluated in a separate API call with its own
        dedicated prompt and few-shot examples. Scores are kept separate
        (not summed). A combined total is also
        stored for sorting/filtering convenience.

        Returns dict with:
          faithfulness          0-5
          faithfulness_rationale
          completeness          0-5
          completeness_rationale
          conciseness           0-5
          conciseness_rationale
          error_tags            list[str]
          score                 sum of three (3-15), for sorting only
        """
        payload = build_judge_input(
            history, model_final_answer, tool_name, target_request,
            gold_value=gold_value,
            is_chain=is_chain,
            terminal_tool_name=terminal_tool_name,
        )

        f  = self._call_criterion(FAITHFULNESS_JUDGE_INSTRUCTIONS,
                                  payload, "faithfulness")
        cp = self._call_criterion(COMPLETENESS_JUDGE_INSTRUCTIONS,
                                  payload, "completeness")
        cn = self._call_criterion(CONCISENESS_JUDGE_INSTRUCTIONS,
                                  payload, "conciseness")

        # Aggregate error tags heuristically from low sub-scores.
        error_tags: List[str] = []
        if f["score"] <= 2:
            error_tags.append("hallucination_or_ignored_output")
        if cp["score"] <= 2:
            error_tags.append("missing_key_info")
        if cn["score"] <= 2:
            error_tags.append("irrelevant_or_verbose")

        return {
            "faithfulness":           f["score"],
            "faithfulness_rationale": f["rationale"],
            "completeness":           cp["score"],
            "completeness_rationale": cp["rationale"],
            "conciseness":            cn["score"],
            "conciseness_rationale":  cn["rationale"],
            "error_tags":                 error_tags,
            # Convenience total (3-15) — not the primary metric
            "score": f["score"] + cp["score"] + cn["score"],
        }



# =============================
# Evaluator
# =============================

class FinalAnswerEvaluator:
    def __init__(self, model_runner: LocalDialogueContinuationModel, judge: GPTJudge):
        self.model_runner = model_runner
        self.judge = judge

    def evaluate_dataset(
        self,
        test_jsonl: str,
        out_dir: str,
        max_samples: Optional[int] = None,
        max_episodes_per_conv: Optional[int] = None,
        gen_max_new_tokens: int = 256,
        gen_temperature: float = 0.0,
        gen_top_p: float = 1.0,
    ) -> Dict[str, Any]:
        safe_mkdir(out_dir)

        data = load_jsonl(test_jsonl)
        print("Loaded conversations:", len(data))
        if max_samples is not None:
            data = data[:max_samples]

        all_episode_rows = []
        episode_scores = []
        faithfulness_scores: List[int] = []
        completeness_scores: List[int] = []
        conciseness_scores: List[int] = []
        conv_to_episode_scores = defaultdict(list)
        error_tag_counts = defaultdict(int)
        tool_score_sum = defaultdict(float)
        tool_score_cnt = defaultdict(int)

        total_extracted_eps = 0

        for conv_id, conv in enumerate(tqdm(data, desc="Extracting + evaluating")):
            episodes = extract_episodes_from_conversation(conv, conv_id)
            print(f"conv {conv_id} -> episodes before truncation: {len(episodes)}")

            if max_episodes_per_conv is not None:
                episodes = episodes[:max_episodes_per_conv]

            total_extracted_eps += len(episodes)

            for ep in episodes:
                model_answer_raw = self.model_runner.generate_next_gpt_turn(
                    history=ep.history,
                    image_rel_paths=ep.image_paths,
                    max_new_tokens=gen_max_new_tokens,
                    temperature=gen_temperature,
                    top_p=gen_top_p,
                )

                model_final_answer = extract_final_value(model_answer_raw)

                # Detect whether the model reproduced the tool-calling pattern
                # instead of giving a final answer.
                #
                # The dominant failure mode (75% of low-scoring cases) is:
                # the model treats the evaluation context as if it were the
                # first turn — it emits a non-empty actions list and a value
                # like "I'll use ToolX to proceed." rather than synthesising
                # the tool output into a real answer. This is a model training
                # issue (context confusion), not a scoring artefact. The judge
                # will correctly score these responses near 0 on faithfulness
                # and completeness, so no scoring adjustment is needed.
                #
                # We distinguish two sub-cases to aid training-side analysis:
                #   "wrong_context_response" — model reproduced the first-turn
                #     tool-calling pattern verbatim; value is a tool-invocation
                #     acknowledgment, not an answer.
                #   "redundant_tool_call"    — model called a tool but still
                #     produced a substantive value field (rarer, milder issue).
                parsed_out = parse_model_output(model_answer_raw)
                redundant_tool_call = detect_redundant_tool_call(model_answer_raw)

                # Detect whether the value field is a tool-invocation intent
                # statement rather than a real answer to the user.
                #
                # Expanded pattern covers cases the original regex missed:
                #   "Certainly! I will utilize..."  (preamble + will)
                #   "Sure! Using the provided..."   (preamble + gerund)
                #   "Of course, I'll use..."        (preamble + contraction)
                # The key signal is any form of future-tense tool invocation,
                # optionally preceded by a politeness preamble.
                TOOL_INVOCATION_RE = re.compile(
                    r"^(?:(?:certainly|sure|of course|absolutely)[!,.]?\s*)?"
                    r"(?:i(?:'ll| will| am going to| would)[\s\w]*?(?:use|utilize|apply|employ|leverage)|"
                    r"(?:let me|using|by employing|by using|by applying))",
                    re.IGNORECASE,
                )
                value_text = parsed_out["value"].strip()
                value_is_invocation = bool(TOOL_INVOCATION_RE.match(value_text))
                wrong_context_response = redundant_tool_call and value_is_invocation

                # === Deterministic short-circuit for behavior failures ===
                #
                # When wrong_context_response is detected, the model gave no
                # real answer — it reproduced the first-turn tool-calling pattern.
                # Sending these to the judge causes inconsistent scores: two
                # nearly identical non-answers received 2 vs 6 because the judge
                # had no reliable basis to differentiate them.
                #
                # Fix: skip the judge for these cases and assign a fixed score.
                # Score=1 rather than 0 because the model at least produces a
                # coherent English sentence (conciseness=1).
                #
                # The judge is still called for redundant_tool_call cases where
                # the value contains a substantive answer — those need content
                # quality evaluation.
                if wrong_context_response:
                    # Deterministic score — skip judge entirely.
                    # Separate rationale stored under each criterion key for
                    # consistency with the new three-call judge schema.
                    _wc_rationale = (
                        "Deterministic: model reproduced first-turn tool-calling "
                        "pattern instead of answering the user. Judge skipped."
                    )
                    judge_result = {
                        "faithfulness":           1,
                        "faithfulness_rationale": _wc_rationale,
                        "completeness":           1,
                        "completeness_rationale": _wc_rationale,
                        "conciseness":            1,
                        "conciseness_rationale": (
                            "Model reproduced the tool-calling pattern instead of "
                            "answering the user; no meaningful user-facing content "
                            "to evaluate."
                        ),
                        "error_tags": ["wrong_context_response"],
                        "score": 3,
                    }
                else:
                    judge_result = self.judge.judge(
                        history=ep.history,
                        model_final_answer=model_final_answer,
                        tool_name=ep.tool_name,
                        target_request=ep.target_request,
                        gold_value=ep.gold_value,
                        is_chain=ep.is_chain,
                        terminal_tool_name=ep.terminal_tool_name,
                    )
                    tags = judge_result.setdefault("error_tags", [])
                    if redundant_tool_call and "redundant_tool_call" not in tags:
                        tags.append("redundant_tool_call")

                score = int(judge_result.get("score", 0))
                episode_scores.append(score)
                conv_to_episode_scores[ep.conv_id].append(score)
                faithfulness_scores.append(int(judge_result.get("faithfulness", 0)))
                completeness_scores.append(int(judge_result.get("completeness", 0)))
                conciseness_scores.append(int(judge_result.get("conciseness", 0)))

                for tag in judge_result.get("error_tags", []) or []:
                    error_tag_counts[tag] += 1

                tool_score_sum[ep.tool_name] += score
                tool_score_cnt[ep.tool_name] += 1

                all_episode_rows.append({
                    "conv_id": ep.conv_id,
                    "episode_id": ep.episode_id,
                    "tool_name": ep.tool_name,
                    "is_artifact_tool": is_artifact_tool(ep.tool_name),
                    "is_chain": ep.is_chain,
                    "terminal_tool_name": ep.terminal_tool_name,
                    "target_request": ep.target_request,
                    "image_paths": ep.image_paths,
                    "history": ep.history,
                    "gold_turn": ep.gold_turn,
                    "gold_value": ep.gold_value,
                    "model_answer_raw": model_answer_raw,
                    "model_final_answer": model_final_answer,
                    "redundant_tool_call": redundant_tool_call,
                    "wrong_context_response": wrong_context_response,
                    "judge": judge_result,
                })

        print("TOTAL evaluated episodes:", total_extracted_eps)

        write_jsonl(os.path.join(out_dir, "episode_results.jsonl"), all_episode_rows)

        metrics: Dict[str, Any] = {}
        if episode_scores:
            n = len(episode_scores)
            metrics["num_episodes"] = n
            # Combined score (3-15) — convenience metric for sorting/filtering
            metrics["avg_episode_score"] = sum(episode_scores) / n
            # Per-criterion averages (each 1-5) — primary metrics
            metrics["avg_faithfulness"]  = sum(faithfulness_scores) / n
            metrics["avg_completeness"]  = sum(completeness_scores) / n
            metrics["avg_conciseness"]   = sum(conciseness_scores) / n
            # Pass rates on the combined 3-15 scale
            metrics["episode_pass_rate_ge_12"] = sum(s >= 12 for s in episode_scores) / n
            metrics["episode_pass_rate_ge_10"] = sum(s >= 10 for s in episode_scores) / n
            # Behavior failure rates
            metrics["wrong_context_response_rate"]   = error_tag_counts.get("wrong_context_response", 0) / n
            metrics["redundant_tool_call_rate"]      = error_tag_counts.get("redundant_tool_call", 0) / n
            metrics["hallucination_or_ignored_rate"] = error_tag_counts.get("hallucination_or_ignored_output", 0) / n
            metrics["missing_key_info_rate"]         = error_tag_counts.get("missing_key_info", 0) / n
            metrics["irrelevant_or_verbose_rate"]    = error_tag_counts.get("irrelevant_or_verbose", 0) / n
        else:
            metrics.update({
                "num_episodes": 0,
                "avg_episode_score": 0.0,
                "avg_faithfulness": 0.0,
                "avg_completeness": 0.0,
                "avg_conciseness": 0.0,
                "episode_pass_rate_ge_12": 0.0,
                "episode_pass_rate_ge_10": 0.0,
                "wrong_context_response_rate": 0.0,
                "redundant_tool_call_rate": 0.0,
                "hallucination_or_ignored_rate": 0.0,
                "missing_key_info_rate": 0.0,
                "irrelevant_or_verbose_rate": 0.0,
            })

        # Conversation-level metrics
        conv_scores = {}
        for cid, scores in conv_to_episode_scores.items():
            if scores:
                conv_scores[cid] = {
                    "num_episodes": len(scores),
                    "avg_score": sum(scores) / len(scores),
                    "min_score": min(scores),
                    "all_pass_ge_12": all(s >= 12 for s in scores),
                    "all_pass_ge_10": all(s >= 10 for s in scores),
                }

        if conv_scores:
            metrics["num_conversations_with_episodes"] = len(conv_scores)
            metrics["avg_conversation_score"] = sum(v["avg_score"] for v in conv_scores.values()) / len(conv_scores)
            metrics["avg_worst_turn_score"] = sum(v["min_score"] for v in conv_scores.values()) / len(conv_scores)
            metrics["whole_case_pass_rate_ge_12"] = sum(v["all_pass_ge_12"] for v in conv_scores.values()) / len(conv_scores)
            metrics["whole_case_pass_rate_ge_10"] = sum(v["all_pass_ge_10"] for v in conv_scores.values()) / len(conv_scores)
        else:
            metrics.update({
                "num_conversations_with_episodes": 0,
                "avg_conversation_score": 0.0,
                "avg_worst_turn_score": 0.0,
                "whole_case_pass_rate_ge_12": 0.0,
                "whole_case_pass_rate_ge_10": 0.0,
            })

        # Per-tool metrics — split artifact vs text tools for fairer comparison
        per_tool: Dict[str, Any] = {}
        for tool, cnt in tool_score_cnt.items():
            per_tool[tool] = {
                "count": cnt,
                "avg_score": tool_score_sum[tool] / max(cnt, 1),
                "is_artifact_tool": is_artifact_tool(tool),
            }

        with open(os.path.join(out_dir, "conversation_summary.json"), "w", encoding="utf-8") as f:
            json.dump(conv_scores, f, ensure_ascii=False, indent=2)

        summary = {
            "metrics": metrics,
            "error_tag_counts": dict(error_tag_counts),
            "per_tool": per_tool,
        }

        with open(os.path.join(out_dir, "summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        worst = sorted(all_episode_rows, key=lambda r: int(r["judge"].get("score", 0)))[:200]
        write_jsonl(os.path.join(out_dir, "worst_episodes.jsonl"), worst)

        return summary


# =============================
# Main
# =============================

def parse_args():
    p = argparse.ArgumentParser(description="LLM-as-a-judge evaluation of final answers.")
    p.add_argument("--test-jsonl", required=True, help="Test split JSONL.")
    p.add_argument("--base-dir", required=True, help="Directory that image paths in the test file are relative to.")
    p.add_argument("--checkpoint", required=True, help="Fine-tuned model directory or HF repo ID.")
    p.add_argument("--processor-dir", default=None, help="Processor directory (defaults to --checkpoint).")
    p.add_argument("--out-dir", default="outputs/final_answer")
    p.add_argument("--judge-model", default="gpt-5.4-mini", help="OpenAI model used as the judge.")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    TEST_JSONL = args.test_jsonl
    BASE_DIR = args.base_dir
    OUT_DIR = args.out_dir
    JUDGE_CACHE = os.path.join(OUT_DIR, "judge_cache.json")

    checkpoint_path = args.checkpoint
    processor_dir = args.processor_dir or args.checkpoint

    safe_mkdir(OUT_DIR)

    print("Loading local Qwen2.5-VL model...")
    from transformers import Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(processor_dir)

    if not getattr(processor, "chat_template", None):
        raise ValueError(
            f"Processor at {processor_dir} is missing chat_template."
        )

    runner = LocalDialogueContinuationModel(
        model=model,
        processor=processor,
        base_dir=BASE_DIR,
    )

    judge = GPTJudge(
        model_name=args.judge_model,
        cache_path=JUDGE_CACHE,
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

    evaluator = FinalAnswerEvaluator(model_runner=runner, judge=judge)

    summary = evaluator.evaluate_dataset(
        test_jsonl=TEST_JSONL,
        out_dir=OUT_DIR,
        max_samples=args.max_samples,
        max_episodes_per_conv=None,
        gen_max_new_tokens=args.max_new_tokens,
        gen_temperature=0.0,
        gen_top_p=1.0,
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Outputs written to {OUT_DIR}/")


if __name__ == "__main__":
    main()
