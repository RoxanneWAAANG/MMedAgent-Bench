"""
Build the single-tool PMC-LLaMA question-answering subset from PubMedQA.

Each PubMedQA example becomes a 4-turn dialogue:
    user request -> assistant calls PMC-LLaMA -> PMC-LLaMA output -> final answer

PMC-LLaMA is a text-only tool, so every record points to the same placeholder image
to keep the multimodal conversation format uniform.

Input: the PubMedQA "pqa_artificial" split (Hugging Face `qiaojin/PubMedQA`) exported
as a JSON list, where each item has `question`, `context.contexts` and `long_answer`.

Usage:
    python build_qa.py \
        --input  /path/to/pubmedqa_artificial.json \
        --output tool_instruct/pmc_llama_medqa_dataset.jsonl \
        --max-samples 10000
"""
import argparse
import json
import random

from tqdm import tqdm

DEFAULT_IMAGE = "dummy_images/dummy_img.png"

prompt_templates = [
    "<image>\nCan you answer this medical question based on the text provided?",
    "<image>\nRead the passages and help answer this biomedical question clearly.",
    "<image>\nWhat's the best answer using the info in these excerpts?",
    "<image>\nGive me a reasoned answer using the info shown.",
    "<image>\nFrom what you see here, how would you answer this medical question?",
    "<image>\nTake a look at the evidence. What's the correct answer and why?",
    "<image>\nBased on this info, what's your conclusion?",
    "<image>\nUse these research snippets to figure out the answer.",
    "<image>\nLet's solve this question using the given evidence.",
    "<image>\nSummarize the answer based on the details in the literature.",
    "<image>\nAnswer the biomedical question using the following study excerpts.",
    "<image>\nRely on the provided texts to answer the question accurately.",
    "<image>\nUse the passages below to construct a correct answer with a brief explanation.",
    "<image>\nRefer to the provided abstracts and generate a complete, supported answer.",
    "<image>\nSupport your answer with information drawn exclusively from the literature below.",
    "<image>\nThis question tests your understanding of the given evidence. Answer and explain.",
    "<image>\nTreat this as a research QA task. Answer using cited evidence.",
    "<image>\nExamine the snippets, then write a correct and concise response.",
    "<image>\nYour goal: produce an accurate answer supported by textual evidence.",
    "<image>\nRead, reason, and respond based on the provided biomedical excerpts.",
    "<image>\nAnalyze the excerpts and respond with an evidence-based conclusion.",
    "<image>\nGenerate an answer to the medical query, explaining briefly how you got there.",
    "<image>\nUse your understanding of the text to answer clearly and concisely.",
    "<image>\nDerive the answer from the evidence, then explain your logic.",
    "<image>\nSupport your answer with a rationale rooted in the provided information.",
    "<image>\nRead the literature, answer the question.",
    "<image>\nEvaluate the context and draw a grounded conclusion.",
    "<image>\nDevelop your answer from these research fragments and explain briefly.",
    "<image>\nSynthesize an answer from the data and outline your thought process.",
    "<image>\nDeliver answer and follow it with reasoning based on the source.",
    "<image>\nAnswer the question using the evidence.",
    "<image>\nUse the snippets to form a concise, accurate response.",
    "<image>\nFrom these texts, give a supported answer.",
    "<image>\nWhat does the evidence suggest? Provide your answer.",
    "<image>\nCreate an answer with a short explanation using the text.",
    "<image>\nUse the paragraphs to answer and summarize your reasoning.",
    "<image>\nRefer to this content to answer the biomedical query.",
    "<image>\nRead the snippets and give your answer with rationale.",
    "<image>\nUse the following to explain and answer the question.",
    "<image>\nYour task: answer based on these passages alone.",
    "<image>\nI'll show you a question and some references. Please respond with a justified answer.",
    "<image>\nHere's a question with relevant literature. What's your answer?",
    "<image>\nI'm giving you evidence for a biomedical query. How would you answer it?",
    "<image>\nUsing this information, help answer the question and explain your logic.",
    "<image>\nReview the texts and respond with reasoning.",
    "<image>\nInterpret the question and answer clearly.",
    "<image>\nAs an assistant, respond to the query with evidence-grounded reasoning.",
    "<image>\nHelp draft an evidence-backed response for this health-related question.",
]

answer_templates = [
    "Here's a concise answer based on the analysis:\n{answer}",
    "Answer and reasoning:\n{answer}",
    "This is the requested information:\n{answer}",
    "Below is the detailed response:\n{answer}",
    "I've completed the review—see the answer here:\n{answer}",
    "Final answer with explanation:\n{answer}",
    "Here is the answer in full:\n{answer}",
    "Here is the solution along with the rationale:\n{answer}",
    "Comprehensive answer:\n{answer}",
    "Detailed answer provided below:\n{answer}",
    "Here's what the evidence indicates:\n{answer}",
    "The question is addressed as follows:\n{answer}",
    "Answer (including supporting details):\n{answer}",
    "Full response:\n{answer}",
    "My complete answer is:\n{answer}",
    "The findings are summarized here:\n{answer}",
    "Answer, with relevant details:\n{answer}",
    "Completed answer:\n{answer}",
    "Response with explanation:\n{answer}",
    "Full explanation and answer:\n{answer}",
    "Here is the final response:\n{answer}",
    "Solution and rationale:\n{answer}",
    "Comprehensive explanation:\n{answer}",
    "The final answer is presented below:\n{answer}",
    "After analysis, the answer is:\n{answer}",
    "Full answer (see details):\n{answer}",
    "My conclusion:\n{answer}",
    "Please find the answer here:\n{answer}",
    "Answer with supporting points:\n{answer}",
    "Here is an in-depth answer:\n{answer}",
    "Explicit answer and reasoning:\n{answer}",
    "Definitive answer:\n{answer}",
    "Answer summary:\n{answer}",
    "Complete solution:\n{answer}",
    "Here's the resolved answer:\n{answer}",
    "The following addresses your query:\n{answer}",
    "Answer (detailed):\n{answer}",
    "Here's the explanation and answer:\n{answer}",
    "My detailed response:\n{answer}",
    "Answer provided below:\n{answer}",
]

def join_context(ctx):
    """Join context passages into a single string"""
    return "\n\n".join(ctx) if isinstance(ctx, list) else str(ctx)

def transform(ex, image_path):
    """Transform a PubMedQA example into the new image-based format while preserving Q&A nature"""
    # Extract original question, context, and answer
    question = ex["question"].strip()
    context = join_context(ex["context"]["contexts"])
    answer = ex.get("long_answer", "").strip()

    # Select random templates
    system_prompt = random.choice(prompt_templates)
    
    # Build the user prompt with question and context (similar to original)
    user_prompt = (
        system_prompt +
        f"\n\n### Question:\n{question}\n\n"
        f"### Context:\n{context}"
    )

    # Create the PMC-LLaMA output message
    pmc_output = f"PMC-LLaMA output: {answer}\n\nAnswer my first request: {system_prompt.replace('<image>', '').strip()}\n\n"

    # Select final response template
    friendly_reply = random.choice(answer_templates).format(answer=answer)

    return {
        "image": image_path,
        "conversations": [
            {
                "from": "human",
                "value": user_prompt
            },
            {
                "from": "gpt",
                "thoughts": "To answer this question and provide a detailed rationale, I'll call the PMC-LLaMA model.",
                "actions": [
                    {
                        "API_name": "PMC-LLaMA",
                        "API_params": {"query": question}
                    }
                ],
                "value": "Calling PMC-LLaMA to get the answer and rationale..."
            },
            {
                "from": "human",
                "value": pmc_output
            },
            {
                "from": "gpt",
                "thoughts": "The PMC-LLaMA tool has completed the analysis. Now I can answer based on its output.",
                "actions": [],
                "value": friendly_reply
            }
        ]
    }

def build_instruction_dataset(input_path, output_path, max_samples, image_path):
    with open(input_path, "r", encoding="utf-8") as f:
        examples = json.load(f)

    subset = examples[:max_samples]

    with open(output_path, "w", encoding="utf-8") as out:
        for ex in tqdm(subset, desc=f"Transforming first {max_samples} examples"):
            record = transform(ex, image_path)
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(subset)} records to '{output_path}'")


def parse_args():
    p = argparse.ArgumentParser(description="Build PMC-LLaMA QA tool-use dialogues from PubMedQA.")
    p.add_argument("--input", required=True, help="PubMedQA JSON file.")
    p.add_argument("--output", required=True, help="Output JSONL path.")
    p.add_argument("--max-samples", type=int, default=10000)
    p.add_argument("--image", default=DEFAULT_IMAGE,
                   help="Placeholder image path written into every record.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    build_instruction_dataset(args.input, args.output, args.max_samples, args.image)
