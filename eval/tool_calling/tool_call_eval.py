"""
Tool-call evaluation for MMedAgent-Plus (paper: "Tool-Calling Accuracy").

Replays each test dialogue turn by turn, lets the model respond to every user turn,
parses the tool calls (`actions` / `API_name`) from each response, and compares them
with the ground-truth `actions` of the next assistant turn.

Usage:
    python tool_call_eval.py \
        --test-jsonl $MMEDAGENT_DATA_ROOT/test/single_tool_multi_round.jsonl \
        --base-dir   $MMEDAGENT_DATA_ROOT \
        --checkpoint /path/to/mmedagent-plus-7b \
        --out-dir    outputs/tool_calling/single_tool_multi_round

Per-dialogue predictions are written to <out-dir>; run generate_eval_metrics.py on
them for precision / recall / F1 and the stratified breakdowns.
"""
import argparse
import os
import re
import json
import torch
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
from transformers import AutoProcessor, AutoTokenizer
from qwen_vl_utils import process_vision_info
import numpy as np
from tqdm import tqdm

class ToolCallEvaluator:
    def __init__(self, model, processor, tokenizer, base_dir: str = ""):
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        # Image paths in the test files are relative to this directory.
        self.base_dir = base_dir
        
    def parse_actions_from_response(self, response: str) -> List[Dict]:
        """
        Parse tool calls from model response.
        Expected format: actions: [{"API_name": "ToolName", "API_params": {...}}]
        """
        actions = []
        
        # Try to find actions in various formats
        patterns = [
            r'actions?\s*[:=]\s*\[(.*?)\]',
            r'API_name["\']?\s*[:=]\s*["\']([^"\']+)["\']',
            r'"API_name"\s*:\s*"([^"]+)"',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if matches:
                if pattern == patterns[0]:  # Full actions array
                    try:
                        # Try to parse as JSON-like structure
                        action_str = matches[0]
                        # Simple regex to extract API names from the action string
                        api_matches = re.findall(r'"API_name"\s*:\s*"([^"]+)"', action_str)
                        for api_name in api_matches:
                            actions.append({"API_name": api_name, "API_params": {}})
                    except:
                        pass
                else:  # Individual API names
                    for match in matches:
                        actions.append({"API_name": match, "API_params": {}})
                break
        
        # Fallback: look for common tool names directly
        if not actions:
            tool_names = [
                "MedSAM", "LLaVA-Rad", "LLaVA-Med", "SpecialistVLMs", 
                "RaTE-NER", "ChatCAD-R", "UniGradICON", "HealthGPT",
                "BiomedClip", "LLaVA", "PMC-LLaMA"
            ]
            
            for tool in tool_names:
                if tool.lower() in response.lower():
                    actions.append({"API_name": tool, "API_params": {}})
        
        return actions
    
    def chat_with_model(self, user_input: str, image_paths: List[str] = None, 
                       conversation_history: List = None) -> Tuple[str, List]:
        """Modified version of your chat function"""
        if conversation_history is None:
            conversation_history = []

        content = []
        
        if image_paths:
            for image_path in image_paths:
                if os.path.exists(image_path):
                    content.append({"type": "image", "image": image_path})
        
        content.append({"type": "text", "text": user_input})
        
        user_message = {
            "role": "human",
            "content": content
        }
        
        conversation_history.append(user_message)
        
        text = self.processor.apply_chat_template(
            conversation_history, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(conversation_history)
        
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.7,
                top_p=0.8
            )
        
        response_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in 
            zip(inputs.input_ids, generated_ids)
        ]

        response = self.processor.batch_decode(
            response_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )[0]
        
        assistant_message = {
            "role": "gpt",
            "content": [{"type": "text", "text": response}]
        }
        conversation_history.append(assistant_message)
        
        return response, conversation_history
    
    def evaluate_single_conversation(self, conversation_data: Dict, conv_id: int) -> Dict:
        """Evaluate a single conversation"""
        results = {
            "conversation_id": conv_id,
            "image": conversation_data.get("image", ""),
            "total_turns": 0,
            "tool_call_turns": 0,
            "correct_tool_calls": 0,
            "total_tool_calls": 0,
            "exact_matches": 0,
            "partial_matches": 0,
            "predictions": [],
            "ground_truth": [],
            "all_turns_correct": True,  # whole-case accuracy
            "error_details": []
        }
        
        image_path = conversation_data.get("image", "")
        conversations = conversation_data.get("conversations", [])
        
        # Build conversation history
        conversation_history = []
        turn_predictions = []  # Store predictions for each turn
        
        for i, turn in enumerate(conversations):
            if turn["from"] == "human":
                # Add user message to history
                user_input = turn["value"]
                
                # Handle image tags
                if isinstance(image_path, list):
                    image_paths = [os.path.join(self.base_dir, img) 
                                for img in image_path if img]
                else:
                    image_paths = [os.path.join(self.base_dir, image_path)]

                user_input = user_input.replace("<image>", "")
                
                # Get model response
                try:
                    response, conversation_history = self.chat_with_model(
                        user_input, image_paths, conversation_history
                    )
                    
                    # Parse predicted actions
                    predicted_actions = self.parse_actions_from_response(response)
                    turn_predictions.append({
                        "turn": i,
                        "user_input": user_input,
                        "model_response": response,
                        "predicted_actions": predicted_actions
                    })
                    
                except Exception as e:
                    print(f"Error in model inference: {e}")
                    predicted_actions = []
                    turn_predictions.append({
                        "turn": i,
                        "user_input": user_input,
                        "error": str(e),
                        "predicted_actions": []
                    })
                
            elif turn["from"] == "gpt":
                # This is the ground truth response
                gt_actions = turn.get("actions", [])
                
                results["total_turns"] += 1
                
                # Get the corresponding prediction
                if turn_predictions:
                    predicted_actions = turn_predictions[-1]["predicted_actions"]
                else:
                    predicted_actions = []
                
                if gt_actions:
                    results["tool_call_turns"] += 1
                    results["total_tool_calls"] += len(gt_actions)
                    
                    # Compare predictions with ground truth
                    gt_tool_names = [action.get("API_name", "") for action in gt_actions]
                    pred_tool_names = [action.get("API_name", "") for action in predicted_actions]
                    
                    results["ground_truth"].append(gt_tool_names)
                    results["predictions"].append(pred_tool_names)
                    
                    # Check if this turn is correct
                    turn_correct = False
                    
                    # Exact match: same tools in same order
                    if gt_tool_names == pred_tool_names:
                        results["exact_matches"] += 1
                        turn_correct = True
                    
                    # Partial match: same tools, possibly different order
                    if set(gt_tool_names) == set(pred_tool_names):
                        results["partial_matches"] += 1
                    
                    # Count correct individual tool calls
                    correct_in_turn = 0
                    for gt_tool in gt_tool_names:
                        if gt_tool in pred_tool_names:
                            results["correct_tool_calls"] += 1
                            correct_in_turn += 1
                    
                    # If not exact match, record error
                    if not turn_correct:
                        results["all_turns_correct"] = False
                        results["error_details"].append({
                            "turn_index": results["total_turns"] - 1,
                            "ground_truth": gt_tool_names,
                            "predicted": pred_tool_names,
                            "user_input": turn_predictions[-1].get("user_input", ""),
                            "model_response": turn_predictions[-1].get("model_response", ""),
                            "error_type": self._classify_error(gt_tool_names, pred_tool_names)
                        })
        
        return results
    
    def _classify_error(self, gt_tools: List[str], pred_tools: List[str]) -> str:
        """Classify the type of error"""
        if not pred_tools:
            return "missing_all_tools"
        elif not gt_tools:
            return "extra_tools"
        elif set(gt_tools) == set(pred_tools):
            return "wrong_order"
        elif set(pred_tools).issubset(set(gt_tools)):
            return "missing_some_tools"
        elif set(gt_tools).issubset(set(pred_tools)):
            return "extra_some_tools"
        else:
            return "wrong_tools"
    
    def evaluate_dataset(self, test_file: str, base_dir: str = "", 
                        max_samples: int = None) -> Dict:
        """Evaluate the entire dataset"""
        
        if base_dir:
            self.base_dir = base_dir

        print(f"Loading test data from {test_file}")
        
        # Load test data
        test_data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_data.append(json.loads(line))
        
        if max_samples:
            import random
            test_data = random.sample(test_data, min(max_samples, len(test_data)))
        
        print(f"Loaded {len(test_data)} conversations")
        
        # Initialize aggregated results
        aggregated_results = {
            "total_conversations": len(test_data),
            "correct_conversations": 0,
            "total_turns": 0,
            "tool_call_turns": 0,
            "correct_tool_calls": 0,
            "total_tool_calls": 0,
            "exact_matches": 0,
            "partial_matches": 0,
            "conversation_results": [],
            "error_samples": [],
            "tool_usage_stats": defaultdict(int),
            "error_analysis": defaultdict(int)
        }
        
        # Evaluate each conversation
        for i, conv_data in enumerate(tqdm(test_data, desc="Evaluating conversations")):
            try:
                results = self.evaluate_single_conversation(conv_data, i)
                
                # Check if whole conversation is correct
                if results["all_turns_correct"]:
                    aggregated_results["correct_conversations"] += 1
                else:
                    # Store error sample
                    aggregated_results["error_samples"].append({
                        "conversation_id": i,
                        "image": conv_data.get("image", ""),
                        "original_data": conv_data,
                        "predictions": results["predictions"],
                        "ground_truth": results["ground_truth"],
                        "error_details": results["error_details"]
                    })
                
                # Aggregate results
                for key in ["total_turns", "tool_call_turns", "correct_tool_calls", 
                           "total_tool_calls", "exact_matches", "partial_matches"]:
                    aggregated_results[key] += results[key]
                
                # Store individual results
                aggregated_results["conversation_results"].append({
                    "conversation_id": i,
                    "image": conv_data.get("image", ""),
                    "all_turns_correct": results["all_turns_correct"],
                    **{k: v for k, v in results.items() 
                       if k not in ["error_details", "conversation_id", "image"]}
                })
                
                # Analyze tool usage
                for gt_tools in results["ground_truth"]:
                    for tool in gt_tools:
                        aggregated_results["tool_usage_stats"][tool] += 1
                
                # Count error types
                for error in results["error_details"]:
                    aggregated_results["error_analysis"][error["error_type"]] += 1
                
            except Exception as e:
                print(f"Error evaluating conversation {i}: {e}")
                aggregated_results["error_analysis"]["evaluation_errors"] += 1
        
        # Calculate metrics
        metrics = self.calculate_metrics(aggregated_results)
        aggregated_results["metrics"] = metrics
        
        return aggregated_results
    
    def calculate_metrics(self, results: Dict) -> Dict:
        """Calculate evaluation metrics"""
        metrics = {}
        
        # 1. Tool Call Accuracy (correct tool calls / total tool calls)
        if results["total_tool_calls"] > 0:
            metrics["tool_call_accuracy"] = results["correct_tool_calls"] / results["total_tool_calls"]
        else:
            metrics["tool_call_accuracy"] = 0.0
        
        # 2. Whole Case Accuracy (conversations with all turns correct / total conversations)
        if results["total_conversations"] > 0:
            metrics["whole_case_accuracy"] = results["correct_conversations"] / results["total_conversations"]
        else:
            metrics["whole_case_accuracy"] = 0.0
        
        # Exact match rate (perfect sequence matches / tool-calling turns)
        if results["tool_call_turns"] > 0:
            metrics["exact_match_rate"] = results["exact_matches"] / results["tool_call_turns"]
            metrics["partial_match_rate"] = results["partial_matches"] / results["tool_call_turns"]
        else:
            metrics["exact_match_rate"] = 0.0
            metrics["partial_match_rate"] = 0.0
        
        # Tool call coverage (turns with tool calls / total turns)
        if results["total_turns"] > 0:
            metrics["tool_call_coverage"] = results["tool_call_turns"] / results["total_turns"]
        else:
            metrics["tool_call_coverage"] = 0.0
        
        return metrics
    
    def save_error_samples(self, results: Dict, out_dir: str):
        """
        Save detailed error samples for analysis
        """
        os.makedirs(out_dir, exist_ok=True)
        
        error_samples = results.get("error_samples", [])
        
        if not error_samples:
            print("No error samples to save.")
            return
        
        # 1. Save all error samples in JSONL format
        error_file = os.path.join(out_dir, "error_samples.jsonl")
        with open(error_file, "w", encoding="utf-8") as f:
            for sample in error_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        print(f"✓ Saved {len(error_samples)} error samples to {error_file}")
        
        # 2. Save original data of error samples (for re-testing)
        original_error_file = os.path.join(out_dir, "error_samples_original.jsonl")
        with open(original_error_file, "w", encoding="utf-8") as f:
            for sample in error_samples:
                f.write(json.dumps(sample["original_data"], ensure_ascii=False) + "\n")
        print(f"✓ Saved original error data to {original_error_file}")
        
        # 3. Create detailed error analysis report
        error_analysis = {
            "total_errors": len(error_samples),
            "error_type_counts": defaultdict(int),
            "tool_specific_errors": defaultdict(lambda: defaultdict(int)),
            "error_examples": []
        }
        
        for sample in error_samples:
            for error_detail in sample["error_details"]:
                error_type = error_detail["error_type"]
                error_analysis["error_type_counts"][error_type] += 1
                
                # Track which tools are being confused
                gt_tools = error_detail["ground_truth"]
                pred_tools = error_detail["predicted"]
                
                for gt_tool in gt_tools:
                    if gt_tool not in pred_tools:
                        error_analysis["tool_specific_errors"][gt_tool]["missed"] += 1
                
                for pred_tool in pred_tools:
                    if pred_tool not in gt_tools:
                        error_analysis["tool_specific_errors"][pred_tool]["hallucinated"] += 1
        
        # Add examples for each error type (first 3 of each)
        error_type_examples = defaultdict(list)
        for sample in error_samples:
            for error_detail in sample["error_details"]:
                error_type = error_detail["error_type"]
                if len(error_type_examples[error_type]) < 3:
                    error_type_examples[error_type].append({
                        "conversation_id": sample["conversation_id"],
                        "turn_index": error_detail["turn_index"],
                        "ground_truth": error_detail["ground_truth"],
                        "predicted": error_detail["predicted"],
                        "user_input": error_detail["user_input"][:200]  # Truncate long inputs
                    })
        
        error_analysis["error_examples"] = dict(error_type_examples)
        
        # Save error analysis report
        analysis_file = os.path.join(out_dir, "error_analysis.json")
        with open(analysis_file, "w", encoding="utf-8") as f:
            json.dump(error_analysis, f, indent=2, ensure_ascii=False)
        print(f"✓ Saved error analysis to {analysis_file}")
        
        # 4. Create human-readable error report
        report_file = os.path.join(out_dir, "error_report.txt")
        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("ERROR ANALYSIS REPORT\n")
            f.write("=" * 70 + "\n\n")
            
            f.write(f"Total Error Samples: {len(error_samples)}\n")
            f.write(f"Error Rate: {len(error_samples) / results['total_conversations'] * 100:.2f}%\n\n")
            
            f.write("Error Type Distribution:\n")
            f.write("-" * 70 + "\n")
            for error_type, count in sorted(error_analysis["error_type_counts"].items(), 
                                           key=lambda x: x[1], reverse=True):
                percentage = count / len(error_samples) * 100
                f.write(f"  {error_type}: {count} ({percentage:.1f}%)\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Tool-Specific Errors:\n")
            f.write("-" * 70 + "\n")
            for tool, errors in sorted(error_analysis["tool_specific_errors"].items()):
                f.write(f"\n  {tool}:\n")
                f.write(f"    Missed (should have called): {errors.get('missed', 0)}\n")
                f.write(f"    Hallucinated (incorrectly called): {errors.get('hallucinated', 0)}\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("Example Errors by Type:\n")
            f.write("-" * 70 + "\n")
            for error_type, examples in error_analysis["error_examples"].items():
                f.write(f"\n{error_type.upper()}:\n")
                for i, example in enumerate(examples, 1):
                    f.write(f"\n  Example {i} (Conv {example['conversation_id']}, Turn {example['turn_index']}):\n")
                    f.write(f"    Ground Truth: {example['ground_truth']}\n")
                    f.write(f"    Predicted: {example['predicted']}\n")
                    f.write(f"    Input: {example['user_input']}\n")
        
        print(f"✓ Saved human-readable error report to {report_file}")
        print(f"\n📊 Error Summary:")
        print(f"  Total Errors: {len(error_samples)}")
        print(f"  Error Rate: {len(error_samples) / results['total_conversations'] * 100:.2f}%")

    def extract_specific_errors(self, test_jsonl: str, ids: List[int], out_path: str):
        """Extract specific error samples by ID"""
        id_set = set(ids)
        with open(test_jsonl, "r", encoding="utf-8") as fin, \
             open(out_path, "w", encoding="utf-8") as fout:
            for i, line in enumerate(fin):
                if i in id_set:
                    fout.write(line)

    def print_results(self, results: Dict):
        """Print evaluation results"""
        print("\n" + "="*70)
        print("TOOL CALL EVALUATION RESULTS")
        print("="*70)
        
        metrics = results["metrics"]
        
        print(f"\nDATASET STATISTICS:")
        print(f"  Total Conversations: {results['total_conversations']}")
        print(f"  Total Turns: {results['total_turns']}")
        print(f"  Tool-calling Turns: {results['tool_call_turns']}")
        print(f"  Tool Call Coverage: {metrics['tool_call_coverage']:.3f}")
        
        print(f"\n🎯 PRIMARY METRICS:")
        print(f"  Tool Call Accuracy: {metrics['tool_call_accuracy']:.3f}")
        print(f"    └─ ({results['correct_tool_calls']} / {results['total_tool_calls']} individual tool calls)")
        print(f"  Whole Case Accuracy: {metrics['whole_case_accuracy']:.3f}")
        print(f"    └─ ({results['correct_conversations']} / {results['total_conversations']} conversations)")
        
        print(f"\nSECONDARY METRICS:")
        print(f"  Exact Match Rate: {metrics['exact_match_rate']:.3f}")
        print(f"  Partial Match Rate: {metrics['partial_match_rate']:.3f}")
        
        print(f"\nTOOL USAGE STATISTICS:")
        tool_stats = results["tool_usage_stats"]
        for tool, count in sorted(tool_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {tool}: {count}")
        
        if results["error_analysis"]:
            print(f"\nERROR TYPE DISTRIBUTION:")
            for error_type, count in sorted(results["error_analysis"].items(), 
                                           key=lambda x: x[1], reverse=True):
                print(f"  {error_type}: {count}")
        
        print("="*70)
    
    def save_results(self, results: Dict, output_file: str):
        """Save detailed results to file"""
        # Don't save original_data in the main results file to reduce size
        results_copy = results.copy()
        if "error_samples" in results_copy:
            results_copy["error_samples"] = [
                {k: v for k, v in sample.items() if k != "original_data"}
                for sample in results_copy["error_samples"]
            ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_copy, f, indent=2, ensure_ascii=False)
        print(f"✓ Detailed results saved to: {output_file}")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate tool-call accuracy of a Qwen2.5-VL agent checkpoint.")
    p.add_argument("--test-jsonl", required=True, help="Test split JSONL.")
    p.add_argument("--base-dir", required=True, help="Directory that image paths in the test file are relative to.")
    p.add_argument("--checkpoint", required=True, help="Fine-tuned model directory or HF repo ID.")
    p.add_argument("--processor-dir", default=None, help="Processor directory (defaults to --checkpoint).")
    p.add_argument("--base-model", default="Qwen/Qwen2.5-VL-7B-Instruct",
                   help="Base model used only to recover a missing chat template.")
    p.add_argument("--out-dir", default="outputs/tool_calling")
    p.add_argument("--max-samples", type=int, default=2000,
                   help="Randomly sample this many dialogues (0 = use all).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    OUT_DIR = args.out_dir
    checkpoint_path = args.checkpoint
    processor_dir = args.processor_dir or args.checkpoint
    base_model_name = args.base_model

    import random
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading model components...")
    
    # Load model
    from transformers import Qwen2_5_VLForConditionalGeneration
    
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        checkpoint_path,
        torch_dtype="auto",
        device_map="auto"
    )
    
    processor = AutoProcessor.from_pretrained(processor_dir)
    
    if not hasattr(processor, 'chat_template') or processor.chat_template is None:
        print("Chat template missing, loading from base model...")
        base_processor = AutoProcessor.from_pretrained(base_model_name)
        processor.chat_template = base_processor.chat_template
    
    tokenizer = processor.tokenizer

    if not getattr(processor, "chat_template", None):
        base_processor = AutoProcessor.from_pretrained(base_model_name)
        processor.chat_template = base_processor.chat_template
    
    print("✓ All components loaded successfully!")
    
    # Initialize evaluator
    evaluator = ToolCallEvaluator(model, processor, tokenizer, base_dir=args.base_dir)
    
    # Run evaluation
    print("\nStarting evaluation...")
    results = evaluator.evaluate_dataset(
        test_file=args.test_jsonl,
        base_dir=args.base_dir,
        max_samples=args.max_samples or None,
    )
    
    # Print results
    evaluator.print_results(results)
    
    # Save detailed results
    output_file = os.path.join(OUT_DIR, "tool_call_evaluation_results.json")
    evaluator.save_results(results, output_file)
    
    # Save error samples for analysis
    evaluator.save_error_samples(results, OUT_DIR)
    
    # Generate summary report
    summary_file = os.path.join(OUT_DIR, "evaluation_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Tool Call Evaluation Summary\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Tool Call Accuracy: {results['metrics']['tool_call_accuracy']:.4f}\n")
        f.write(f"Whole Case Accuracy: {results['metrics']['whole_case_accuracy']:.4f}\n")
        f.write(f"Exact Match Rate: {results['metrics']['exact_match_rate']:.4f}\n")
        f.write(f"Partial Match Rate: {results['metrics']['partial_match_rate']:.4f}\n")
        f.write(f"Tool Call Coverage: {results['metrics']['tool_call_coverage']:.4f}\n")
        f.write(f"\nCorrect Conversations: {results['correct_conversations']} / {results['total_conversations']}\n")
        f.write(f"Correct Tool Calls: {results['correct_tool_calls']} / {results['total_tool_calls']}\n")
        f.write(f"Error Samples: {len(results.get('error_samples', []))}\n")
    
    print(f"\n✅ Evaluation completed! Check {OUT_DIR}/ for results:")
    print(f"  - tool_call_evaluation_results.json: Full evaluation results")
    print(f"  - evaluation_summary.txt: Quick summary")
    print(f"  - error_samples.jsonl: Detailed error samples")
    print(f"  - error_samples_original.jsonl: Original format for re-testing")
    print(f"  - error_analysis.json: Structured error analysis")
    print(f"  - error_report.txt: Human-readable error report")


if __name__ == "__main__":
    main()
