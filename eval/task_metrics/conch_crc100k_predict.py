"""
CONCH zero-shot predictions on the CRC100K test set (task-specific metrics: CONCH).

Each image is matched against prompt ensembles (22 templates x class synonyms) for the
nine CRC100K tissue classes, and the predicted class key is written to "output".
Score the result with classification_metrics.py.

Input JSON: a list of {"image": ..., "query": ..., "gt": "{ADI}", ...}.

Requires the CONCH package and weights: https://github.com/mahmoodlab/CONCH

Usage:
    python conch_crc100k_predict.py \
        --input      /path/to/CONCH_crc100k_eval.json \
        --output     outputs/task_metrics/conch_crc100k_pred.json \
        --checkpoint /path/to/CONCH/checkpoints/conch/pytorch_model.bin \
        --image-root /path/to/crc100k
"""
import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from conch.open_clip_custom import create_model_from_pretrained, tokenize, get_tokenizer

prompt_templates = [
    "CLASSNAME",
    "a photomicrograph showing CLASSNAME",
    "a photomicrograph of CLASSNAME",
    "an image of CLASSNAME",
    "an image showing CLASSNAME",
    "an example of CLASSNAME",
    "CLASSNAME is shown",
    "this is CLASSNAME",
    "there is CLASSNAME",
    "a histopathological image showing CLASSNAME",
    "a histopathological image of CLASSNAME",
    "a histopathological photograph of CLASSNAME",
    "a histopathological photograph showing CLASSNAME",
    "shows CLASSNAME",
    "presence of CLASSNAME",
    "CLASSNAME is present",
    "an H&E stained image of CLASSNAME",
    "an H&E stained image showing CLASSNAME",
    "an H&E image showing CLASSNAME",
    "an H&E image of CLASSNAME",
    "CLASSNAME, H&E stain",
    "CLASSNAME, H&E"
]

CRC = {'ADI': ['adipose', 'adipose tissue', 'adipocytes', 'fat', 'fat cells'],
    'BACK': ['background', 'penmarking', 'empty space', 'background artifacts'],
    'DEB': ['debris', 'colorectal adenocarcinoma debris and necrosis', 'necrosis', 'necrotic debris'],
    'LYM': ['lymphocytes', 'lymphoid aggregate', 'immune cells', 'lymphoid infiltrate', 'inflammatory cells'],
    'MUC': ['mucus', 'mucin', 'mucus pool', 'mucin pool'],
    'MUS': ['smooth muscle', 'smooth muscle tissue', 'muscle', 'muscularis propria', 'muscularis mucosa'],
    'NORM': ['normal colon mucosa', 'uninvolved colon mucosa', 'benign colon mucosa', 'benign epithelium'],
    'STR': ['cancer-associated stroma', 'tumor-associated stroma', 'stromal cells', 'stromal tissue', 'stroma'],
    'TUM': ['colorectal adenocarcinoma epithelium', 'colorectal adenocarcinoma', 'tumor', 'adenocarcinoma', 'malignant epithelium'],}


def encode_class_prompts(model, tokenizer, class_dict, device):
    prompts, keys = [], []
    for key, synonyms in class_dict.items():
        for syn in synonyms:
            for template in prompt_templates:
                prompts.append(template.replace("CLASSNAME", syn))
                keys.append(key)
    tokenized = tokenize(texts=prompts, tokenizer=tokenizer).to(device)
    return model.encode_text(tokenized), keys


def main():
    p = argparse.ArgumentParser(description="CONCH zero-shot classification on CRC100K.")
    p.add_argument("--input", required=True, help="Test JSON with image paths and ground truth.")
    p.add_argument("--output", required=True, help="Where to write the JSON with predictions.")
    p.add_argument("--checkpoint", required=True, help="CONCH pytorch_model.bin.")
    p.add_argument("--image-root", default=None, help="Prefix for relative image paths in --input.")
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    model, preprocess = create_model_from_pretrained("conch_ViT-B-16", args.checkpoint, device=device)
    model.eval()

    with torch.inference_mode():
        text_embeddings, keys = encode_class_prompts(model, get_tokenizer(), CRC, device)
    print(f"[CRC] {len(CRC)} classes, {len(keys)} prompts -> text_emb {tuple(text_embeddings.shape)}")

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} images from {args.input}")

    failed = 0
    with torch.inference_mode():
        pbar = tqdm(data, desc="Inference", unit="img")
        for item in pbar:
            image_path = item["image"]
            if args.image_root and not os.path.isabs(image_path):
                image_path = os.path.join(args.image_root, image_path)
            try:
                image_tensor = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                image_embeddings = model.encode_image(image_tensor)
                sim_scores = (image_embeddings @ text_embeddings.T * model.logit_scale.exp()).softmax(dim=-1).cpu().numpy()
                item["output"] = [keys[sim_scores.argmax()]]
                pbar.set_postfix(failed=failed, last=Path(image_path).name[:20])
            except Exception as e:
                failed += 1
                item["output"] = None
                item["error"] = str(e)
                tqdm.write(f"[ERROR] {image_path}: {e}")

    print(f"Inference complete. Failed: {failed}/{len(data)}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    main()
