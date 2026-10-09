# Task-specific metrics

Scripts that score the outputs of individual specialist tools against the ground truth of each task.

| Task | Tool | Metrics | Status |
|---|---|---|---|
| Pathology classification | CONCH | micro F1, macro F1 | [conch_crc100k_predict.py](conch_crc100k_predict.py) + [classification_metrics.py](classification_metrics.py) |
| Classification | BiomedCLIP | micro F1, macro F1 | [classification_metrics.py](classification_metrics.py) |
| QA / rationale generation | PMC-LLaMA | ROUGE-L, accuracy | to be released |
| Named entity recognition | RaTE-NER | RadGraph F1, entity-level F1 | to be released |
| Retinal segmentation | IterNet | Dice, HD95 | to be released |
| General segmentation | MedSAM | Dice, HD95 | to be released |
| Report generation | LLaVA-Rad | ROUGE-L, BLEU-4, CheXpert F1, RadGraph F1 | to be released |

## Classification (CONCH, BiomedCLIP)

`classification_metrics.py` reads a JSON list of records with a ground-truth field (`gt`) and a prediction field (`output`), and reports accuracy (equal to micro F1 for single-label prediction) and macro / weighted precision, recall and F1.

CONCH on CRC100K (7,180 images, nine tissue classes). This step requires the [CONCH](https://github.com/mahmoodlab/CONCH) package and weights:

```bash
python conch_crc100k_predict.py \
    --input      /path/to/CONCH_crc100k_eval.json \
    --output     outputs/task_metrics/conch_crc100k_pred.json \
    --checkpoint /path/to/CONCH/checkpoints/conch/pytorch_model.bin \
    --image-root /path/to/crc100k

python classification_metrics.py --input outputs/task_metrics/conch_crc100k_pred.json
```

BiomedCLIP outputs (one record per image, with `gt` and `output` labels) are scored the same way:

```bash
python classification_metrics.py --input /path/to/biomedclip_outputs.json
```

Reference results:

| | BiomedCLIP | CONCH (CRC100K) |
|---|---:|---:|
| Samples | 7,923 | 7,180 |
| Accuracy / micro F1 | 0.739 | 0.618 |
| Macro precision | 0.630 | 0.660 |
| Macro recall | 0.647 | 0.631 |
| Macro F1 | 0.623 | 0.610 |
| Weighted F1 | 0.734 | 0.605 |
