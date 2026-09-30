# Task-specific metrics

The scripts that score specialist tool outputs from end-to-end execution will be added here in a later release:

| Task | Tool | Metrics |
|---|---|---|
| QA / rationale generation | PMC-LLaMA | ROUGE-L, accuracy |
| Named entity recognition | RaTE-NER | RadGraph F1, entity-level F1 |
| Pathology classification | CONCH | micro F1, macro F1 |
| Classification | BiomedCLIP | micro F1, macro F1 |
| Retinal segmentation | IterNet | Dice, HD95 |
| General segmentation | MedSAM | Dice, HD95 |
| Report generation | LLaVA-Rad | ROUGE-L, BLEU-4, CheXpert F1, RadGraph F1 |
