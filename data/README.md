# Data

The dataset is intentionally not committed to this repository. The analysis uses the public **Medical Transcriptions (MT Samples)** dataset distributed on Kaggle and filters it to six specialties:

- Cardiovascular / Pulmonary
- Gastroenterology
- Neurology
- Obstetrics / Gynecology
- Orthopedic
- Urology

## Expected local files

Place the following files in this directory before running the full notebook:

| File | Purpose |
|---|---|
| `mtsamples.csv` | Original downloaded MT Samples dataset |
| `mtsamples_six_specialties.csv` | Generated cleaned six-class dataset |
| `mtsamples_with_ner.csv` | Generated NER cache with entity text, types, counts, and errors |
| `train.csv` | Optional persisted stratified training split |
| `test.csv` | Optional persisted stratified test split |

The source dataset and generated intermediate CSVs are excluded by `.gitignore`. Do not commit clinical text or model checkpoints. Although the source dataset is presented as de-identified sample transcription data, users should still review the dataset license, provenance, and privacy requirements before redistribution or production use.

The split used in this project is stratified, with `test_size=0.20` and `random_state=42`.

## Preparation commands

From the repository root:

```bash
python -m src.data --input data/mtsamples.csv --output-dir data
python -m src.ner --input data/mtsamples.csv --output data/mtsamples_with_ner.csv
```

NER extraction is checkpointed and resumable because it is the slowest preprocessing stage. All generated files in this directory remain ignored by Git.

