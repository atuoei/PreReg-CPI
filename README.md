# Classification pretraining-regression fine-tuning for compound-target interaction prediction

This repository documents the code used to generate the results for our article. The updated package, which is continuously being developed, can be found at [this repository](https://github.com/atuoei/REP_CPI). Please submit an issue or email my19653@163.com with any questions.

### Sample Usage

`python main.py --model DeepCPI --split ck --mode pretrain-finetune`

### Repository Organization

- `src`: Python modules containing protein and molecular featurizers, prediction architectures, and data loading scripts
- `data`: All training datasets
- `Predict`: Predicted target spectra for reproductive effect analysis
  - `predict.py` -- Loads trained models and performs prediction
  - `Analyze.ipynb` -- Statistical analysis of predicted target spectra
  - `cluster.ipynb` -- Clustering analysis of target spectra for highly active compounds
- `interpret`: Model interpretation and substructure alert identification
  - `interpret.py` -- Computes gradient-based importance and feature sensitivity
  - `broad_alerts.py` -- Identifies globally important substructures
  - `enrichment_test.py` -- Performs enrichment analysis to evaluate the significance of global substructures
- `disease`: Disease–target association analysis

  - `Analyze.ipynb` -- Computes shared and specific targets and performs disease clustering
