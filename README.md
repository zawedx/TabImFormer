# LMTDE

## Data preprocessing

1. put the data at DATA_PATH
2. follow the step1.ipynb

## Training and evaluation

```
bash train.sh
```

## Hyperparameter experiment

```
cd ./experiment/hyperparameter_experiment
bash gridsearch.sh
```

## Image-only experiment

No difference in code. Please modify the metadata file to mask tabular data.

```
bash train.sh
```

## Table-only experiment

No difference in code. Please modify the metadata file to mask mri data.

```
bash train.sh
```

## SHAP experiment

SHAP value visualization

```
cd ./experiment/shap_experiment
bash shap.sh
```
