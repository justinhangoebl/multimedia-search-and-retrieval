# MMSR Practical Part

## Setup

create venv

install requirements.txt

```shell
pip install -r requirements.txt
```

Copy the dataset provided from the moodle in with the folder "dataset/..."

## Generate relevancy

Open preprocessing.ipynb and run the notebook to generate ground truths.

## Generate models

Run the generation.ipynb notebook to get the tracks for simpler models.

Run AutoEncoder.ipynb, MKGCN.ipynb, diversity-vs-ndcg.ipynb, early_fusion.ipynb, early_fusion_graph.ipynb, late_fusion.ipynb and metadata_graph.ipynb to get the matrices for the advanced models.

## Evaluation

Run the accuracy_experiments.ipynb notebook to get accuracies for all models.

Run beyond_accuracy_experiments.ipynb for beyond accuracies.

## User interface

Hosted [here](https://huggingface.co/spaces/Abhi0531/MMSR-3)

Can be created locally by running app.py after the matrices have been generated. 

Enter song name and/or artist name to get retrieved tracks. Partial matches also work, text fields work as a filtering technique. A dropdown helps select the desired track. 

## Steps to reproduce evaluation results

1. Run the preprocessing.ipynb notebook to get matrices required later.

2. Run the generation.ipynb notebook to get recommendations for the simple models. Run AutoEncoder.ipynb, MKGCN.ipynb, diversity-vs-ndcg.ipynb, early_fusion.ipynb, early_fusion_graph.ipynb, late_fusion.ipynb and metadata_graph.ipynb to get the matrices for the advanced models.

3. Run the accuracy_experiments.ipynb and beyond_accuracy_experiments.ipynb notebooks to get the accuracy values.

4. Run app.py for the UI.
