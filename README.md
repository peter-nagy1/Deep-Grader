# Automatic Program Grading Models
This repository contains code and data for reproducing the experiments in [Towards Deep Learning Models for Automatic Computer Programs Grading](LINK TO PAPER).
* Deep Grader (CodeBERT)
* Deep Grader (UniXcoder)
* Deep Siamese Grader (CodeBERT)
* Deep Siamese Grader (UniXcoder)

Deep Grader and Deep Siamese Grader are models leveraging large pre-trained models (CodeBERT, UniXcoder) fine-tuned on the task of automatic program grading with Python and C++ programming languages.

## Data
Python and C++ data are available in forms of Pandas DataFrames separating train, validation, and test sets for both question independent and question dependent settings. Furthermore, we provide programs with their assigned grades and unpaired programs used in Majority Vote Grading and Incremental Transductive Grading.

## Dependency
* pip install torch
* pip install transformers

## Fine-Tune
We provide fine-tuning settings for automatic program grading.

Example training of Deep Grader using UniXcoder as the encoder on Python data utilizing the question independent setting:
```
python run.py \
    --model_name deep_grader \
    --encoder_name microsoft/unixcoder-base \
    --output_dir saved_checkpoints \
    --language python \
    --setting independent \
```
