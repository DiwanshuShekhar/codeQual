# CodeQual: A dataset for fine-tuning Large Language Models for code quality assessment task

CodeQual is a novel fine-tuning dataset comprising 2250 Python code snippets from [CodeNet][https://zenodo.org/doi/10.5281/zenodo.4814769], each annotated by GPT-4 with a code quality label (low average, high). This dataset can be employed to fine-tune any Large Language Models trained on code for Code Quality Assessment task where a model is tasked with predicting a given code into it's quality tier. In this repository, we present [CodeQual](https://zenodo.org/doi/10.5281/zenodo.11062805) along with the codebase to fine-tune the pre-trained CodeBert Model to show it’s utility in fine-tuning LLMs.

# How to use this repository
This repository can be primarily used to fine-tune a code-trained LLM. The code uses [weights and bias](https://wandb.ai/) for experiment management and [direnv](https://direnv.net/) for environment variable management. You must have `.envrc` file in the project root directory with the following environment variables defined. The values shown for the environment variables are examples values suitable for fine-tuning a CodeBert mnodel.
```
export MODEL_NAME="microsoft/codebert-base"
export WANDB_PROJECT="codequalbert"
```
Once the environment variables are defined, you can fine-tune your desired pre-trained model with CodeQual using the following command from the project repository -
```
python driver.py --fine-tune testing1 3 0.00005 4
```
The first argument is the `experiment name` which is used to create a folder with this name under the experiments `directory` directory in project root. This folder houses the extracted checkpoints from the fine-tuning experiment. The `experiment name` is also used to uniquely name the experiment in Weights and Bias. The second, third and the fourth arguments are the epoch number, learning rate and the batch size, respectively.

# CodeEval Benchmark dataset
While this repository constitutes the [CodeQual fine-tuning dataset](data/codeQualDatasetv1), it can also be freely and independently downloaded from its permanent doi [link](https://zenodo.org/doi/10.5281/zenodo.11062805). The repository uses the CodeQual dataset in the Huggingface format which is present in [here](data/hf_code_qual_dataset_v1).
