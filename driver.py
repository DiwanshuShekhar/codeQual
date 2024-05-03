import argparse
import json
from json import JSONDecodeError

import httpx
import openai

from codeQual import ROOT_DIR, logging
from codeQual.annotate import Annotator
from codeQual.codenet import CodeNetPython
from codeQual.finetune import get_trainer
from codeQual.gpt import ChatGPT

SYSTEM = """You will be provided with a piece of Python code delimited by three backticks (```) followed by it's description delimited by three double quotes (\"\"\"). Your task is to follow the following steps to answer user queries.

Step 1: Assess the quality of the code in terms of the following criteria - Functionality, Readability, Pythonic, Error Handling and Efficiency. The definitions of these criteria are given below:

Functionality - does the code work as per the provided description?
Readability - how cognitively heavy the code is for a human reading the code?
Pythonic - does the code follow python practices?
Error Handling - does the code handle errors to manage edge cases?
Efficiency - is the code flexible enough to scale to large data?

Step 2: From a score of 1 to 5 where 5 being of the highest quality, score each criteria in Step 1

---

Provide output for step 2 in a  JSON format as follows.
```json
{"step2": {"functionality": "...","readability": "...", "pythonic": "...", "error_handling": "...", "efficiency": "..."}}
```
"""

parser = argparse.ArgumentParser(description="CodeQual Data and Model Training")

parser.add_argument("--annotate", nargs="+", help="annotate codenet data using chatgpt")
parser.add_argument(
    "--fine-tune", help="fine-tune model using annotated data", nargs="+"
)
parser.add_argument(
    "--search", help="search for best hyper-parameters", action="store_true"
)

import os


def write_code_qual_data(output_dir: str) -> None:
    init_msg = [{"role": "system", "content": SYSTEM}]
    chat_gpt_client = ChatGPT("gpt-4-turbo-preview", init_msg, 42, 1.0, 500)
    codenet_python_client = CodeNetPython("data/CodeQualData/py800_sampled_2each")
    skip_problems = []
    with open("data/CodeQualData/problems.txt", "r") as f:
        skip_problems = f.read().splitlines()

    annotator = Annotator(
        codenet_python_client,
        chat_gpt_client,
        output_dir,
        skip_problem_ids=skip_problems,
    )
    for problem_id, submission_id, response in annotator.annotate():
        # print(response)
        try:
            response = "\n".join(response.split("\n")[1:-1])
            chatgpt_response = json.loads(response)
        except JSONDecodeError as e:
            logging.exception(
                f"Error decoding response: {response} for problem_id: {problem_id} and submission_id: {submission_id}"
            )
            annotator.write_error(problem_id, submission_id, response, str(e))
        else:
            annotator.write_data(problem_id, submission_id, chatgpt_response)
            annotator.write_checkpoint(submission_id)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.annotate:
        write_code_qual_data(args.annotate[0])
    if args.fine_tune:
        trainer = get_trainer(
            args.fine_tune[0],
            int(args.fine_tune[1]),
            float(args.fine_tune[2]),
            int(args.fine_tune[3]),
        )
        trainer.train()
    if args.search:
        from ray.tune.search.hyperopt import HyperOptSearch
        from ray.tune.search import ConcurrencyLimiter
        from ray.tune.schedulers import ASHAScheduler
        from ray import tune

        hpo = HyperOptSearch(
            metric="eval_accuracy", mode="max"
        )  # objective, accuracy, f1, precision, recall
        hpo = ConcurrencyLimiter(hpo, max_concurrent=2)
        space = {
            "learning_rate": tune.loguniform(1e-5, 1e-1),
            "num_train_epochs": tune.choice([1, 5, 10]),
            "per_device_train_batch_size": tune.choice([1, 2, 4]),
        }

        def hp_search(trial):
            return space

        # documentation:
        # https://huggingface.co/docs/transformers/v4.39.3/en/hpo_train#hyperparameter-search-using-trainer-api
        # https://docs.ray.io/en/latest/tune/getting-started.html

        trainer = get_trainer("codequal-hyperparameter-search", 10, 1.396e-05, 8)
        best_trial = trainer.hyperparameter_search(
            hp_space=hp_search,
            direction="maximize",
            backend="ray",
            n_trials=10,
            # Choose among many libraries:
            # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
            search_alg=hpo,
            # Choose among schedulers:
            # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
            scheduler=ASHAScheduler(metric="eval_accuracy", mode="max"),
            resources_per_trial={"cpu": 2},
        )
        print(f"best trail:\n{best_trial}")
        # for n, v in best_trial.hyperparameters.items():
        #     setattr(trainer.args, n, v)

        # trainer.train()
