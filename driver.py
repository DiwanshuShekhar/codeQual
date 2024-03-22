import argparse
import json
from json import JSONDecodeError

import httpx
import openai

from codeQual import ROOT_DIR, logging
from codeQual.annotate import Annotator
from codeQual.codenet import CodeNetPython
from codeQual.finetune import trainer
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
    "--fine-tune", help="fine-tune model using annotated data", action="store_true"
)


def write_code_qual_data(output_dir: str) -> None:
    init_msg = [{"role": "system", "content": SYSTEM}]
    chat_gpt_client = ChatGPT("gpt-4-turbo-preview", init_msg, 42, 1.0, 500)
    codenet_python_client = CodeNetPython("data/CodeQualData/py800_sampled")
    annotator = Annotator(
        codenet_python_client, chat_gpt_client, skip_problem_ids=["p02546", "p03494"]
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
        except openai.BadRequestError as e:
            logging.exception(
                f"Error decoding response: {response} for problem_id: {problem_id} and submission_id: {submission_id}"
            )
        except httpx.HTTPStatusError as e:
            logging.exception(
                f"Error decoding response: {response} for problem_id: {problem_id} and submission_id: {submission_id}"
            )
            annotator.write_error(problem_id, submission_id, response, str(e))
            annotator.write_error(problem_id, submission_id, response, str(e))
        else:
            annotator.write_data(
                problem_id, submission_id, chatgpt_response, output_dir
            )
            annotator.write_checkpoint(submission_id, output_dir)


if __name__ == "__main__":
    args = parser.parse_args()
    if args.annotate:
        write_code_qual_data(args.output_dir)
    if args.fine_tune:
        trainer.train()
