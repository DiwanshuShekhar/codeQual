import json
from json import JSONDecodeError

from codeQual import ROOT_DIR, logging
from codeQual.annotate import Annotator
from codeQual.codenet import CodeNetPython
from codeQual.gpt import ChatGPT

SYSTEM = """You will be provided with a piece of Python code delimited by three backticks (```). Your task is to follow the following steps to answer user queries.

Step 1: Provide a brief description of the code

Step 2: Provide descriptive assessment of the quality of the code in terms of the following criteria - Functionality, Readability, Pythonic, Error Handling and Efficiency. The definitions of these criteria are given below:

Functionality - does the code work?
Readability - how cognitively heavy the code is for a human reading the code?
Pythonic - does the code follow python practices?
Error Handling - does the code handle errors to manage edge cases?
Efficiency - is the code flexible enough to scale to large data?

Step 3: From a score of 1 to 5 where 5 being of the highest quality, score each criteria in Step 2

---

Provide output for all above steps in a single JSON format as follows:

{"step1": "...",
 "step2": {"functionality": "...","readability": "...", "pythonic": "...", "error_handling": "...", "efficiency": "..."},
 "step3": {"functionality": "...","readability": "...", "pythonic": "...", "error_handling": "...", "efficiency": "..."}
}
"""


def write_code_qual_data(path: str) -> None:
    init_msg = [{"role": "system", "content": SYSTEM}]
    chat_gpt_client = ChatGPT("gpt-4", init_msg, 42, 1.0, 500)
    codenet_python_client = CodeNetPython("data/CodeNet/python_800")
    annotator = Annotator(codenet_python_client, chat_gpt_client)

    data = {}

    with open(path, "a") as f:
        for problem_id, submission_id, response in annotator.annotate():
            # print(response)
            try:
                chatgpt_response = json.loads(response)
            except JSONDecodeError as e:
                logging.exception(
                    f"Error decoding response: {response} for problem_id: {problem_id} and submission_id: {submission_id}"
                )
                annotator.write_error(problem_id, submission_id, response, str(e))
            else:
                data["problem_id"] = problem_id
                data["submission_id"] = submission_id
                data["problem_description"] = chatgpt_response["step1"]
                data["quality_assessment"] = chatgpt_response["step2"]
                data["quality_score"] = chatgpt_response["step3"]
                json.dump(data, f)
                f.write("\n")
                annotator.write_checkpoint(submission_id)


def write_errors(errors: list) -> None:
    with open("data/CodeQualData/errors.jsonl", "w") as f:
        for error in errors:
            data = dict(problem_id=error[0], submission_id=error[1], response=error[2])
            json.dump(data, f)
            f.write("\n")


if __name__ == "__main__":
    write_code_qual_data("data/CodeQualData/all_data.jsonl")
