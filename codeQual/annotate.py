import json
import os
from concurrent import futures
from typing import Any

from codeQual import logging
from codeQual.codenet import CodeNetPython
from codeQual.gpt import ChatGPT


class Annotator:
    def __init__(
        self,
        codenet_python: CodeNetPython,
        chatgpt: ChatGPT,
        output_dir: str,
        skip_problem_ids: list[str] = [],
    ):
        self.codenet_python = codenet_python
        self.chatgpt = chatgpt
        self.usr_msgs = []
        self.output_dir = output_dir
        for submission in self.codenet_python.next_submission():
            problem_id = submission.split("/")[-2]
            if problem_id in skip_problem_ids:
                continue
            submission_id = submission.split("/")[-1].split(".")[0]
            code = open(submission).read()
            user_content = "```\n" + code + "\n```\n"
            user_content += (
                '"""\n' + self.codenet_python.get_problem_desc(problem_id) + '\n"""\n'
            )
            user_msg = [{"role": "user", "content": user_content}]
            self.usr_msgs.append((problem_id, submission_id, user_msg))

        with open("data/CodeQualData/checkpoint.txt", "r") as f:
            self.annotated = f.read().splitlines()

    def annotate(self):
        for m in self.usr_msgs:
            if m[1] in self.annotated:
                continue
            logging.info(f"Annotating {m[0]} - {m[1]}")
            resp = self.chatgpt.get_response(m[2])
            yield m[0], m[1], resp

    def write_checkpoint(self, submission_id: str) -> None:
        checkpoint = os.path.join(self.output_dir, "submissions.txt")
        with open(checkpoint, "a") as f:
            f.write(submission_id + "\n")

    def write_error(
        self, problem_id: str, submission_id: str, response: str, err_msg: str
    ) -> None:
        directory = os.path.join(self.output_dir, problem_id)
        with open(f"{directory}/{submission_id}", "w") as f:
            f.write(problem_id + " " + submission_id + "\n")
            f.write(response + "\n")
            f.write(err_msg + "\n")

    def write_data(
        self,
        problem_id: str,
        submission_id: str,
        chatgpt_response: Any,
    ) -> None:
        directory = os.path.join(self.output_dir, problem_id)
        if not os.path.exists(directory):
            os.makedirs(directory)

        data = {}
        with open(f"{directory}/data.jsonl", "a") as f:
            data["problem_id"] = problem_id
            data["submission_id"] = submission_id
            data["problem_description"] = self.codenet_python.get_problem_desc(
                problem_id
            )
            data["quality_assessment"] = chatgpt_response["step1"]
            data["quality_score"] = chatgpt_response["step2"]
            json.dump(data, f)
            f.write("\n")
