from concurrent import futures

from codeQual import logging
from codeQual.codenet import CodeNetPython
from codeQual.gpt import ChatGPT


class Annotator:
    def __init__(self, codenet_python: CodeNetPython, chatgpt: ChatGPT):
        self.codenet_python = codenet_python
        self.chatgpt = chatgpt
        self.usr_msgs = []
        for submission in self.codenet_python.next_submission():
            problem_id = submission.split("/")[-2]
            submission_id = submission.split("/")[-1].split(".")[0]
            code = open(submission).read()
            user_content = "```\n" + code + "\n```\n"
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
        with open("data/CodeQualData/checkpoint.txt", "a") as f:
            f.write(submission_id + "\n")

    def write_error(
        self, problem_id: str, submission_id: str, response: str, err_msg: str
    ) -> None:
        with open(f"data/CodeQualData/errors/{submission_id}.jsonl", "w") as f:
            f.write(problem_id + " " + submission_id + "\n")
            f.write(response + "\n")
            f.write(err_msg + "\n")
