import json
import os


class CodeNetPython:
    def __init__(self, path: str):
        self.path = path
        self.problem_desc = {}
        with open(
            os.path.join(
                os.path.dirname(self.path), "py800_metadata_problem_desc.jsonl"
            ),
            "r",
        ) as f:
            for line in f:
                data = json.loads(line)
                self.problem_desc[data["problem_id"]] = data["problem"]

    def next_submission(self):
        """
        Walks through the path of the dataset and returns the next submission
        """
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".py"):
                    yield os.path.join(root, file)

    def get_problem_desc(self, problem_id: str) -> str:
        return self.problem_desc[problem_id]
