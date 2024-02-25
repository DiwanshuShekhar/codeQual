import os


class CodeNetPython:
    def __init__(self, path: str):
        self.path = path

    def next_submission(self):
        """
        Walks through the path of the dataset and returns the next submission
        """
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith(".py"):
                    yield os.path.join(root, file)
