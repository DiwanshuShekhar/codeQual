from codeQual.codenet import CodeNetPython
from codeQual.gpt import ChatGPT


class Annotator:
    def __init__(self, codenet_python: CodeNetPython, chatgpt: ChatGPT):
        self.codenet_python = codenet_python
        self.chatgpt = chatgpt

    def annotate(self):
        for submission in self.codenet_python.next_submission():
            problem_id = submission.split("/")[-2]
            submission_id = submission.split("/")[-1].split(".")[0]
            code = open(submission).read()
            user_content = "```\n" + code + "\n```\n"
            user_msg = [{"role": "user", "content": user_content}]
            yield problem_id, submission_id, self.chatgpt.get_response(user_msg)
