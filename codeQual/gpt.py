from typing import Optional

from openai import OpenAI


class ChatGPT:
    def __init__(
        self,
        model: str,
        init_messages: list[dict],
        seed: int,
        temperature: float,
        max_tokens: int,
        stop: Optional[str] = None,
    ) -> None:
        self.model = model
        self.init_messages = init_messages
        self.seed = seed
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stop = stop

    def get_response(self, user_msg: list[dict]) -> str:
        self.init_messages.extend(user_msg[2])

        client = OpenAI()
        response = client.chat.completions.create(
            model=self.model,
            messages=self.init_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stop=self.stop,
            seed=self.seed,
        )
        return response.choices[0].message.content
