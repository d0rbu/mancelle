"""
AlpacaTask: yahma/alpaca-cleaned dataset for instruction-following SFT.
https://huggingface.co/datasets/yahma/alpaca-cleaned

~51K instruction-output pairs. Each row has:
  - instruction (str): the task instruction
  - input (str): optional additional context
  - output (str): the expected assistant response

When `input` is non-empty, user content is "{instruction}\n\n{input}".
Otherwise, user content is just the instruction.
"""

from datasets import load_dataset

from tasks.common import Task


class Alpaca(Task):
    """yahma/alpaca-cleaned dataset. ~51K instruction-output pairs."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.ds = load_dataset("yahma/alpaca-cleaned", split="train").shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self) -> int:
        """Return the total number of examples in the dataset."""
        return self.length

    def get_example(self, index: int) -> dict:
        """Return a conversation dict for the given index.

        Format: {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
        """
        row = self.ds[index]
        instruction = row["instruction"]
        input_text = row["input"]
        output = row["output"]

        # Precondition assertions
        assert isinstance(instruction, str) and len(instruction) > 0, (
            f"instruction must be a non-empty string, got {instruction!r}"
        )
        assert isinstance(output, str) and len(output) > 0, (
            f"output must be a non-empty string, got {output!r}"
        )

        # Build user content: concatenate instruction + input if input is non-empty
        if input_text and len(input_text.strip()) > 0:
            user_content = f"{instruction}\n\n{input_text}"
        else:
            user_content = instruction

        conversation = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": output},
            ]
        }
        return conversation
