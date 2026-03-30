from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """

        # raise NotImplementedError()
        messages = [
              {"role": "system", "content": "You are a precise unit conversion assistant. Show steps and end with exactly <answer>VALUE</answer>."},
              {"role": "user", "content": "How many seconds are there in 2 hours?"},
              {"role": "assistant", "content": "1 hour = 3600 seconds.\n2 * 3600 = 7200.\n<answer>7200</answer>"},
              {"role": "user", "content": "Convert 5 kilometers to meters."},
              {"role": "assistant", "content": "1 km = 1000 m.\n5 * 1000 = 5000.\n<answer>5000</answer>"},
              {"role": "user", "content": "Convert 3.7 liters to milliliters."},
              {"role": "assistant", "content": "1 L = 1000 mL.\n3.7 * 1000 = 3700.\n<answer>3700</answer>"},
              {"role": "user", "content": question},
          ]
        
        # Apply the chat template to format the messages
        prompt = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False
        )
        
        return prompt


def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
