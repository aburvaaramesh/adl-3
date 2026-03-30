from .base_llm import BaseLLM
from .data import Dataset, benchmark


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "sft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def tokenize(tokenizer, question: str, answer: str):
    """
    Tokenize a data element.
    We first append the <EOS> token to the question / answer pair.
    Then we tokenize and construct the ground truth `labels`.
    `labels[i] == -100` for the question or masked out parts, since we only want to supervise
    the answer.
    """
    full_text = f"{question} {answer}{tokenizer.eos_token}"

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token
    full = tokenizer(full_text, padding="max_length", truncation=True, max_length=128)

    input_ids = full["input_ids"]
    question_len = len(tokenizer(question)["input_ids"])

    # Create labels: mask out the prompt part
    labels = [-100] * question_len + input_ids[question_len:]

    for i in range(len(labels)):
        if full["attention_mask"][i] == 0:
            labels[i] = -100

    full["labels"] = labels
    return full


def format_example(prompt: str, answer: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """
    # raise NotImplementedError()
    answer_float = float(answer)
    formatted_answer = f"<answer>{answer_float:.1f}</answer>" if answer_float == int(answer_float) else f"<answer>{answer_float:.2f}</answer>"

    return {
        "question": prompt,
        "answer": formatted_answer
    }


class TokenizedDataset:
    def __init__(self, tokenizer, data: Dataset, format_fn):
        """
        Use the
        - BaseLLM.tokenizer
        - Dataset
        - format_fn which converts a data element into a dict with entries
          - question: str
          - answer: str
        """
        self.format_fn = format_fn
        self.tokenizer = tokenizer
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        formated_data = self.format_fn(*self.data[idx])
        return tokenize(self.tokenizer, **formated_data)


def train_model(
    output_dir: str,
    **kwargs,
):
    # raise NotImplementedError()
    from pathlib import Path
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import Trainer, TrainingArguments

    # Load base model and dataset
    llm = BaseLLM()
    train_dataset = Dataset("train")

    # Create LoRA config with reasonable parameters for a 360M model
    lora_config = LoraConfig(
        r=16,  # Rank - keeps model size reasonable
        lora_alpha=64,  # 4x the rank
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        lora_dropout=0.1,
    )

    # Apply LoRA to the model
    llm.model = get_peft_model(llm.model, lora_config)

    # Enable input requires grad for gradient checkpointing
    if llm.device == "cuda":
        llm.model.enable_input_require_grads()

    # Create tokenized dataset
    tokenized_dataset = TokenizedDataset(
        llm.tokenizer,
        train_dataset,
        format_example
    )

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        learning_rate=1e-4,
        gradient_checkpointing=True,
        optim="adamw_8bit",
        report_to="tensorboard",
        logging_steps=10,
        save_strategy="epoch",
    )

    # Create trainer and train
    trainer = Trainer(
        model=llm.model,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    trainer.train()

    # Save the model - save only the LoRA weights
    llm.model.save_pretrained(output_dir)
    test_model(output_dir)


def test_model(ckpt_path: str):
    testset = Dataset("valid")
    llm = BaseLLM()

    # Load the model with LoRA adapters
    from peft import PeftModel

    llm.model = PeftModel.from_pretrained(llm.model, ckpt_path).to(llm.device)

    benchmark_result = benchmark(llm, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
