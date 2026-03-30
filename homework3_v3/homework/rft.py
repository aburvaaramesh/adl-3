from .base_llm import BaseLLM
from .sft import test_model


def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def train_model(
    output_dir: str,
    **kwargs,
):
    # Reuse much of the SFT code here
    # raise NotImplementedError()
    # Reuse much of the SFT code here
    import json
    from pathlib import Path
    from peft import get_peft_model, LoraConfig, TaskType
    from transformers import Trainer, TrainingArguments
    from .sft import TokenizedDataset, tokenize
    from .base_llm import BaseLLM

    # Check if RFT data exists, if not generate it
    rft_data_path = Path(__file__).parent.parent / "data" / "rft.json"
    if not rft_data_path.exists():
        from .datagen import generate_dataset
        print("RFT data not found, generating...")
        generate_dataset(str(rft_data_path))

    # Load RFT data
    class RFTDataset:
        def __init__(self, json_path: str):
            with open(json_path) as f:
                self.data = json.load(f)
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            question, answer, reasoning = self.data[idx]
            # Return question and the full reasoning which includes <answer> tag
            return (question, reasoning)

    llm = BaseLLM()
    rft_dataset = RFTDataset(str(rft_data_path))

    # Create LoRA config - slightly larger for RFT
    lora_config = LoraConfig(
        r=24,  # Slightly larger rank for RFT
        lora_alpha=96,  # 4x the rank
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

    # Create tokenized dataset for RFT
    class RFTTokenizedDataset:
        def __init__(self, tokenizer, data):
            self.tokenizer = tokenizer
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            question, reasoning = self.data[idx]
            # For RFT, we want to train on the full reasoning + answer
            return tokenize(self.tokenizer, question, reasoning)

    tokenized_dataset = RFTTokenizedDataset(llm.tokenizer, rft_dataset)

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


if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
