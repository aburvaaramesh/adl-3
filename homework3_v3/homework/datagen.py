def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    # raise NotImplementedError()
    import json
    from pathlib import Path
    from tqdm import tqdm
    from .cot import CoTModel
    from .data import Dataset, is_answer_valid

    # Load dataset
    train_dataset = Dataset("train")

    # Load CoTModel with the larger 1.7B-Instruct model for better quality
    print("Loading CoT model with 1.7B-Instruct...")
    model = CoTModel(checkpoint="HuggingFaceTB/SmolLM2-1.7B-Instruct")

    collected_data = []
    failed_count = 0

    # Process each question in the dataset
    for idx in tqdm(range(len(train_dataset)), desc="Generating RFT data"):
        question, correct_answer = train_dataset[idx]
        
        # Generate multiple completions
        prompt = model.format_prompt(question)
        
        try:
            # Generate multiple sequences for this question
            generations = model.batched_generate(
                [prompt],
                num_return_sequences=oversample,
                temperature=temperature
            )
            
            # Handle return type - batched_generate returns either list[str] or list[list[str]]
            if isinstance(generations[0], list):
                generations = generations[0]
            
            # Find a correct answer among the generations
            found_correct = False
            for generation in generations:
                # Parse the answer from the generation
                answer = model.parse_answer(generation)
                
                # Check if this answer is correct (within 10% tolerance as recommended)
                if not (answer != answer):  # Check it's not NaN
                    if is_answer_valid(answer, correct_answer, relative_tolerance=0.1):
                        # Store the data point: [question, correct_answer, reasoning_with_answer]
                        collected_data.append([
                            question,
                            float(correct_answer),
                            generation.strip()
                        ])
                        found_correct = True
                        break  # Found a correct answer, move to next question
            
            if not found_correct:
                failed_count += 1
        
        except Exception as e:
            # Skip on error
            failed_count += 1
            continue

    # Save the collected data
    output_path = Path(output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(collected_data, f, indent=2)

    success_rate = len(collected_data) / len(train_dataset) if len(train_dataset) > 0 else 0
    print(f"\n=== RFT Data Generation Complete ===")
    print(f"Generated {len(collected_data)} RFT data points")
    print(f"Success rate: {success_rate*100:.1f}%")
    print(f"Failed/skipped: {failed_count}")
    print(f"Saved to {output_json}")


if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
