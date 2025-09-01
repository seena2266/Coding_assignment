import os
import json
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Set a device for training (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

def create_formatted_dataset(dataset_path: str, num_samples: int = 100):
    """
    Loads and formats the DriveLM dataset for supervised fine-tuning.

    Args:
        dataset_path (str): The path to the DriveLM dataset JSON file.
        num_samples (int): The number of samples to use for the subset.
                           A small number is chosen for demonstration.

    Returns:
        datasets.Dataset: A Hugging Face Dataset object formatted for fine-tuning.
    """
    try:
        # Load data from the specified JSON file
        with open(dataset_path, 'r') as f:
            data = json.load(f)

        # We'll fine-tune on a small subset for demonstration purposes
        subset_data = data[:num_samples]

        formatted_examples = []
        for item in subset_data:
            # The DriveLM dataset contains 'conversations' with a reasoning chain
            # followed by the final answer. We format this into an instruction
            # and response structure for the model to learn from.
            # Example format: "Question: [question]\n\nReasoning: [reasoning]\n\nAnswer: [answer]"
            # We'll use the 'reasoning' as the core of our fine-tuning.
            
            # The structure of the data is a bit complex, we need to extract the
            # relevant text parts. The "reasoning" is key to the user's request.
            # We will construct a single prompt string from the available data.
            
            question = item.get("question", "")
            # Assume the reasoning is part of the answer for simplicity
            answer = item.get("answer", {}).get("answer_text", "")
            reasoning = item.get("reasoning", "")
            
            if question and answer and reasoning:
                formatted_text = (
                    f"### Instruction:\n{question}\n\n"
                    f"### Response:\n{reasoning}\n\n{answer}"
                )
                formatted_examples.append({"text": formatted_text})

        # Create a Hugging Face Dataset from the list of dictionaries
        dataset = Dataset.from_list(formatted_examples)
        print(f"Created dataset with {len(dataset)} formatted examples.")
        return dataset
    except Exception as e:
        print(f"Error loading or processing dataset: {e}")
        return None


def fine_tune_llm():
    """
    Main function to orchestrate the fine-tuning process.
    """
    print(f"Using device: {device}")
    
    # 1. Define model and tokenizer
    # Using a small model for a fast demonstration.
    model_name = "distilbert/distilgpt2"
    # Alternatively, for a larger model, you could use a quantized version
    # like "mistralai/Mistral-7B-v0.1" with a BitsAndBytesConfig
    # to load the model in 4-bit, which is not necessary for distilgpt2.
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Set the padding token to the end-of-sequence token
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # 2. Prepare the dataset
    # In a real-world scenario, you would download the dataset from Hugging Face
    # and use the `create_formatted_dataset` function. For this code snippet,
    # we'll assume a local JSON file for a simple example.
    # Note: Replace 'drivelm_subset.json' with your actual file path.
    # You would need to download the full dataset from Hugging Face:
    # `datasets.load_dataset("OpenDriveLab/DriveLM", "nu_v1_0")` and then
    # process the split you want (e.g., 'train')
    
    # Mocking a small dataset for this example, as the full dataset is large.
    # In a real-world use case, you would load the full dataset.
    sample_data = [
        {"question": "What is the primary action required?", "reasoning": "Based on the intersection and the approaching vehicles, the car should slow down to allow traffic to clear.", "answer": "Slow down."},
        {"question": "Describe the most important object in the scene.", "reasoning": "The traffic light is red, which is a critical signal for the ego vehicle's next action. Therefore, it is the most important object.", "answer": "The red traffic light."},
        {"question": "How should the ego vehicle react to the pedestrian?", "reasoning": "A pedestrian is crossing the road ahead. The safest and legal action is to stop and wait for them to pass completely.", "answer": "Stop and wait for the pedestrian to cross."},
        {"question": "What is the predicted path of the other vehicle?", "reasoning": "The vehicle in the adjacent lane has its left turn signal on, indicating its intention to turn at the upcoming intersection.", "answer": "It will turn left at the next intersection."},
    ]
    # Save the mock data to a temporary file
    temp_file_path = "data/drivelm/parsed_nuscenes_driveLM.json"
    with open(temp_file_path, "w") as f:
        json.dump(sample_data, f)
    
    # Now, load the mock dataset using the formatter
    train_dataset = create_formatted_dataset(temp_file_path, num_samples=len(sample_data))
    
    if train_dataset is None:
        return
    
    # 3. Configure LoRA
    # LoRA configures which parts of the model to adapt.
    peft_config = LoraConfig(
        r=16,  # The rank of the update matrices. Lower means more efficient.
        lora_alpha=32,  # A scaling factor for the LoRA adapter.
        lora_dropout=0.05,  # Dropout for the LoRA layers.
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # 4. Set up the Trainer and training arguments
    training_args = TrainingArguments(
        output_dir="./drivelm_finetuned",
        per_device_train_batch_size=2,
        num_train_epochs=1,
        learning_rate=2e-4,
        fp16=True, # Use FP16 for faster training and reduced memory
        logging_steps=10,
        optim="paged_adamw_32bit",
    )

    # Use SFTTrainer from TRL, which simplifies the process.
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,
        args=training_args,
    )
    
    # 5. Start training
    print("Starting fine-tuning...")
    trainer.train()
    print("Fine-tuning completed!")
    
    # 6. Save the trained model and tokenizer
    trainer.save_model("./drivelm_finetuned")
    tokenizer.save_pretrained("./drivelm_finetuned")
    print("Model and tokenizer saved to drivelm_finetuned/")

if __name__ == "__main__":
    # Ensure necessary packages are installed
    try:
        import datasets, transformers, peft, trl
        fine_tune_llm()
    except ImportError:
        print("Required libraries not found. Please install them first:")
        print("pip install transformers peft datasets trl accelerate bitsandbytes torch")
        
