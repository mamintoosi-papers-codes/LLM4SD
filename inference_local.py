import argparse
import os
import json
import time
import ollama

# Model name mapping for Ollama compatibility
MODEL_MAPPING = {
    "mistral-7b": "mistral:7b",
    "falcon-7b": "falcon:7b",
    "falcon-40b": "falcon:40b",
    "galactica-6.7b": "galactica:6.7b",
    "galactica-30b": "galactica:30b"
}

def get_system_prompt():
    """Define system prompt format based on model type."""
    model = args.model.lower()
    if model in ['falcon-7b', 'falcon-40b', 'mistral-7b']:
        system_prompt = "{instruction}\n"
    elif model in ["galactica-6.7b", "galactica-30b"]:
        system_prompt = ("Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.\n\n"
                         "### Instruction:\n{instruction}\n\n### Response:\n")
    else:
        raise NotImplementedError(f"No system prompt setting for the model: {model}.")
    return system_prompt

def get_inference_prompt():
    """Load inference prompt from JSON file."""
    prompt_file = os.path.join(args.input_folder, args.input_file)
    with open(prompt_file, 'r') as f:
        prompt_dict = json.load(f)
    dataset_key = args.dataset # + "_small" M. Amintoosi
    print(f'Extracting {dataset_key} dataset inference prompt...')
    return prompt_dict[dataset_key]

def get_model_response(model, instruction):
    """Use Ollama's local model to generate responses."""
    print(f"Querying Ollama Model: {model}")
    # Apply model name mapping
    ollama_model = MODEL_MAPPING.get(model, model)

    response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": instruction.strip()}])

    generated_text = response["message"]["content"]
    
    print(generated_text)
    return generated_text

def main():
    system_prompt = get_system_prompt()
    inference_prompt = get_inference_prompt()
    formatted_prompt = system_prompt.format_map({'instruction': inference_prompt.strip()})

    print("Starting inference using Ollama's local model...")
    response_text = get_model_response(args.model, formatted_prompt)

    # Save response to file
    output_file_folder = os.path.join(args.output_folder, args.model, args.dataset)
    os.makedirs(output_file_folder, exist_ok=True)
    output_file = os.path.join(output_file_folder, f"{args.model}_inference_response.txt")
    
    with open(output_file, 'w') as f:
        f.write(response_text)
    
    print(f"✅ Inference completed! Response saved to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbbp', help='Dataset/task name')
    parser.add_argument('--model', type=str, default='falcon-7b', help='LLM model name (Ollama version)')
    parser.add_argument('--input_folder', type=str, default='prompt_file', help='Inference prompt file folder')
    parser.add_argument('--input_file', type=str, default='inference_prompt.json', help='Inference prompt JSON file')
    parser.add_argument('--output_folder', type=str, default='inference_model_response', help='Output folder')

    args = parser.parse_args()

    start = time.time()
    main()
    end = time.time()
    print(f"Inference/Time elapsed: {end-start} seconds")
