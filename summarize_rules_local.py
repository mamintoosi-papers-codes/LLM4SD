import argparse
import os
import json
import time
import ollama

# Fixed summarization model
OLLAMA_SUMMARY_MODEL = "gemma3:27b"

def load_inference_rules(input_folder, dataset, model):
    """Load inferred knowledge rules from saved files."""
    input_path = os.path.join("inference_model_response", input_folder, dataset)
    input_file = os.path.join(input_path, f"{model}_inference_response.txt")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Error: Could not find inference rules file at {input_file}")

    with open(input_file, 'r') as f:
        rules = f.readlines()
    
    print(f"✅ Loaded {len(rules)} inferred rules from {input_file}")
    return rules

def summarize_rules_with_gemma(rules):
    """Use Gemma 3:27B for summarization."""
    formatted_prompt = (
        "You are an advanced AI specializing in scientific summarization. "
        "Please review the following inference rules, eliminate duplicates, and return a concise summary:\n\n"
        + "\n".join(rules) 
        + "\n\nProvide a refined list of the most critical scientific rules."
    )

    response = ollama.chat(model=OLLAMA_SUMMARY_MODEL, messages=[{"role": "user", "content": formatted_prompt.strip()}])
    summarized_rules = response["message"]["content"]
    
    print(f"✅ Summarized rules:\n{summarized_rules}")
    return summarized_rules

def main():
    print("🔹 Step 4: Summarizing Inference Rules Locally using Gemma 3:27B...")

    rules = load_inference_rules(args.input_model_folder, args.dataset, args.input_model_folder)
    summarized_rules = summarize_rules_with_gemma(rules)

    output_file_folder = os.path.join(args.output_folder, args.input_model_folder, args.dataset)
    os.makedirs(output_file_folder, exist_ok=True)
    output_file = os.path.join(output_file_folder, f"{args.input_model_folder}_summarized_rules.txt")

    with open(output_file, 'w') as f:
        f.write(summarized_rules)
    
    print(f"✅ Summarization completed! Results saved to: {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model_folder', type=str, default='inference_model_response', help='Folder containing inference results')
    parser.add_argument('--dataset', type=str, default='bbbp', help='Dataset/task name')
    parser.add_argument('--output_folder', type=str, default='summarized_inference_rules', help='Output folder for summarized rules')

    args = parser.parse_args()

    start = time.time()
    main()
    end = time.time()
    print(f"Summarization/Time elapsed: {end-start} seconds")
