
import pandas as pd
import os
import json
#import torch # Not needed
#from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig # Not needed
#from transformers import LlamaTokenizer, LlamaForCausalLM # Not needed
import ollama # Added
import argparse
from tqdm import tqdm

def load_dataset(dataset_name):
    df = pd.read_csv(dataset_name)
    df_str = df.iloc[:].to_string(index=False, header=False)
    df_str_list = df_str.split('\n')
    df_str_list = [f.strip() for f in df_str_list]
    return df_str_list

def get_inference_prompt():
    prompt_file = os.path.join(args.prompt_folder, args.prompt_file)
    with open(prompt_file, 'r') as f:
        prompt_dict = json.load(f)
    print(f'Extracting {args.dataset} {args.subtask} task(s) Inference prompt ....')
    if args.dataset in list(prompt_dict.keys()) and not args.subtask:
        prompt = prompt_dict[args.dataset]
    elif args.subtask:
        prompt = prompt_dict[args.dataset][args.subtask]
    else:
        raise NotImplementedError(f"""No data knowledge prompt for task {args.dataset} {args.subtask}.""")
    return prompt

def get_token_limit(model, for_response=False):
    """Returns the token limitation of provided model"""
    model = model.lower()
    if for_response:  # For get response
        if model in ['falcon-7b', 'falcon-40b', "galactica-6.7b", "galactica-30b", "chemdfm"]:
            num_tokens_limit = 2048
        elif model in ['chemllm-7b']:
            num_tokens_limit = 4096
    else:  # For split input list
        if model in ['falcon-7b', 'falcon-40b', "galactica-6.7b", "galactica-30b", "chemdfm"]:
            num_tokens_limit = round(2048*3/4)  # 1/4 part for the response, 512 tokens
        elif model in ['chemllm-7b']:
            num_tokens_limit = round(4096*3/4)
        else:
            raise NotImplementedError(f"""get_token_limit() is not implemented for model {model}.""")
    return num_tokens_limit

def get_system_prompt():
    # prompt format depends on the LLM, you can add the system prompt here
    model = args.model.lower()
    if model in ['falcon-7b', 'falcon-40b']:
        system_prompt = "{instruction}\n"
    elif model in ["galactica-6.7b", "galactica-30b", "chemllm-7b"]:
        system_prompt = ("Below is an instruction that describes a task. "
                         "Write a response that appropriately completes the request.\n\n"
                         "### Instruction:\n{instruction}\n\n### Response:\n")
    elif model in ["chemdfm"]:
        system_prompt = "Human: {instruction}\nAssistant:"
    else:
        raise NotImplementedError(f"""No system prompt setting for the model: {model} .""")
    return system_prompt

def split_smile_list(smile_content_list, dk_prompt, tokenizer, list_num):
    """
    Each list can be directly fed into the model
    """
    token_limitation = get_token_limit(args.model)  # Get input token limitation for current model
    system_prompt = get_system_prompt()
    all_smile_content = dk_prompt + '\n'+'\n'.join(smile_content_list)
    formatted_all_smile = system_prompt.format_map({'instruction': all_smile_content})
    token_num_all_smile = len(tokenizer.tokenize(formatted_all_smile))
    if token_num_all_smile > token_limitation:  # Need to do split
        list_of_smile_label_lists = []
        for _ in tqdm(range(list_num)):  # Generate request number of sub lists
            current_list = []
            cp_smile_content_list = copy.deepcopy(smile_content_list)
            current_prompt = system_prompt.format_map({'instruction': dk_prompt})  # only inference prompt, without smile&label
            current_prompt_len = len(tokenizer.tokenize(current_prompt))
            while current_prompt_len <= token_limitation:
                if cp_smile_content_list:
                    smile_label = random.choice(cp_smile_content_list)  # Randomly select an element
                    smile_prompt = dk_prompt + '\n' + '\n'.join(current_list) + '\n' + smile_label
                    smile_input_prompt = system_prompt.format_map({'instruction': smile_prompt})
                    current_prompt_len = len(tokenizer.tokenize(smile_input_prompt))
                    if current_prompt_len > token_limitation:
                        cp_smile_content_list.remove(smile_label)  # Maybe this smile string is too long, remove it and try to add another shorter one
                    else:
                        current_list.append(smile_label)
                        cp_smile_content_list.remove(smile_label)  # no duplicated smile string in one sub-list
                else:
                    break

            list_of_smile_label_lists.append(current_list)
    else:
        list_of_smile_label_lists = [[sc + '\n' for sc in smile_content_list]]
    return list_of_smile_label_lists

def get_hf_tokenizer_pipeline(model, is_8bit=False):
    """Return HF tokenizer for the model - now uses Ollama"""
    model = model.lower()

    # Ollama model names (adapt as needed)
    if model == 'falcon-7b':
        ollama_model = "falcon:7b"  # Ollama format
    elif model == 'falcon-40b':
        ollama_model = "falcon:40b"  #Adapt to your ollama installation
    elif model == "galactica-6.7b":
        ollama_model = "galactica:6.7b" #Adapt to your ollama installation
    elif model == "galactica-30b":
        ollama_model = "galactica:30b" #Adapt to your ollama installation
    elif model == "chemllm-7b":
        ollama_model = "chemllm:7b" #Adapt to your ollama installation
    elif model == "chemdfm":
        ollama_model = "chemdfm:13b" #Adapt to your ollama installation
    else:
        raise NotImplementedError(f"Cannot find Ollama model for {model}.")

    # In Ollama, we don't need to load a tokenizer or pipeline separately.
    # The Ollama server handles that internally.
    # We'll just return the model name for use in the inference function.
    return ollama_model # Return the Ollama model name

def get_synthesize_prompt():
    """
    Read prompt json file to load prompt for the task, return a task name list and a prompt list
    """
    prompt_file = os.path.join(args.input_folder, args.input_file)
    pk_prompt_list = []
    task_list = []
    with open(prompt_file, 'r') as f:
        prompt_dict = json.load(f)
    if args.model.lower() in ["falcon-7b", "galactica-6.7b", "chemllm-7b", "chemdfm"]:
        dataset_key = args.dataset + "_small"
    else:
        dataset_key = args.dataset + "_big"
    print(f'Extracting {dataset_key} dataset prior knowledge prompt ....')
    if not args.subtask:
        task_list.append(args.dataset)
        pk_prompt_list.append(prompt_dict[dataset_key])
    elif args.subtask:  # for tox21 and sider
        print(f"Extracting {args.subtask} task prior knowledge prompt ....")
        task_list.append(args.dataset + "_" + args.subtask)
        pk_prompt_list.append(prompt_dict[dataset_key][args.subtask])
    else:
        raise NotImplementedError(f"""No prior knowledge prompt for task {args.dataset}.""")
    return task_list, pk_prompt_list

def get_model_response(model, list_of_smile_label_lists, dk_prompt):
    """Get response from the model using Ollama."""
    input_list = [dk_prompt + '\n' + '\n'.join(s) for s in list_of_smile_label_lists]
    response_list = []
    for smile_label in input_list:
        try:
            response = ollama.chat(model=model, messages=[{"role": "user", "content": smile_label}])
            generated_text = response["message"]["content"]
            response_list.append(generated_text)
        except Exception as e:
            print(f"Error generating response: {e}")
            response_list.append("Error during generation.")  # Handle errors gracefully
    return response_list

def main():
    """Main function to load data, generate responses, and save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_folder', type=str, default='prompt_file', help='Prompt file folder')
    parser.add_argument('--prompt_file', type=str, default='inference_prompt.json', help='Inference prompt json file')
    parser.add_argument('--input_folder', type=str, default='scaffold_datasets', help="load training dataset")
    parser.add_argument('--output_folder', type=str, default='inference_model_response')
    parser.add_argument('--dataset', type=str, default='bbbp', help='dataset name')
    parser.add_argument('--subtask', type=str, default='', help='subtask of tox21/sider dataset')
    parser.add_argument('--list_num', type=int, default=30, help='number of lists for model inference')
    parser.add_argument('--model', type=str, default="galactica-6.7b", help='model for data knowledge')
    args = parser.parse_args()

    # Load dataset and prompt
    file_folder = os.path.join(args.input_folder, args.dataset)
    train_file_path = os.path.join(file_folder, args.dataset + '_train.csv')
    smile_label_list = load_dataset(train_file_path)
    dk_prompt = get_inference_prompt()
    list_of_smile_label_lists = split_smile_list(smile_label_list, dk_prompt)

    # Generate responses
    output_file_folder = os.path.join(args.output_folder, args.model, args.dataset)
    if not os.path.exists(output_file_folder):
        os.makedirs(output_file_folder)
    output_file_name = f"{args.model}_{args.dataset}_dk_response_sample.txt"
    output_file = os.path.join(output_file_folder, output_file_name)
    print(f'Start getting response from model {args.model}....')
    response_list = get_model_response(args.model, list_of_smile_label_lists, dk_prompt)

    # Save responses
    with open(output_file, 'w') as f:
        for response in response_list:
            f.write(response)
            f.write("\n\n================================\n\n")

# Placeholder functions (replace with your actual implementations)
def load_dataset(file_path):
    """Load dataset from file."""
    # Replace with your actual dataset loading code
    return []

def get_inference_prompt():
    """Get inference prompt."""
    # Replace with your actual prompt loading code
    return ""

def split_smile_list(smile_label_list, dk_prompt):
    """Split smile label list."""
    # Replace with your actual splitting code
    return []

if __name__ == '__main__':
    main()