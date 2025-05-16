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

def get_synthesize_prompt():
    """
    Read prompt json file to load prompt for the task, return a task name list and a prompt list
    """
    prompt_file = os.path.join(args.input_folder, args.input_file)
    pk_prompt_list = []
    task_list = []
    with open(prompt_file, 'r') as f:
        prompt_dict = json.load(f)

    dataset_key = args.dataset + "_small"  # Adjusted for Ollama compatibility
    print(f'Extracting {dataset_key} dataset prior knowledge prompt ....')

    if not args.subtask:
        task_list.append(args.dataset)
        pk_prompt_list.append(prompt_dict[dataset_key])
    elif args.subtask:
        print(f"Extracting {args.subtask} task prior knowledge prompt ....")
        task_list.append(args.dataset + "_" + args.subtask)
        pk_prompt_list.append(prompt_dict[dataset_key][args.subtask])
    else:
        raise NotImplementedError(f"""No prior knowledge prompt for task {args.dataset}.""")

    return task_list, pk_prompt_list

def get_pk_model_response(model, pk_prompt_list):
    """
    Use Ollama's local model for generating responses.
    """
    # Apply model name mapping
    ollama_model = MODEL_MAPPING.get(model, model)

    response_list = []
    for pk_prompt in pk_prompt_list:
        print(f"Querying Ollama Model: {ollama_model}")
        
        response = ollama.chat(model=ollama_model, messages=[{"role": "user", "content": pk_prompt.strip()}])
        generated_text = response["message"]["content"]
        
        print(generated_text)
        response_list.append(generated_text)

    return response_list

def main():
    task_list, pk_prompt_list = get_synthesize_prompt()
    response_list = get_pk_model_response(args.model, pk_prompt_list)
    
    output_file_folder = os.path.join(args.output_folder, args.model, args.dataset)
    if args.subtask:
        subtask_name = "_" + args.subtask
    else:
        subtask_name = ''
    
    output_file = os.path.join(output_file_folder, f'{args.model}{subtask_name}_pk_response.txt')
    os.makedirs(output_file_folder, exist_ok=True)
    
    with open(output_file, 'w') as f:
        for i in range(len(task_list)):
            f.write(f'Task Name: {task_list[i]}\n')
            f.write('Response from Model:\n')
            f.write(response_list[i])
            f.write("\n\n================================\n\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='bbbp', help='Dataset/task name')
    parser.add_argument('--subtask', type=str, default='', help='Subtask of tox21/sider dataset')
    parser.add_argument('--model', type=str, default='falcon-7b', help='LLM model name (Ollama version)')
    parser.add_argument('--input_folder', type=str, default='prompt_file', help='Synthesize prompt file folder')
    parser.add_argument('--input_file', type=str, default='synthesize_prompt.json', help='Synthesize prompt JSON file')
    parser.add_argument('--output_folder', type=str, default='synthesize_model_response', help='Output folder')

    args = parser.parse_args()

    start = time.time()
    main()
    end = time.time()
    print(f"Synthesize/Time elapsed: {end-start} seconds")
