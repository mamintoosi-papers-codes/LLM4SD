import requests
from pprint import pprint

class ChemistryModelEvaluator:
    def __init__(self, base_url="http://localhost:11434/v1"):
        self.base_url = base_url
        
    def evaluate_models(self, models):
        """Compare different models on BBBP rule generation."""
        prompt = {
            "system": "You are an expert computational chemist. Provide 20-30 concise, scientifically valid rules to predict blood-brain barrier permeability (BBBP) using these guidelines:\n"
                     "1. Each rule should specify a molecular feature and its effect (e.g., 'MW < 400 Da increases permeability')\n"
                     "2. Include both structural (e.g., functional groups) and physicochemical properties (e.g., logP)\n"
                     "3. Format as numbered list with brief explanations",
            "user": "Generate the rules now."
        }
        
        results = {}
        for model in models:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": prompt["system"]},
                        {"role": "user", "content": prompt["user"]}
                    ],
                    "temperature": 0.3  # Reduce randomness for scientific accuracy
                }
            ).json()
            results[model] = response["choices"][0]["message"]["content"]
        
        return results

# Models to compare (ordered by expected performance)
MODELS = [
    "gemma3:27b",
    "mistral:7b",
    "t1c/deepseek-math-7b-rl:Q4",
    "falcon:7b"
]

evaluator = ChemistryModelEvaluator()
results = evaluator.evaluate_models(MODELS)

print("\n" + "="*50 + "\nMODEL COMPARISON: BBBP PREDICTION RULES\n" + "="*50)
for model, response in results.items():
    print(f"\n🔥 {model.upper()} RESULTS:\n")
    print(response[:1000] + "...")  # Print first 1000 chars to compare quality
    print("\n" + "-"*80)