# Complete Benchmarking Guide for Causal Reasoning Models

## Benchmarking Tasks Overview

This guide covers evaluation for 5 benchmarking tasks:
1. **Standard Gemma-270m-it** (baseline, no fine-tuning)
2. **Fine-tuned Transitivity** (standard loss)
3. **Fine-tuned D-separation** (standard loss)  
4. **Semantic Transitivity V4** (with semantic loss)
5. **Semantic D-separation V2** (with semantic loss)

## Setup Instructions for Colab Pro

### Cell 1: Environment Setup
```python
# Install required packages
!pip install -q transformers datasets torch accelerate
!pip install -q unsloth
!pip uninstall unsloth -y && pip install --upgrade --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

import torch
import json
import os
import numpy as np
from tqdm import tqdm
from unsloth import FastLanguageModel

print("Environment setup complete!")
```

### Cell 2: Mount Google Drive and Setup Paths
```python
# Mount Google Drive to access saved models
from google.colab import drive
drive.mount('/content/drive')

# Define model paths
MODEL_PATHS = {
    "standard_gemma": "unsloth/gemma-3-270m-it-bnb-4bit",
    "transitivity_baseline": "/content/drive/MyDrive/gemma_transitivity_models/transitivity_v1_merged",
    "dseparation_baseline": "/content/drive/MyDrive/gemma_dseparation_models/dseparation_v1_merged",
    "transitivity_semantic_v4": "/content/drive/MyDrive/gemma_semantic_models/transitivity/semantic_v4_merged",
    "dseparation_semantic": "/content/drive/MyDrive/gemma_semantic_models/dseparation/semantic_v2_merged" 
}

# Data path
DATA_PATH = "/content/drive/MyDrive/causal_data"  # Update with your path

print("Available models:")
for name, path in MODEL_PATHS.items():
    print(f"  {name}: {path}")
```

### Cell 3: Data Loading Functions
```python
def load_jsonl(file_path):
    """Load JSONL file"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Verify data files exist
test_files = [
    "length_eval.jsonl",
    "branching_eval.jsonl",
    "reversed_eval.jsonl",
    "shuffled_eval.jsonl",
    "long_names_eval.jsonl",
]

print("Checking for evaluation files...")
for test_file in test_files:
    path = f"{DATA_PATH}/eval/{test_file}"
    if os.path.exists(path):
        print(f"  ✓ {test_file}")
    else:
        print(f"  ✗ {test_file} - NOT FOUND")
```

### Cell 4: Model Selection and Loading
```python
# SELECT WHICH MODEL TO BENCHMARK
# Change this variable to test different models
MODEL_TO_TEST = "standard_gemma"  # Options: "standard_gemma", "transitivity_baseline", "dseparation_baseline", 
                                   #          "transitivity_semantic_v4", "dseparation_semantic"

print(f"\n{'='*60}")
print(f"LOADING MODEL: {MODEL_TO_TEST}")
print("="*60)

# Load selected model
model_path = MODEL_PATHS[MODEL_TO_TEST]

if MODEL_TO_TEST == "standard_gemma":
    # Load from HuggingFace
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=512,
        dtype=torch.float32,
        load_in_4bit=True,
    )
    print("Loaded standard Gemma-270M-IT from HuggingFace")
else:
    # Load from local path
    if os.path.exists(model_path):
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=512,
            dtype=torch.float32,
            load_in_4bit=True,
        )
        print(f"Loaded fine-tuned model from: {model_path}")
    else:
        print(f"ERROR: Model not found at {model_path}")
        print("Please ensure the model is saved in Google Drive")

print(f"Model loaded successfully!")
print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
```

### Cell 5: Core Evaluation Functions
```python
def predict_yes_no_logits(model, tokenizer, premise, hypothesis, device="cuda"):
    """
    Fast evaluation using logits comparison (no generation)
    """
    prompt = f"""<start_of_turn>user
Given the following causal relationships:
{premise}

{hypothesis}
Please answer with only 'Yes' or 'No'.<end_of_turn>
<start_of_turn>model
"""
    
    # Tokenize
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids=inputs["input_ids"], 
                        attention_mask=inputs["attention_mask"])
        logits = outputs.logits
    
    # Get last position logits
    last_pos = inputs["attention_mask"].sum(dim=1) - 1
    logits_next = logits[0, last_pos.item(), :]
    
    # Compare Yes vs No logits
    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]
    no_id = tokenizer.encode("No", add_special_tokens=False)[0]
    
    yes_logit = logits_next[yes_id].item()
    no_logit = logits_next[no_id].item()
    
    return "Yes" if yes_logit >= no_logit else "No"

def predict_yes_no_generate(model, tokenizer, premise, hypothesis, device="cuda"):
    """
    Alternative: Using generation (slower but sometimes more stable)
    """
    prompt = f"""<start_of_turn>user
Given the following causal relationships:
{premise}

{hypothesis}
Please answer with only 'Yes' or 'No'.<end_of_turn>
<start_of_turn>model"""
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract answer
    if "Yes" in response:
        return "Yes"
    elif "No" in response:
        return "No"
    else:
        return "Unknown"
```

### Cell 6: Comprehensive Evaluation Function
```python
def evaluate_model_comprehensive(model, tokenizer, test_file, max_samples=200, 
                                 method="logits", device="cuda", verbose=True):
    """
    Comprehensive evaluation with detailed metrics
    """
    # Load data
    test_path = f"{DATA_PATH}/eval/{test_file}"
    
    if not os.path.exists(test_path):
        print(f"File not found: {test_path}")
        return None
    
    test_data = load_jsonl(test_path)
    
    # Sample if needed
    if len(test_data) > max_samples:
        import random
        random.seed(42)
        test_data = random.sample(test_data, max_samples)
    
    # Initialize metrics
    correct = 0
    yes_count = 0
    no_count = 0
    true_yes = 0
    true_no = 0
    tp = 0  # True Positives (correctly predicted Yes)
    tn = 0  # True Negatives (correctly predicted No)
    fp = 0  # False Positives (incorrectly predicted Yes)
    fn = 0  # False Negatives (incorrectly predicted No)
    
    # Evaluation loop
    if verbose:
        print(f"Evaluating on {test_file} ({len(test_data)} samples)...")
    
    for i, item in enumerate(tqdm(test_data, disable=not verbose)):
        # Get prediction
        if method == "logits":
            pred = predict_yes_no_logits(model, tokenizer, 
                                        item["premise"], item["hypothesis"], device)
        else:
            pred = predict_yes_no_generate(model, tokenizer, 
                                         item["premise"], item["hypothesis"], device)
        
        # Update counts
        if pred == "Yes":
            yes_count += 1
        else:
            no_count += 1
        
        if item["label"] == "Yes":
            true_yes += 1
        else:
            true_no += 1
        
        # Update accuracy metrics
        if pred == item["label"]:
            correct += 1
            if pred == "Yes":
                tp += 1
            else:
                tn += 1
        else:
            if pred == "Yes":
                fp += 1
            else:
                fn += 1
    
    # Calculate metrics
    accuracy = correct / len(test_data) * 100
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    if verbose:
        print(f"\n{test_file} Results:")
        print(f"  Accuracy: {accuracy:.1f}% ({correct}/{len(test_data)})")
        print(f"  Precision: {precision:.1f}%")
        print(f"  Recall: {recall:.1f}%")
        print(f"  F1 Score: {f1:.1f}%")
        print(f"  Predictions - Yes: {yes_count} ({yes_count/len(test_data)*100:.1f}%)")
        print(f"  Predictions - No: {no_count} ({no_count/len(test_data)*100:.1f}%)")
        print(f"  True Labels - Yes: {true_yes} ({true_yes/len(test_data)*100:.1f}%)")
        print(f"  True Labels - No: {true_no} ({true_no/len(test_data)*100:.1f}%)")
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_predictions": yes_count,
        "no_predictions": no_count,
        "confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    }
```

### Cell 7: Run Full Evaluation Suite
```python
print("="*60)
print(f"BENCHMARKING: {MODEL_TO_TEST}")
print("="*60)
print(f"Model Path: {MODEL_PATHS[MODEL_TO_TEST]}")
print(f"Device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU'}")
print("="*60)

all_results = {}

for test_file in test_files:
    print(f"\n{'='*60}")
    print(f"Testing on: {test_file}")
    print("="*60)
    
    results = evaluate_model_comprehensive(
        model, tokenizer,
        test_file,
        max_samples=200,
        method="logits",  # Use "generate" if logits method has issues
        device="cuda",
        verbose=True
    )
    
    if results:
        test_name = test_file.replace("_eval.jsonl", "")
        all_results[test_name] = results
```

### Cell 8: Summary and Baseline Comparison
```python
# Baseline results for comparison
baseline_results = {
    'length': 86.0,
    'branching': 6.0,
    'reversed': 96.0,
    'shuffled': 85.0,
    'long_names': 96.0
}

# Known best results (V4 semantic)
best_semantic_results = {
    'length': 99.5,
    'branching': 61.5,
    'reversed': 97.0,
    'shuffled': 95.0,
    'long_names': 100.0
}

print("\n" + "="*60)
print(f"EVALUATION SUMMARY - {MODEL_TO_TEST}")
print("="*60)

# Summary table
print("\nDetailed Results:")
print("-"*70)
print(f"{'Test Type':<15} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10}")
print("-"*70)

for test_name, results in all_results.items():
    print(f"{test_name:<15} {results['accuracy']:>8.1f}% "
          f"{results['precision']:>8.1f}% "
          f"{results['recall']:>8.1f}% "
          f"{results['f1']:>8.1f}%")

# Calculate average
avg_accuracy = sum(r['accuracy'] for r in all_results.values()) / len(all_results)
avg_f1 = sum(r['f1'] for r in all_results.values()) / len(all_results)

print("-"*70)
print(f"{'AVERAGE':<15} {avg_accuracy:>8.1f}% {'':>10} {'':>10} {avg_f1:>8.1f}%")

# Multi-way comparison
print("\n" + "="*60)
print("MULTI-MODEL COMPARISON")
print("="*60)
print(f"{'Test Type':<15} {'Baseline':<12} {'Current':<12} {'Best V4':<12} {'vs Baseline':<15}")
print("-"*85)

for test_name in baseline_results.keys():
    baseline_acc = baseline_results[test_name]
    current_acc = all_results[test_name]['accuracy'] if test_name in all_results else 0
    best_acc = best_semantic_results[test_name]
    improvement = current_acc - baseline_acc
    
    print(f"{test_name:<15} {baseline_acc:>10.1f}% {current_acc:>10.1f}% "
          f"{best_acc:>10.1f}% {improvement:>+12.1f}%")

avg_baseline = sum(baseline_results.values()) / len(baseline_results)
avg_best = sum(best_semantic_results.values()) / len(best_semantic_results)
print("-"*85)
print(f"{'AVERAGE':<15} {avg_baseline:>10.1f}% {avg_accuracy:>10.1f}% "
      f"{avg_best:>10.1f}% {avg_accuracy-avg_baseline:>+12.1f}%")
```

### Cell 9: Export Results with Model Info
```python
import json
from datetime import datetime

# Create comprehensive results
results_export = {
    "timestamp": datetime.now().isoformat(),
    "model_tested": MODEL_TO_TEST,
    "model_path": MODEL_PATHS[MODEL_TO_TEST],
    "evaluation_results": all_results,
    "comparison": {
        test: {
            "baseline": baseline_results[test],
            "current_model": all_results[test]['accuracy'] if test in all_results else 0,
            "best_semantic_v4": best_semantic_results[test],
            "improvement_vs_baseline": (all_results[test]['accuracy'] - baseline_results[test]) if test in all_results else 0
        }
        for test in baseline_results.keys()
    },
    "average_metrics": {
        "current_accuracy": avg_accuracy,
        "current_f1": avg_f1,
        "baseline_avg": avg_baseline,
        "best_v4_avg": avg_best,
        "improvement_vs_baseline": avg_accuracy - avg_baseline
    }
}

# Save filename based on model tested
filename = f"benchmark_{MODEL_TO_TEST}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(filename, 'w') as f:
    json.dump(results_export, f, indent=2)

print(f"\nResults saved to {filename}")

# Display download link in Colab
from google.colab import files
files.download(filename)
```

### Cell 10: Quick Test Across All Models (Optional)
```python
# This cell runs a quick test on all available models
# Useful for getting a quick overview

print("="*60)
print("QUICK BENCHMARK - ALL MODELS")
print("="*60)

quick_results = {}

for model_name in ["standard_gemma", "transitivity_baseline", "transitivity_semantic_v4"]:
    print(f"\nTesting: {model_name}")
    print("-"*40)
    
    try:
        # Load model
        model_path = MODEL_PATHS[model_name]
        if model_name == "standard_gemma":
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                max_seq_length=512,
                dtype=torch.float32,
                load_in_4bit=True,
            )
        else:
            if os.path.exists(model_path):
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name=model_path,
                    max_seq_length=512,
                    dtype=torch.float32,
                    load_in_4bit=True,
                )
            else:
                print(f"  Model not found at {model_path}")
                continue
        
        # Test on branching only (key metric)
        results = evaluate_model_comprehensive(
            model, tokenizer,
            "branching_eval.jsonl",
            max_samples=50,  # Quick test with fewer samples
            method="logits",
            device="cuda",
            verbose=False
        )
        
        if results:
            quick_results[model_name] = results['accuracy']
            print(f"  Branching Accuracy: {results['accuracy']:.1f}%")
    
    except Exception as e:
        print(f"  Error testing {model_name}: {e}")

# Summary
print("\n" + "="*60)
print("QUICK BENCHMARK SUMMARY (Branching Task)")
print("-"*60)
for model_name, acc in quick_results.items():
    print(f"{model_name:<30} {acc:>8.1f}%")
```

## Usage Instructions

### For Complete Benchmarking Workflow:

1. **Test Standard Gemma** (Baseline)
   - Set `MODEL_TO_TEST = "standard_gemma"` in Cell 4
   - Run all cells
   - Save results

2. **Test Fine-tuned Transitivity**
   - Set `MODEL_TO_TEST = "transitivity_baseline"` in Cell 4
   - Run cells 4-9 again
   - Save results

3. **Test Fine-tuned D-separation**
   - Set `MODEL_TO_TEST = "dseparation_baseline"` in Cell 4
   - Run cells 4-9 again
   - Save results

4. **Test Semantic Transitivity V4**
   - Set `MODEL_TO_TEST = "transitivity_semantic_v4"` in Cell 4
   - Run cells 4-9 again
   - Save results

5. **Test Semantic D-separation** (when available)
   - Set `MODEL_TO_TEST = "dseparation_semantic"` in Cell 4
   - Run cells 4-9 again
   - Save results

### Expected Directory Structure in Google Drive:
```
/content/drive/MyDrive/
├── causal_data/
│   └── eval/
│       ├── length_eval.jsonl
│       ├── branching_eval.jsonl
│       ├── reversed_eval.jsonl
│       ├── shuffled_eval.jsonl
│       └── long_names_eval.jsonl
├── gemma_transitivity_models/
│   └── transitivity_v1_merged/
├── gemma_dseparation_models/
│   └── dseparation_v1_merged/
└── gemma_semantic_models/
    ├── transitivity/
    │   └── semantic_v4_merged/
    └── dseparation/
        └── semantic_v2_merged/
```

## Notes

- Ensure GPU is enabled: Runtime → Change runtime type → T4 GPU (or better)
- Each full evaluation takes ~5-10 minutes depending on GPU
- Results are automatically saved with timestamps
- The quick benchmark (Cell 10) is useful for rapid testing
- Use `method="generate"` if logits method gives unexpected results
