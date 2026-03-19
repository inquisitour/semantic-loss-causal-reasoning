# Complete Guide for Causal Reasoning Experiments with Integrated Axiomatic Training and Semantic Loss

## Table of Contents
1. Introduction
2. Environment Setup
3. Project Structure
4. Data Generation
5. Model Implementation
6. Training
7. Evaluation
8. Experiments
9. Analysis
10. Iterative Improvement
11. Final Steps

## 1. Introduction

This guide provides a comprehensive approach to experimenting with and evaluating our novel integrated method for causal reasoning, which combines axiomatic training with a semantic loss function. Our goal is to demonstrate improved generalization, sample efficiency, and performance on complex causal structures compared to the baseline axiomatic training method.

## 2. Environment Setup

```bash
# Create a new conda environment
conda create -n causal_reasoning python=3.8
conda activate causal_reasoning

# Install required packages
pip install torch torchvision torchaudio
pip install transformers datasets networkx matplotlib scipy pandas tqdm
```

## 3. Project Structure

Create the following directory structure:

```
causal_reasoning/
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── generate_data.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── gpt2_causal.py
│   │   └── semantic_loss.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── axiomatic_training.py
│   │   └── integrated_training.py
│   └── evaluation/
│       ├── __init__.py
│       └── evaluate.py
├── scripts/
│   ├── run_experiments.py
│   └── analyze_results.py
├── config/
│   └── experiments.py
├── results/
└── requirements.txt
```

## 4. Data Generation

```python
# src/data/generate_data.py

import networkx as nx
import random

def generate_causal_graph(num_nodes, edge_probability):
    G = nx.gnp_random_graph(num_nodes, edge_probability, directed=True)
    return nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])

def apply_causal_axioms(G):
    # Apply transitivity axiom
    for node in G.nodes():
        successors = list(G.successors(node))
        for s1 in successors:
            for s2 in G.successors(s1):
                if s2 not in successors:
                    G.add_edge(node, s2)
    return G

def generate_example(G):
    nodes = list(G.nodes())
    start = random.choice(nodes)
    end = random.choice([n for n in nodes if n != start])
    
    path_exists = nx.has_path(G, start, end)
    
    premise = f"{start} causes {', '.join(G.successors(start))}. "
    for node in G.nodes():
        if node != start:
            premise += f"{node} causes {', '.join(G.successors(node))}. "
    
    hypothesis = f"Does {start} cause {end}?"
    label = 1 if path_exists else 0
    
    return {
        "premise": premise.strip(),
        "hypothesis": hypothesis,
        "label": label
    }

def generate_dataset(num_graphs, nodes_per_graph, edge_probability, examples_per_graph):
    dataset = []
    for _ in range(num_graphs):
        G = generate_causal_graph(nodes_per_graph, edge_probability)
        G = apply_causal_axioms(G)
        for _ in range(examples_per_graph):
            dataset.append(generate_example(G))
    return dataset

# Usage
train_data = generate_dataset(1000, 5, 0.3, 10)
val_data = generate_dataset(100, 5, 0.3, 10)
test_data = generate_dataset(100, 5, 0.3, 10)
```

## 5. Model Implementation

```python
# src/models/gpt2_causal.py

import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model

class GPT2ForCausalReasoning(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.config = GPT2Config(
            vocab_size=vocab_size,
            n_positions=512,
            n_embd=512,
            n_layer=12,
            n_head=8,
        )
        self.transformer = GPT2Model(self.config)
        self.classifier = nn.Linear(512, 2)  # Binary classification

    def forward(self, input_ids, attention_mask=None):
        outputs = self.transformer(input_ids, attention_mask=attention_mask)
        logits = self.classifier(outputs.last_hidden_state[:, -1, :])
        return logits

# src/models/semantic_loss.py

import torch

def causal_semantic_loss(logits, labels, premises):
    # Placeholder implementation - replace with actual semantic loss
    probs = torch.softmax(logits, dim=1)
    consistent_probs = torch.where(labels == 1, probs[:, 1], probs[:, 0])
    return -torch.log(consistent_probs).mean()
```

## 6. Training

```python
# src/training/axiomatic_training.py

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_axiomatic(model, train_loader, val_loader, num_epochs, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    return model

# src/training/integrated_training.py

def train_integrated(model, train_loader, val_loader, num_epochs, device, lambda_semantic):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            premises = batch['premises']
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            ce_loss = criterion(outputs, labels)
            sem_loss = causal_semantic_loss(outputs, labels, premises)
            total_loss = ce_loss + lambda_semantic * sem_loss
            total_loss.backward()
            optimizer.step()
            
            train_loss += total_loss.item()
        
        train_loss /= len(train_loader)
        
        # Validation
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion, device)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    return model
```

## 7. Evaluation

```python
# src/evaluation/evaluate.py

import torch
from tqdm import tqdm

def evaluate_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(data_loader)
    accuracy = correct / total
    return avg_loss, accuracy
```

## 8. Experiments

```python
# config/experiments.py

EXPERIMENTS = [
    {
        'name': 'baseline',
        'model': 'axiomatic',
        'data': {'num_graphs': 1000, 'nodes_per_graph': 5, 'edge_probability': 0.3, 'examples_per_graph': 10},
        'training': {'num_epochs': 10, 'batch_size': 32}
    },
    {
        'name': 'integrated',
        'model': 'integrated',
        'data': {'num_graphs': 1000, 'nodes_per_graph': 5, 'edge_probability': 0.3, 'examples_per_graph': 10},
        'training': {'num_epochs': 10, 'batch_size': 32, 'lambda_semantic': 0.1}
    },
    # Add more configurations for different experiments
]

# scripts/run_experiments.py

import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from config.experiments import EXPERIMENTS
from src.data.generate_data import generate_dataset
from src.models.gpt2_causal import GPT2ForCausalReasoning
from src.training.axiomatic_training import train_axiomatic
from src.training.integrated_training import train_integrated
from src.evaluation.evaluate import evaluate_model

def run_experiments():
    results = {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    for exp in EXPERIMENTS:
        print(f"Running experiment: {exp['name']}")
        
        # Generate data
        train_data = generate_dataset(**exp['data'])
        val_data = generate_dataset(**exp['data'])
        test_data = generate_dataset(**exp['data'])
        
        # Tokenize data
        def tokenize_function(examples):
            return tokenizer(examples["premise"] + " " + examples["hypothesis"], padding="max_length", truncation=True)
        
        train_dataset = Dataset.from_dict(train_data).map(tokenize_function, batched=True)
        val_dataset = Dataset.from_dict(val_data).map(tokenize_function, batched=True)
        test_dataset = Dataset.from_dict(test_data).map(tokenize_function, batched=True)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=exp['training']['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=exp['training']['batch_size'])
        test_loader = DataLoader(test_dataset, batch_size=exp['training']['batch_size'])
        
        # Create model
        model = GPT2ForCausalReasoning(vocab_size=tokenizer.vocab_size).to(device)
        
        # Train model
        if exp['model'] == 'axiomatic':
            model = train_axiomatic(model, train_loader, val_loader, exp['training']['num_epochs'], device)
        elif exp['model'] == 'integrated':
            model = train_integrated(model, train_loader, val_loader, exp['training']['num_epochs'], device, exp['training']['lambda_semantic'])
        
        # Evaluate model
        criterion = nn.CrossEntropyLoss()
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        results[exp['name']] = {'test_loss': test_loss, 'test_accuracy': test_accuracy}
        
        print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return results

if __name__ == "__main__":
    results = run_experiments()
    print(results)
```

## 9. Analysis

```python
# scripts/analyze_results.py

import matplotlib.pyplot as plt
import pandas as pd

def plot_results(results):
    df = pd.DataFrame(results).T
    
    plt.figure(figsize=(10, 6))
    df['test_accuracy'].plot(kind='bar')
    plt.title('Test Accuracy Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    df['test_loss'].plot(kind='bar')
    plt.title('Test Loss Comparison')
    plt.xlabel('Experiment')
    plt.ylabel('Loss')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/loss_comparison.png')
    plt.close()

def analyze_results(results):
    plot_results(results)
    
    baseline_acc = results['baseline']['test_accuracy']
    integrated_acc = results['integrated']['test_accuracy']
    improvement = (integrated_acc - baseline_acc) / baseline_acc * 100
    
    print(f"Baseline Accuracy: {baseline_acc:.4f}")
    print(f"Integrated Approach Accuracy: {integrated_acc:.4f}")
    print(f"Improvement: {improvement:.2f}%")

if __name__ == "__main__":
    # Load results from a file or pass them directly
    results = {...}  # Load your results here
    analyze_results(results)
```

## 10. Iterative Improvement

1. Hyperparameter Tuning:
   - Use Ray Tune or Optuna for automated hyperparameter optimization
   - Parameters to tune: learning rate, batch size, lambda_semantic, model architecture (e.g., number of layers)

2. Model Architecture Improvements:
   - Experiment with different sizes while maintaining the GPT-2 style architecture
   - Implement custom attention mechanisms for causal reasoning

3. Loss Function Refinement:
   - Analyze the behavior of the semantic loss
   - Experiment with different formulations of the semantic loss
   - Implement and test the following variations:

```python
# src/models/semantic_loss.py

def improved_causal_semantic_loss(logits, labels, premises):
    probs = torch.softmax(logits, dim=1)
    positive_probs = probs[:, 1]
    negative_probs = probs[:, 0]
    
    # Parse premises to extract causal relationships
    causal_relations = parse_premises(premises)
    
    # Compute consistency loss
    consistency_loss = 0
    for i, relations in enumerate(causal_relations):
        for cause, effect in relations:
            cause_prob = positive_probs[i]
            effect_prob = positive_probs[i]
            consistency_loss += torch.abs(cause_prob - effect_prob)
    
    # Combine with original loss
    original_loss = -torch.log(torch.where(labels == 1, positive_probs, negative_probs)).mean()
    
    return original_loss + 0.1 * consistency_loss

def parse_premises(premises):
    # Implement parsing logic to extract causal relationships from premises
    # Return a list of lists, where each inner list contains (cause, effect) tuples
    pass
```

4. Data Augmentation:
   - Implement more complex causal structures in data generation
   - Add noise or perturbations to test robustness

```python
# src/data/generate_data.py

def generate_complex_causal_graph(num_nodes, edge_probability):
    G = nx.gnp_random_graph(num_nodes, edge_probability, directed=True)
    G = nx.DiGraph([(u,v) for (u,v) in G.edges() if u<v])
    
    # Add some cyclic relationships
    for _ in range(num_nodes // 4):
        nodes = random.sample(list(G.nodes()), 3)
        G.add_edge(nodes[0], nodes[1])
        G.add_edge(nodes[1], nodes[2])
        G.add_edge(nodes[2], nodes[0])
    
    return G

def add_noise_to_graph(G, noise_level=0.1):
    for u, v in list(G.edges()):
        if random.random() < noise_level:
            G.remove_edge(u, v)
    return G

# Update generate_dataset function to use these new methods
def generate_dataset(num_graphs, nodes_per_graph, edge_probability, examples_per_graph, add_noise=False):
    dataset = []
    for _ in range(num_graphs):
        G = generate_complex_causal_graph(nodes_per_graph, edge_probability)
        G = apply_causal_axioms(G)
        if add_noise:
            G = add_noise_to_graph(G)
        for _ in range(examples_per_graph):
            dataset.append(generate_example(G))
    return dataset
```

5. Scaling Experiments:
   - Gradually increase the size and complexity of causal graphs
   - Analyze performance vs. computational requirements
   - Implement distributed training for larger models and datasets

```python
# config/experiments.py

SCALING_EXPERIMENTS = [
    {
        'name': 'small',
        'data': {'num_graphs': 1000, 'nodes_per_graph': 5, 'edge_probability': 0.3, 'examples_per_graph': 10},
    },
    {
        'name': 'medium',
        'data': {'num_graphs': 2000, 'nodes_per_graph': 10, 'edge_probability': 0.2, 'examples_per_graph': 20},
    },
    {
        'name': 'large',
        'data': {'num_graphs': 5000, 'nodes_per_graph': 20, 'edge_probability': 0.1, 'examples_per_graph': 50},
    },
]

# Add this to run_experiments.py
def run_scaling_experiments():
    results = {}
    for exp in SCALING_EXPERIMENTS:
        start_time = time.time()
        exp_results = run_single_experiment(exp)
        end_time = time.time()
        
        results[exp['name']] = {
            'accuracy': exp_results['test_accuracy'],
            'training_time': end_time - start_time,
            'memory_usage': get_max_memory_usage(),
        }
    
    return results

def get_max_memory_usage():
    # Implement logic to get maximum memory usage during training
    pass
```

## 11. Final Steps

1. Compile comprehensive results:

```python
# scripts/compile_results.py

import json
import pandas as pd

def compile_results():
    all_results = {}
    
    # Load results from different experiment runs
    with open('results/baseline_results.json', 'r') as f:
        all_results['baseline'] = json.load(f)
    
    with open('results/integrated_results.json', 'r') as f:
        all_results['integrated'] = json.load(f)
    
    with open('results/scaling_results.json', 'r') as f:
        all_results['scaling'] = json.load(f)
    
    # Create a summary DataFrame
    summary = pd.DataFrame(all_results).T
    summary.to_csv('results/summary.csv')
    
    return summary

if __name__ == "__main__":
    summary = compile_results()
    print(summary)
```

2. Generate final visualizations:

```python
# scripts/visualize_results.py

import matplotlib.pyplot as plt
import seaborn as sns

def plot_accuracy_comparison(summary):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=summary.index, y='test_accuracy', data=summary)
    plt.title('Test Accuracy Comparison Across Experiments')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('results/accuracy_comparison.png')

def plot_scaling_performance(scaling_results):
    df = pd.DataFrame(scaling_results).T
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.scatterplot(x='nodes_per_graph', y='accuracy', data=df, ax=ax1)
    ax1.set_title('Accuracy vs Graph Size')
    
    sns.scatterplot(x='nodes_per_graph', y='training_time', data=df, ax=ax2)
    ax2.set_title('Training Time vs Graph Size')
    
    plt.tight_layout()
    plt.savefig('results/scaling_performance.png')

if __name__ == "__main__":
    summary = pd.read_csv('results/summary.csv', index_col=0)
    plot_accuracy_comparison(summary)
    plot_scaling_performance(summary['scaling'])
```

3. Prepare final report:

Create a Jupyter notebook or Markdown file that summarizes the experimental process, results, and key findings. Include:

- Overview of the approach
- Experimental setup
- Key results and visualizations
- Analysis of the integrated approach's performance compared to the baseline
- Discussion of scaling behavior
- Insights into the effectiveness of the semantic loss
- Limitations and future work

## 12. Conclusion and Next Steps

This guide provides a comprehensive framework for experimenting with and evaluating our novel integrated approach to causal reasoning. By following these steps, you can:

1. Implement the baseline axiomatic training and our integrated approach
2. Generate synthetic causal reasoning datasets
3. Train and evaluate models using both approaches
4. Analyze results and compare performance
5. Iteratively improve the model through various techniques

To continue improving the results:

1. Regularly update the `config/experiments.py` file with new experimental configurations
2. Implement and test new variations of the semantic loss function
3. Explore more complex causal structures in the data generation process
4. Conduct thorough hyperparameter optimization
5. Investigate the model's behavior on different types of causal queries
6. Collaborate with domain experts to validate the model's causal reasoning capabilities on real-world problems

# Document all changes, experiments, and results thoroughly. Regularly update the analysis scripts to incorporate new insights and visualizations as the project evolves.
