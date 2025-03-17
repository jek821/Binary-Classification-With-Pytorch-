# main.py
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import utils.data as data

# Import model classes
from models_abstracted.perceptron_wrapper import PerceptronModel
from models_abstracted.linear_program_wrapper import LinearProgrammingModel
from models_abstracted.cross_entropy_wrapper import CrossEntropyModel
from models_abstracted.least_squares_wrapper import LeastSquaresModel
from models_abstracted.softmax_wrapper import SoftmaxModel

# Configuration
NUM_RUNS = 1000  # Default number of runs for experiment mode
SAMPLE_SIZE = 10000  # Data points per run

# Create directories
graphs_dir = os.path.join(os.path.dirname(__file__), "graphs")
os.makedirs(graphs_dir, exist_ok=True)

# Create Data Utility
data_util = data.Data()

def run_experiment(num_runs=1):
    """Run experiments for all models"""
    # Define models to use
    models = [
        PerceptronModel(learning_rate=0.01, max_iterations=1000),
        LinearProgrammingModel(),
        CrossEntropyModel(learning_rate=0.01, epochs=1000),
        LeastSquaresModel(learning_rate=0.01, epochs=1000)
        #SoftmaxModel(learning_rate=0.01, epochs=1000)
    ]
    
    # Result storage
    all_results = []
    aggregated_results = {model.name: {'train_acc': [], 'test_acc': [], 'time': [], 'memory': []} 
                          for model in models}
    
    # Run experiments
    for i in range(num_runs):
        if i % 100 == 0 and i > 0:
            print(f"Completed {i} runs...")
            
        # Generate data for this run
        X, y = data_util.generate_linearly_separable(n_samples=SAMPLE_SIZE, dim=2)
        X_train, X_test, y_train, y_test = data_util.partition_data(train_ratio=0.8)
        
        # Create a copy of the data with converted labels for models that need -1/1 labels
        y_train_neg = 2 * y_train - 1  # Convert to -1,1
        y_test_neg = 2 * y_test - 1    # Convert to -1,1
        
        # Run each algorithm and collect results
        run_results = []
        for model in models:
            # Select the appropriate labels based on model requirements
            if hasattr(model, 'requires_negative_labels') and model.requires_negative_labels:
                result = model.evaluate(X_train, y_train_neg, X_test, y_test_neg)
            else:
                result = model.evaluate(X_train, y_train, X_test, y_test)
                
            run_results.append(result)
            
            # Store for aggregation
            aggregated_results[model.name]['train_acc'].append(result['Training Accuracy'])
            aggregated_results[model.name]['test_acc'].append(result['Testing Accuracy'])
            aggregated_results[model.name]['time'].append(result['Time (s)'])
            aggregated_results[model.name]['memory'].append(result['Memory (MB)'])
        
        # Store the first run's detailed results for display
        if i == 0:
            all_results = run_results
    
    # Calculate averages for each algorithm
    summary_results = []
    for model in models:
        metrics = aggregated_results[model.name]
        summary = {
            'Model': model.name,
            'Avg Training Accuracy': np.mean(metrics['train_acc']),
            'Avg Testing Accuracy': np.mean(metrics['test_acc']),
            'Avg Time (s)': np.mean(metrics['time']),
            'Avg Memory (MB)': np.mean(metrics['memory'])
        }
        summary_results.append(summary)
    
    return all_results, summary_results

def create_bar_charts(results, title_suffix="", filename_prefix=""):
    """Create bar charts comparing algorithms"""
    model_names = [result['Model'] for result in results]
    
    # Create accuracy comparison chart
    plt.figure(figsize=(10, 5))
    x = np.arange(len(model_names))
    width = 0.35
    
    train_acc = [result['Avg Training Accuracy'] if 'Avg Training Accuracy' in result else result['Training Accuracy'] for result in results]
    test_acc = [result['Avg Testing Accuracy'] if 'Avg Testing Accuracy' in result else result['Testing Accuracy'] for result in results]
    
    plt.bar(x - width/2, train_acc, width, label='Training Accuracy')
    plt.bar(x + width/2, test_acc, width, label='Testing Accuracy')
    
    plt.xlabel('Models')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Training vs Testing Accuracy {title_suffix}')
    plt.xticks(x, model_names)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save accuracy chart
    acc_path = os.path.join(graphs_dir, f"{filename_prefix}accuracy_comparison.png")
    plt.savefig(acc_path)
    print(f"Accuracy chart saved to {acc_path}")
    
    # Create performance metrics chart
    plt.figure(figsize=(10, 8))
    
    # Time subplot
    plt.subplot(2, 1, 1)
    times = [result['Avg Time (s)'] if 'Avg Time (s)' in result else result['Time (s)'] for result in results]
    plt.bar(model_names, times, color='skyblue')
    plt.xlabel('Models')
    plt.ylabel('Time (s)')
    plt.title(f'Execution Time Comparison {title_suffix}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Memory subplot
    plt.subplot(2, 1, 2)
    memory = [result['Avg Memory (MB)'] if 'Avg Memory (MB)' in result else result['Memory (MB)'] for result in results]
    plt.bar(model_names, memory, color='lightgreen')
    plt.xlabel('Models')
    plt.ylabel('Memory Usage (MB)')
    plt.title(f'Memory Usage Comparison {title_suffix}')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    perf_path = os.path.join(graphs_dir, f"{filename_prefix}performance_comparison.png")
    plt.savefig(perf_path)
    print(f"Performance chart saved to {perf_path}")

def format_results_for_display(results):
    display_results = []
    for result in results:
        display_result = {k: v for k, v in result.items()}
        
        # Format accuracies
        if 'Training Accuracy' in result:
            display_result['Training Accuracy'] = f"{result['Training Accuracy']:.2f}%"
        if 'Testing Accuracy' in result:
            display_result['Testing Accuracy'] = f"{result['Testing Accuracy']:.2f}%"
        if 'Avg Training Accuracy' in result:
            display_result['Avg Training Accuracy'] = f"{result['Avg Training Accuracy']:.2f}%"
        if 'Avg Testing Accuracy' in result:
            display_result['Avg Testing Accuracy'] = f"{result['Avg Testing Accuracy']:.2f}%"
            
        # Format time and memory
        if 'Time (s)' in result:
            display_result['Time (s)'] = f"{result['Time (s)']:.4f}"
        if 'Memory (MB)' in result:
            display_result['Memory (MB)'] = f"{result['Memory (MB)']:.4f}"
        if 'Avg Time (s)' in result:
            display_result['Avg Time (s)'] = f"{result['Avg Time (s)']:.4f}"
        if 'Avg Memory (MB)' in result:
            display_result['Avg Memory (MB)'] = f"{result['Avg Memory (MB)']:.4f}"
            
        display_results.append(display_result)
    
    return display_results

def main():
    print("\n" + "="*50)
    print("Running Classification Algorithm Comparison")
    print("="*50)
    
    # Ask user if they want to run a single test or multiple experiments
    user_input = input("Run multiple experiments? (y/n, default=n): ").lower()
    run_experiments = user_input.startswith('y')
    
    if run_experiments:
        # Ask for number of runs
        try:
            num_runs = int(input(f"Number of runs (default={NUM_RUNS}): "))
        except ValueError:
            num_runs = NUM_RUNS
            
        print(f"\nRunning {num_runs} experiments...")
        single_results, summary_results = run_experiment(num_runs)
        
        # Display first run results
        print("\n" + "="*50)
        print("First Run Results:")
        print("="*50)
        print(pd.DataFrame(format_results_for_display(single_results)).to_string(index=False))
        
        # Display summary results
        print("\n" + "="*50)
        print(f"Summary Results ({num_runs} runs):")
        print("="*50)
        print(pd.DataFrame(format_results_for_display(summary_results)).to_string(index=False))
        
        # Create charts
        create_bar_charts(single_results, "- First Run", "single_")
        create_bar_charts(summary_results, f"- Average of {num_runs} Runs", "avg_")
    else:
        # Run a single experiment and display results
        single_results, _ = run_experiment(1)
        
        print("\n" + "="*50)
        print("Results:")
        print("="*50)
        print(pd.DataFrame(format_results_for_display(single_results)).to_string(index=False))
        
        # Create charts
        create_bar_charts(single_results, "- Single Run", "")
    
    print("\nExecution complete.")

if __name__ == "__main__":
    main()