import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def calculate_entropy(log_probs):
    """
    Calculates entropy from log probabilities.
    """
    log_probs = np.array(log_probs, dtype=float)
    probs = np.exp(log_probs)
    entropy = -np.sum(probs * log_probs)
    return entropy

def analyze_log_file(file_path):
    """
    Analyzes a single log file to extract log probabilities at specified steps and calculate entropy.
    """
    print(f"Analyzing file: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

    # Find all global steps and their positions in the file.
    step_matches = list(re.finditer(r"'misc/global_step': (\d+)\.0", log_content))
    steps = [{'step': int(float(m.group(1))), 'pos': m.start()} for m in step_matches]

    # Find all action_logprobs or response_logprobs and their positions.
    log_probs_pattern = re.compile(r"(?:'action_logprobs':|response_logprobs=)\s*[(\[]([^)\]]+)[)\]]", re.DOTALL)
    log_probs_matches = list(log_probs_pattern.finditer(log_content))
    
    entropies = {}

    # For each step, find the log probabilities between the metric log of the previous step and the current step.
    for i in range(len(steps)):
        current_step_info = steps[i]
        current_step_val = current_step_info['step']
        
        # --- Key modification here (1/2) ---
        # Only process the first 100 steps.
        if current_step_val > 100:
            break
        
        # --- Key modification here (2/2) ---
        # Only process steps that are multiples of 5 and not zero.
        if current_step_val == 0 or current_step_val % 5 != 0:
            continue

        start_pos = steps[i-1]['pos']
        end_pos = current_step_info['pos']

        # Find log probabilities within the log segment for the current step.
        for log_prob_match in log_probs_matches:
            if start_pos < log_prob_match.start() < end_pos:
                log_probs_str = log_prob_match.group(1)
                try:
                    # Clean the string and convert it to a list of floats.
                    log_probs = [float(p.strip()) for p in log_probs_str.replace('\n', '').split(',') if p.strip()]
                    if log_probs:
                        entropy = calculate_entropy(log_probs)
                        entropies[current_step_val] = entropy
                        # Take only the first found log probability entry for each step segment.
                        break 
                except ValueError as e:
                    print(f"Error processing log probabilities for step {current_step_val}: {e}")
                    break
    return entropies

def main():
    """
    Main function to configure and run the entropy analysis.
    """
    # ==================== User Configuration ====================
    # TODO: Ensure these paths are correct relative to where you run the script.
    log_files = {
        'log1': './run_1.log',
        "log2": './run_2.log',
        "log3": './run_3.log',  
        "log4": './run_4.log',
        "log5": './run_5.log',
    }
    # ============================================================
    
    all_entropies = {}
    for name, path in log_files.items():
        # Check if the file exists.
        if not os.path.exists(path):
            print(f"Warning: Path '{path}' does not exist, skipping this file.")
            continue
        entropies = analyze_log_file(path)
        if entropies:
            all_entropies[name] = entropies

    if not all_entropies:
        print("Failed to extract any entropy data from the log files. Please check the file paths and content.")
        return

    # Create a DataFrame for comparison.
    df = pd.DataFrame(all_entropies)
    df.index.name = 'Step'
    df = df.sort_index()

    print("\n--- Entropy Comparison (First 100 Steps, Every 5 Steps) ---")
    print(df.to_string())

    # Plot the results.
    plt.figure(figsize=(12, 7))
    for column in df.columns:
        plt.plot(df.index, df[column], marker='o', linestyle='-', label=column)
    
    plt.title('Entropy vs. Training Step for Different Experiments (First 100 Steps, Sampled Every 5)')
    plt.xlabel('Training Step')
    plt.ylabel('Entropy')
    plt.legend()
    plt.grid(True)
    
    # Set x-axis ticks for better visibility.
    if not df.empty:
        max_step = min(df.index.max(), 100)
        plt.xticks(np.arange(0, max_step + 5, 5))
    
    plt.tight_layout()
    
    output_filename = '***.png'
    #plt.savefig(output_filename)
    print(f"\nComparison plot saved as {output_filename}")

if __name__ == '__main__':
    main()
