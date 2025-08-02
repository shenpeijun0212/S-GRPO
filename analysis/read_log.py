import os
import json
import re
from collections import Counter, defaultdict
import pandas as pd
from IPython.display import display

def extract_fixed_benchmarks(log_path):
    """
    Parses a log file to extract accuracy and token length for specified benchmarks.

    Args:
        log_path (str): The path to the log file.

    Returns:
        dict: A dictionary containing lists of results for each benchmark.
    """
    benchmarks = ['math', 'amc', 'minerva', 'olympiad_bench']
    results = {bench: [] for bench in benchmarks}

    # Regex to find benchmark data in log lines.
    pattern = re.compile(
        r"'eval/(?P<bench>math|amc|minerva|olympiad_bench)/(?P<metric>accuracy|response_tok_len)': (?P<value>[^\s,]+)"
    )

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        current = {bench: {} for bench in benchmarks}
        for line in f:
            match = pattern.search(line)
            if match:
                bench = match.group("bench")
                metric = match.group("metric")
                value_str = match.group("value")
                try:
                    value = float(value_str)
                except ValueError:
                    # Skip values that cannot be converted to float.
                    continue

                current[bench][metric] = value
                # When both metrics for a benchmark are found, store them and reset.
                if "accuracy" in current[bench] and "response_tok_len" in current[bench]:
                    results[bench].append({
                        "accuracy": current[bench]["accuracy"],
                        "response_tok_len": current[bench]["response_tok_len"]
                    })
                    current[bench] = {}
    return results

def display_field_by_benchmark(dfs, log_files, bench_name: str, field_type: str):
    """
    Filters and displays a specific field (accuracy or length) for a given benchmark.

    Args:
        dfs (dict): Dictionary of DataFrames for each benchmark.
        log_files (dict): Dictionary of log files to determine column order.
        bench_name (str): The name of the benchmark to display.
        field_type (str): The type of field to display ('acc' or 'len').
    """
    assert bench_name in dfs, f"Unknown benchmark: {bench_name}"
    assert field_type in ("acc", "len"), "field_type must be 'acc' or 'len'"

    df = dfs[bench_name].copy()

    # Keep only the relevant columns based on the field_type.
    selected_cols = [col for col in df.columns if col.endswith(f"_{field_type}")]
    if not selected_cols:
        return pd.DataFrame()

    # Reorder columns to match the order in log_files for consistent display.
    sorted_cols = [f'{label}_{field_type}' for label in log_files if f'{label}_{field_type}' in selected_cols]
    return df[sorted_cols]

def main():
    """
    Main function to configure, execute, and display the benchmark analysis.
    """
    # ==================== User Configuration ====================
    # TODO: Update this dictionary with your model labels and log file paths.
    log_files = {
        "./run.log",
    }

    # Names of benchmarks to extract from the logs.
    benchmarks = ['math', 'amc', 'minerva', 'olympiad_bench']

    # Extract data from all specified log files.
    all_results = {label: extract_fixed_benchmarks(path) for label, path in log_files.items()}

    # Organize the extracted data into pandas DataFrames.
    dfs = {}
    for bench in benchmarks:
        # Ensure the script continues even if a benchmark is missing from some log files.
        if not any(bench in res for res in all_results.values()):
            continue
        
        records = defaultdict(dict)
        for label, result in all_results.items():
            if bench in result:
                for i, entry in enumerate(result[bench]):
                    step = i * 16
                    records[step][f'{label}_acc'] = entry['accuracy']
                    records[step][f'{label}_len'] = entry['response_tok_len']
        
        df = pd.DataFrame.from_dict(records, orient='index').sort_index()
        dfs[bench] = df

    # Modify the DataFrames by shifting the index and adding an initial row.
    modified_dfs = {}
    for bench_name, df in dfs.items():
        if bench_name == 'average':
            # For 'average', keep the DataFrame as is.
            modified_dfs[bench_name] = df.copy()
        else:
            # For other benchmarks, perform shift and insert operations.
            
            # 1. Increment the index of the original DataFrame by 16.
            shifted_df = df.copy()
            shifted_df.index = shifted_df.index + 16
            
            # 2. Create a new row at step 0 with all values set to -1.
            new_row = pd.DataFrame(-1, index=[0], columns=df.columns)
            
            # 3. Concatenate the new row and the shifted DataFrame.
            modified_df = pd.concat([new_row, shifted_df])
            modified_dfs[bench_name] = modified_df

    dfs = modified_dfs

    # Display the 'average' accuracy from the logs.
    for name in ['math', 'amc', 'minerva', 'olympiad_bench']:
        print(name)
        display(display_field_by_benchmark(dfs, log_files, name, 'acc'))

    print("\n------Average Accuracy (math, amc, minerva, olympiad_bench) ------")

    # Calculate a custom average accuracy across a specific set of benchmarks.
    benchmarks_for_avg = ['math', 'amc', 'minerva', 'olympiad_bench']
    acc_dfs_to_avg = []

    # Collect all DataFrames for which the average needs to be calculated.
    for bench_name in benchmarks_for_avg:
        if bench_name in dfs:
            acc_df = display_field_by_benchmark(dfs, log_files, bench_name, 'acc')
            if not acc_df.empty:
                acc_dfs_to_avg.append(acc_df)
        else:
            print(f"Warning: Benchmark '{bench_name}' not found and will be excluded from the custom average.")

    # Check if any DataFrames were collected before attempting to calculate the average.
    if acc_dfs_to_avg:
        # Use pd.concat and .mean() to robustly calculate the average, which handles potential alignment issues.
        # Stack all DataFrames together.
        concatenated_df = pd.concat(acc_dfs_to_avg)
        # Group by index (step) and calculate the mean.
        custom_average_df = concatenated_df.groupby(concatenated_df.index).mean()
        
        display(custom_average_df)
    else:
        print("Could not calculate custom average because no data was found for the required benchmarks.")

if __name__ == '__main__':
    main()
