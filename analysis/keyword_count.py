import os
import json
import re
from collections import Counter
import pandas as pd

def count_keywords_in_file(file_path, keywords):
    """
    Opens a JSON file and counts the occurrences of specified keywords within the 'output' field of each dictionary.

    Args:
        file_path (str): The path to the JSON file.
        keywords (list): A list of keywords to count.

    Returns:
        Counter or str: A Counter object with keyword counts, or an error message string if an issue occurs.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        return f"Error: File not found at {file_path}"
    except json.JSONDecodeError:
        return f"Error: Failed to decode JSON from {file_path}"
    
    if not isinstance(data, list):
        return "Error: JSON content is not a list of objects."

    # Convert keywords to lowercase for case-insensitive matching.
    keywords_lower = [k.lower() for k in keywords]
    file_counts = Counter({k: 0 for k in keywords_lower})
    
    for item in data:
        if isinstance(item, dict) and 'output' in item:
            output_content = item['output']
            texts_to_search = []
            
            # The 'output' field can be either a list of strings or a single string.
            if isinstance(output_content, list):
                for text_item in output_content:
                    if isinstance(text_item, str):
                        texts_to_search.append(text_item)
            elif isinstance(output_content, str):
                texts_to_search.append(output_content)

            # Search for keywords in the extracted text.
            for text in texts_to_search:
                text_lower = text.lower()
                for keyword in keywords_lower:
                    try:
                        # Use regex to find whole words or phrases.
                        # re.escape handles special characters (e.g., '-') in keywords.
                        matches = re.findall(r'\b' + re.escape(keyword) + r'\b', text_lower)
                        file_counts[keyword] += len(matches)
                    except re.error as e:
                        print(f"  - Regex error for keyword '{keyword}': {e}")

    return file_counts

def main():
    """
    Main function to define file paths and keywords, then invoke the analysis function.
    """
    # ==================== User Configuration ====================
    # Set the numeric prefixes for the files you want to process.
    n_values = [16, 32, 48, 64, 80, 96]
    
    # Set the keywords you want to count.
    keywords_to_count = [
        "check again", 
        "re-evaluate", 
        "re-examine",
        "recheck", 
        "reconsider", 
        "rethink",
        "verify again"
    ]
    
    # Define the directory path where the files are located.
    #### Attention! change your file path here!
    directory = os.path.join(
        'eval_results' 
    )
    
    # Define the file suffixes to look for.
    target_suffixes = [
        '_amc.json',
        '_math.json',
        '_minerva.json',
        '_olympiad_bench.json'
    ]
    # ============================================================
    
    files_to_process = []
    
    print(f"Scanning directory: {os.path.abspath(directory)}")
    print(f"Target file prefixes: {n_values}\n")
    
    try:
        all_files_in_dir = os.listdir(directory)
        # Iterate through all specified numeric prefixes.
        for n in n_values:
            prefix_to_find = f"{n}_"
            # Scan the directory to find all files matching the current prefix and target suffixes.
            for filename in all_files_in_dir:
                if filename.startswith(prefix_to_find):
                    for suffix in target_suffixes:
                        if filename.endswith(suffix):
                            files_to_process.append(filename)
                            break
    except FileNotFoundError:
        print(f"Error: Directory not found at '{directory}'")
        return

    if not files_to_process:
        print(f"Warning: No target files starting with any of the prefixes {n_values} were found in '{directory}'.")
        print(f"Please check the n_values or the directory path.")
        return

    files_to_process.sort()
    print(f"Found {len(files_to_process)} files to process:\n{files_to_process}\n")

    all_results = {}
    
    for filename in files_to_process:
        full_path = os.path.join(directory, filename)
        result = count_keywords_in_file(full_path, keywords_to_count)
        all_results[filename] = result

    valid_results = {k: v for k, v in all_results.items() if isinstance(v, Counter)}
    
    if not valid_results:
        print("Failed to extract data from any files. Please check if the file contents match the expected JSON format.")
        return

    df = pd.DataFrame(valid_results).fillna(0).astype(int)
    
    # Calculate the sum of each keyword across all files and add a 'Total' column.
    df['Total'] = df.sum(axis=1)
    
    # Reindex to ensure all keywords are in the report, even if their total count is 0.
    df = df.reindex([k.lower() for k in keywords_to_count], fill_value=0)

    df.index.name = "Keyword"
    
    print("\n--- Keyword Statistics Summary ---")
    # For clarity, print only the total column for each keyword.
    summary_df = df[['Total']]
    print(summary_df)

    # If you want to see the detailed counts for each individual file, you can uncomment the line below.
    # print("\n--- Detailed Counts (Per File) ---")
    # print(df.to_string())


if __name__ == '__main__':
    main()
