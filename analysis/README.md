 # Analysis

This folder contains tools for analyzing experimental logs, extracting keywords, and computing entropy-based statistics. These scripts help you organize and summarize experimental data efficiently.

## 📁 Contents

- **`read_log.py`**  
  Parses and summarizes training log files. It extracts relevant metrics such as accuracy and evaluation results across steps.

- **`read_keywords.py`**  
  Extracts and counts the occurrence of important keywords or concepts from logs.

- **`read_entropy.py`**  
  Computes entropy values from logs to measure output uncertainty and track training stability.

## 🔧 Usage Instructions

For each script, you only need to **modify the log file path** to point to your own log file. No additional configuration is required.

### Example (in `read_log.py`):

```python
# Replace this path with your own log file
log_path = "path/to/your/log.txt"
