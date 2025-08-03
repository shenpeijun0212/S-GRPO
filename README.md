# Enhanced OAT for Mathematical Reasoning

This repository builds upon [Sail-SG's OAT project](https://github.com/sail-sg/oat/tree/main/oat), extending its capabilities with improvements in training stability, evaluation, and data processing, particularly for math reasoning tasks.

## 📁 Project Structure

- **`analysis/`**  
  Contains scripts for reading and processing logs, extracting keywords, and computing entropy statistics. You can use this module to organize, summarize, and analyze experimental data.

- **`datasets/`**  
  Includes five datasets used in training and evaluation.  
  > **Note:** The AIME dataset, consisting of only 30 high-difficulty high school competition problems, leads to significant variance during training. While we tested on AIME, it is excluded from overall statistical summaries.

- **`deploy_dpsk/`**  
  Original folder from the OAT project, retained for compatibility and comparison.

- **`example/`**  
  Provides an example output of our evaluation results on four datasets (excluding AIME), using the `Qwen2.5-Math-1.5B-Instruct` model.

- **`log/`**  
  Stores training logs for reference and reproducibility.

- **`r1_zero/`**  
  Contains baseline experiment settings and configurations.

- **`train/`**  
  Includes all training code for the GRPO method, evaluation scripts for datasets, and execution scripts for running experiments.

## 🔧 Environment

This project builds on the environment defined in the original [OAT repository](https://github.com/sail-sg/oat/tree/main/oat). Please follow the setup instructions there first. Any additional dependencies or changes will be listed in our respective module folders (e.g., `train/requirements.txt` if available).

## 📌 Notes

- The training and evaluation pipeline has been tailored for math problem-solving tasks.
- Detailed logs and example outputs are provided for transparency and reproducibility.

## 📞 Contact

For questions or collaboration, please feel free to open an issue or reach out via pull request.

---

