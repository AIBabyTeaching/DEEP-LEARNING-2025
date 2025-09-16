# Deep Learning Course 2025

## Overview
This repository hosts the 14-week AAST Deep Learning Course for 2025, built around modern TensorFlow and Keras tooling. It provides instructor-curated notebooks, reusable utilities, and lightweight automation so the material scales with future cohorts.

Students work primarily on CPUs by default with the option to extend to NVIDIA GPU acceleration. Reproducible conda environments, pre-configured linting, and notebook hygiene checks keep collaboration smooth across Windows, macOS, and Linux.

## Authors
> **Course Lead:** AAST Machine Intelligence Faculty  \\
> **Maintainers:** Deep Learning Course 2025 Teaching Team  \\
> **Contact:** dl-course@aast.edu

## Installation
1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda).
2. Create and activate the default CPU environment:
   ```bash
   conda env create -f environment-cpu.yml
   conda activate deep-learning-course-2025
   ```
3. (Optional) Install the NVIDIA GPU-ready environment on supported Linux or Windows WSL2 setups:
   ```bash
   conda env create -f environment-gpu.yml
   conda activate deep-learning-course-2025-gpu
   ```
   > 💡 GPU support requires recent NVIDIA drivers plus CUDA-enabled hardware. On Windows, WSL2 with Ubuntu is the recommended configuration.
4. Register the environment as a Jupyter kernel:
   ```bash
   python -m ipykernel install --user --name deep-learning-course-2025 --display-name "DL 2025"
   ```

### TensorBoard Quickstart
After running a training script that logs to `runs/`, launch TensorBoard:
```bash
tensorboard --logdir runs
```
Open the provided URL in your browser to inspect scalars, histograms, and profiling traces.

## Contents
```
notebooks/    # W01_...W14_... instructional notebooks (paired with jupytext)
data/         # lightweight README and download guidance for TFDS
dl_utils/     # shared TensorFlow utilities (data, training, visualization)
templates/    # reusable notebook scaffolds
scripts/      # automation helpers such as new_week.py
.github/      # CI workflows
```

## Weekly Outline
- **W01** Intro to DL & TensorFlow/Keras; Colab/conda setup; tensors & gradients
- **W02** Perceptron & MLP; activations; initialization; He/Xavier
- **W03** Optimization (SGD/Adam); regularization (L2, dropout); early stopping
- **W04** CNN basics; conv/pool; small image classifier
- **W05** Advanced CNN; augmentation; transfer learning; fine-tuning
- **W06** RNN/LSTM/GRU; sequence modeling; text generation toy task
- **W07** Attention; transformer fundamentals
- **W08** NLP with keras-nlp; tokenization; pretrained workflows
- **W09** CV with keras-cv; detection/segmentation pipelines
- **W10** Efficient input pipelines (tf.data, TFRecord); performance tips
- **W11** TensorBoard; debugging; profiling
- **W12** Export & deployment (SavedModel, TFLite)
- **W13** Responsible AI; evaluation, bias, robustness, reproducibility
- **W14** Capstone project template & rubric

## How to Add a New Week
Use the helper script to scaffold a new notebook pair from the template:
```bash
python scripts/new_week.py W15_topic_name
```
This copies `templates/notebook_template.py` into `notebooks/W15_topic_name.py`, attempts to generate the paired `.ipynb` via jupytext, and prints follow-up instructions if jupytext is unavailable. After opening the notebook in JupyterLab, update the title, objectives, and exercise prompts.

## Contributing
- Install the pre-commit hooks once per machine:
  ```bash
  pre-commit install
  ```
- Format Python and notebook sources before committing:
  ```bash
  black .
  ruff check .
  nbqa black notebooks/
  nbqa ruff notebooks/
  ```
- Commit clean notebooks: hooks run nbstripout/nbQA to keep diffs lightweight.

## License
This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).
