# Data Directory

Course notebooks primarily rely on [TensorFlow Datasets](https://www.tensorflow.org/datasets) for reproducible data access. When you run a notebook the first time, TFDS will download datasets to your configured cache directory (default: `~/tensorflow_datasets`).

If a notebook requires custom assets, place small files under this folder and commit them only when they are lightweight and redistributable. Larger datasets should be hosted externally—reference the download URL in the notebook and update `.gitignore` if you introduce additional subdirectories (for example, `data/raw/`).

For Google Colab usage, prefer mounting Google Drive or using TFDS to avoid local storage limits.
