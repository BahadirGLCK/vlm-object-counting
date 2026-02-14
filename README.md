# vlm-object-counting

This project focuses on fine-tuning and evaluating vision-language models with Unsloth.

## Project Structure

```
vlm-object-counting/
├── configs/                # Global configuration files (not specific to one experiment)
├── data/
│   └── inputs/             # Raw input data (prompts, CSVs, videos)
│       ├── PROMPT.txt
│       ├── ground_truth.csv
│       └── vlm_videos/     # Video files
├── experiments/            # Output directory for all finetuning/evaluation runs
│   └── YYYY-MM-DD_HH-MM-SS_finetune/ # Example experiment directory
│       ├── experiment_config.json    # Snapshot of config used for this run
│       ├── data_split/               # Information about data splits (e.g., test_video_names.txt)
│       ├── finetuned_model/          # Saved model checkpoints (LoRA, tokenizer, merged, Ollama/GGUF)
│       │   └── ollama/               # GGUF files + Modelfiles + helper ollama commands
│       └── evaluation_results/       # Outputs from the evaluation process
├── notebooks/              # Jupyter notebooks for experimentation and analysis
├── requirements.txt        # Project dependencies
├── main.py                 # Main pipeline script to run finetuning and/or evaluation
├── scripts/                # Standalone execution scripts for individual processes
│   ├── run_finetuning.py
│   └── run_evaluation.py
├── src/                    # Source code for the vlm-object-counting toolkit
│   ├── __init__.py
│   ├── config.py           # Centralized configuration parameters
│   ├── experiment_manager.py # Class to manage experiment artifacts and configurations
│   ├── training.py         # Contains the Trainer class for the finetuning process
│   ├── finetuning.py       # Main script for the finetuning process (uses Trainer)
│   ├── evaluating.py       # Contains the Evaluator class for the evaluation process
│   ├── evaluation.py       # Main script for the evaluation process (uses Evaluator)
│   └── utils.py            # Shared utility functions
└── tests/                  # Test scripts (to be developed)
```

The core logic is organized into classes like `ExperimentManager`, `Trainer`, and `Evaluator` to promote modularity and reusability.

## Setup

### Prerequisites

*   Python 3.9+ (Python 3.10 is used in the Dockerfile)
*   `pip` (Python package installer)
*   NVIDIA GPU with CUDA support (for Unsloth and 4-bit/8-bit training)
*   Docker (if using the provided Docker environment)
*   NVIDIA Container Toolkit (if using Docker with GPU support)

### System Dependencies

Install the following system-level dependencies. These are generally required for building Python packages with C extensions and for multimedia processing:

```bash
sudo apt-get update
sudo apt-get install -y build-essential python3-dev libaio-dev ffmpeg libsm6 libxext6 libmpich-dev
```

### Python Environment Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd vlm-object-counting
    ```

2.  **Create and activate a virtual environment** (recommended):

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
    (On Windows, use `venv\Scripts\activate`)

3.  **Install Python dependencies**:

    The `requirements.txt` file lists all necessary packages. Install them using:

    ```bash
    pip install -r requirements.txt
    ```
    If you encounter issues with xformers or bitsandbytes related to your CUDA version, you might need to install them with specific CUDA compatibility. The `requirements.txt` includes an extra index URL for PyTorch that often helps.

4.  **Triton Cache (Optional but Recommended for Unsloth/xformers)**:

    For optimal performance with libraries like xformers that use Triton, create and set permissions for the autotune directory:

    ```bash
    mkdir -p ~/.triton/autotune
    chmod 700 ~/.triton/autotune
    ```
    Note: The Dockerfile sets up a similar cache directory at `/app/.triton_cache` within the container.

## Docker Setup (Recommended for Server Environments)

A `Dockerfile` is provided to create a consistent environment with all necessary dependencies.

### Prerequisites for Docker Usage

*   Docker installed on your system.
*   NVIDIA Container Toolkit installed if you intend to use GPUs with Docker. This allows Docker containers to access your NVIDIA GPUs.

### Building the Docker Image

1.  **Ensure `requirements.txt` is complete and accurate**:
    *   Verify that all Python dependencies are listed with their correct versions.
    *   For PyTorch, ensure it is specified for CUDA 12.1 (as used in the base Docker image). Example for `requirements.txt`:
        ```
        torch --index-url https://download.pytorch.org/whl/cu121
        # Other packages like:
        # pandas==2.0.3
        # unsloth
        # ...etc.
        ```

2.  **Build the image**:
    Navigate to the project root directory (where the `Dockerfile` is located) and run:
    ```bash
    docker build -t vlm-object-counting-dev .
    ```
    You can replace `vlm-object-counting-dev` with your preferred image tag.

### Running the Docker Container

Once the image is built, you can run a container:

*   **With GPU access (recommended for training/evaluation):**
    ```bash
    docker run -it --rm --gpus all -v $(pwd)/experiments:/app/experiments -v $(pwd)/data/inputs:/app/data/inputs vlm-object-counting-dev
    ```
    *   `-it`: Interactive mode with a pseudo-TTY.
    *   `--rm`: Automatically remove the container when it exits.
    *   `--gpus all`: Makes all NVIDIA GPUs available to the container.
    *   `-v $(pwd)/experiments:/app/experiments`: Mounts your local `experiments` directory into the container. This allows experiment artifacts (models, logs) to persist on your host machine even after the container exits.
    *   `-v $(pwd)/data/inputs:/app/data/inputs`: Mounts your local `data/inputs` directory. This is useful if your input data is large and you don't want to copy it into the image every time you build.

*   **Without GPU access (for CPU-only tasks or debugging):**
    ```bash
    docker run -it --rm -v $(pwd)/experiments:/app/experiments -v $(pwd)/data/inputs:/app/data/inputs vlm-object-counting-dev
    ```

Upon running the container, you will be dropped into a `bash` shell inside `/app`, with the Python virtual environment activated and all dependencies ready.

**Inside the container, you can then run the pipeline scripts as usual:**

```bash
python main.py
# or
python scripts/run_finetuning.py
# etc.
```

## Usage

The primary way to run the finetuning and evaluation pipelines is through `main.py`.

### Running the Full Pipeline (Finetuning + Evaluation)

```bash
python main.py
```

### Skipping Finetuning

If you want to skip finetuning and only run evaluation on a previously finetuned model, you need to specify the experiment directory:

```bash
python main.py --skip_finetuning --eval_experiment_dir experiments/YYYY-MM-DD_HH-MM-SS_finetune
```
(Replace `experiments/YYYY-MM-DD_HH-MM-SS_finetune` with the actual path to your experiment directory)

### Skipping Evaluation

```bash
python main.py --skip_evaluation
```

### Running Standalone Scripts

You can also run finetuning or evaluation as standalone processes:

*   **Finetuning:**
    ```bash
    python scripts/run_finetuning.py
    ```
    This will create a new timestamped experiment directory under `experiments/`.

*   **Evaluation:**
    ```bash
    python scripts/run_evaluation.py experiments/YYYY-MM-DD_HH-MM-SS_finetune
    ```
    (Replace `experiments/YYYY-MM-DD_HH-MM-SS_finetune` with the path to the experiment you want to evaluate.)

### Ollama Export (via Finetuning Pipeline)

The training pipeline now keeps existing exports and also creates Ollama-ready artifacts using Unsloth GGUF export:

*   Existing exports are still saved:
    *   LoRA checkpoint
    *   Tokenizer
    *   Merged 16-bit model
*   New Ollama exports are saved under:
    *   `experiments/<run>/finetuned_model/ollama/`
    *   Includes `.gguf` files, generated `Modelfile.*` files, and `ollama_commands.txt`.

Configuration lives in `src/config.py`:

*   `ENABLE_OLLAMA_EXPORT`
*   `OLLAMA_GGUF_QUANTIZATION_METHODS`
*   `OLLAMA_MODEL_NAME_PREFIX`
*   `OLLAMA_LIBRARY_NAMESPACE` (optional, for `ollama push`)
*   `OLLAMA_MODELFILE_PARAMETERS`

After finetuning, run the generated helper commands:

```bash
cd experiments/<run>/finetuned_model/ollama
cat ollama_commands.txt
```

This follows the official Ollama flow (`Modelfile` with `FROM ./model.gguf`, then `ollama create` and optional `ollama push`).

## Contributing

(Guidelines for contributing to the project - TBD) 
