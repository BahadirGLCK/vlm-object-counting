import random
import pathlib
import logging
import re
from tqdm import tqdm
import pandas as pd
# from datasets import Dataset # Dataset class is not directly used, examples are lists of dicts
from unsloth import FastVisionModel
from unsloth.trainer import (
    UnslothVisionDataCollator
)
from trl import SFTTrainer, SFTConfig
from transformers import AutoProcessor

from .utils import row_to_example
from .experiment_manager import ExperimentManager

class Trainer:
    """
    Handles the model finetuning process, including data preparation, 
    model initialization, training, and artifact saving.

    Attributes:
        config: The global configuration module.
        exp_manager (ExperimentManager): Manager for the current experiment.
        logger (logging.Logger): Logger instance, typically from ExperimentManager.
        train_ds (list): Training dataset, list of examples.
        eval_ds (list): Evaluation dataset for during training.
        test_ds (list): Test dataset (split but not used directly by UnslothTrainer here).
        model: The finetuned model instance.
        tokenizer: The tokenizer associated with the model.
        processor: The processor associated with the model.
    """
    def __init__(self, config, experiment_manager: ExperimentManager):
        """
        Initializes the Trainer.

        Args:
            config: The global configuration module.
            experiment_manager: The ExperimentManager instance for the current run.
        """
        self.config = config
        self.exp_manager = experiment_manager
        self.logger = getattr(experiment_manager, 'logger', logging.getLogger(f"{__name__}.Trainer"))
        if not self.exp_manager.logger:
             self.logger.warning("ExperimentManager logger not found. Trainer using fallback logger.")
        
        random.seed(self.config.RANDOM_SEED)
        self.logger.info(f"Trainer initialized for experiment: {self.exp_manager.current_experiment_path}. Random seed: {self.config.RANDOM_SEED}")
        # Initialize attributes that will be set later
        self.train_ds, self.eval_ds, self.test_ds = [], [], []
        self.train_ds_meta, self.eval_ds_meta, self.test_ds_meta = [], [], [] # For storing metadata like video_name
        self.model, self.tokenizer, self.processor = None, None, None

    def prepare_data_splits(self, ohs_prompt: str) -> tuple[list, list, list]:
        """
        Loads data from the CSV specified in config, prepares examples using
        `row_to_example`, and splits them into training, evaluation, and test sets.
        The main datasets (train_ds, eval_ds, test_ds) will contain only the
        model inputs (e.g., {"messages": ...}), while metadata like video names
        will be stored in corresponding _meta lists.

        Args:
            ohs_prompt (str): The OHS prompt string to be used in examples.

        Returns:
            tuple[list, list, list]: A tuple containing train, eval, and test datasets (model inputs only).
        """
        self.logger.info("Preparing data splits...")
        raw_df = pd.read_csv(self.config.CSV_PATH, header=None, names=["video_name", "gemini_answer"], encoding="utf-8")
        self.logger.info(f"Loaded {len(raw_df)} rows from {self.config.CSV_PATH}")
        
        # Temporarily store examples with their metadata before splitting
        full_examples_with_meta = []
        for _, row in tqdm(raw_df.iterrows(), total=len(raw_df), desc="Preparing examples"):
            model_input_data = row_to_example(row, ohs_prompt) # This now returns {"messages": ...}
            # Construct video_path here for consistent use, or just store video_name if preferred for meta
            video_path = self.config.VIDEO_DIR / row["video_name"]
            full_examples_with_meta.append({
                "data": model_input_data, 
                "video_path": str(video_path) # Store full path or just row["video_name"]
            })
            
        random.shuffle(full_examples_with_meta)
        self.logger.info(f"Created and shuffled {len(full_examples_with_meta)} examples with metadata.")
        
        train_idx = int(len(full_examples_with_meta) * self.config.TRAIN_SPLIT_RATIO)
        eval_idx_end = int(len(full_examples_with_meta) * (self.config.TRAIN_SPLIT_RATIO + self.config.EVAL_SPLIT_RATIO))
        
        # Populate main datasets with only model input data
        self.train_ds = [ex["data"] for ex in full_examples_with_meta[:train_idx]]
        self.eval_ds  = [ex["data"] for ex in full_examples_with_meta[train_idx:eval_idx_end]]
        self.test_ds  = [ex["data"] for ex in full_examples_with_meta[eval_idx_end:]]

        # Populate meta datasets
        self.train_ds_meta = full_examples_with_meta[:train_idx]
        self.eval_ds_meta  = full_examples_with_meta[train_idx:eval_idx_end]
        self.test_ds_meta  = full_examples_with_meta[eval_idx_end:]
        
        self.logger.info(f"Data splits prepared: Train ({len(self.train_ds)}), Eval ({len(self.eval_ds)}), Test ({len(self.test_ds)})")
        return self.train_ds, self.eval_ds, self.test_ds

    def save_data_split_info(self) -> None:
        """
        Saves the video filenames for each data split (train, eval, test) 
        into respective .txt files in the experiment's data_split directory.
        Relies on `prepare_data_splits` having been called first and _meta lists populated.
        """
        self.logger.info("Saving data split information...")
        if not self.train_ds_meta and not self.eval_ds_meta and not self.test_ds_meta: # Check if any meta split has data
            self.logger.error("Metadata for data splits is empty or not prepared. Call prepare_data_splits() first.")
            raise ValueError("Metadata for data splits not prepared or are empty. Call prepare_data_splits() first.")
            
        data_split_dir = self.exp_manager.get_data_split_dir(self.config.DATA_SPLIT_DIR_NAME)
        with open(data_split_dir / "train_video_names.txt", "w", encoding="utf-8") as f:
            for example_meta in self.train_ds_meta:
                f.write(pathlib.Path(example_meta["video_path"]).name + "\n") # Use video_path from meta
        with open(data_split_dir / "eval_video_names.txt", "w", encoding="utf-8") as f:
            for example_meta in self.eval_ds_meta:
                f.write(pathlib.Path(example_meta["video_path"]).name + "\n")
        with open(data_split_dir / "test_video_names.txt", "w", encoding="utf-8") as f:
            for example_meta in self.test_ds_meta:
                f.write(pathlib.Path(example_meta["video_path"]).name + "\n")
        self.logger.info(f"Data split information saved in {data_split_dir}")

    def initialize_and_prepare_model(self) -> tuple[any, any, any]:
        """
        Initializes the base model (FastVisionModel) from pretrained weights, 
        applies PEFT (LoRA) configuration, and prepares it for training.
        Sets `self.model`, `self.tokenizer`, `self.processor`.

        Returns:
            tuple: The initialized model, tokenizer, and processor.
        """
        self.logger.info("Initializing and preparing model...")
        lora_config_dict = self.config.LORA_CONFIG.copy()
        if 'random_state' not in lora_config_dict:
            lora_config_dict["random_state"] = self.config.RANDOM_SEED
            self.logger.info(f"Applied RANDOM_SEED ({self.config.RANDOM_SEED}) to LORA_CONFIG's random_state.")
            
        self.logger.info(f"Loading base model: {self.config.BASE_MODEL_ID} with max_seq_len: {self.config.MAX_SEQ_LEN_FINETUNE}")
        model, tokenizer = FastVisionModel.from_pretrained(
            self.config.BASE_MODEL_ID,
            load_in_4bit=True,
            use_gradient_checkpointing="unsloth",
            max_seq_length=self.config.MAX_SEQ_LEN_FINETUNE,
        )
        
        self.logger.info(f"Applying PEFT (LoRA) to the model with config: {lora_config_dict}")
        self.model = FastVisionModel.get_peft_model(model, **lora_config_dict)
        FastVisionModel.for_training(self.model)
        self.processor = AutoProcessor.from_pretrained(self.config.BASE_MODEL_ID)
        self.tokenizer = tokenizer # Tokenizer is returned by from_pretrained
        self.logger.info("Model initialized, LoRA applied, and prepared for training.")
        return self.model, self.tokenizer, self.processor

    def train_model(self, continue_experiment: str | None = None, checkpoint_path: str | None = None) -> any:
        """
        Configures and runs the UnslothTrainer to finetune the model.
        Relies on `initialize_and_prepare_model` and `prepare_data_splits` 
        having been called.

        Args:
            continue_experiment (str | None): Path to an existing experiment directory to continue training from.
            checkpoint_path (str | None): Path to a specific checkpoint within the experiment directory to resume from.
                                         If None and continue_experiment is set, will use the latest checkpoint.

        Returns:
            The trained model instance (modified in-place).
        """
        self.logger.info("Configuring UnslothTrainer and starting model training...")
        if not all([self.model, self.tokenizer, self.processor, self.train_ds, self.eval_ds]):
            self.logger.error("Model, tokenizer, processor, or data splits not ready before training.")
            raise ValueError("Model/data not properly initialized before calling train_model.")

        dc = UnslothVisionDataCollator(self.model, processor=self.processor)
        
        current_training_args = self.config.TRAINING_ARGS_CONFIG.copy()
        finetuned_model_output_dir = self.exp_manager.get_finetuned_model_dir(self.config.FINETUNED_MODEL_DIR_NAME)
        current_training_args["output_dir"] = str(finetuned_model_output_dir)
        
        # Handle checkpoint resumption
        if continue_experiment:
            if checkpoint_path:
                resume_from_checkpoint = checkpoint_path
                self.logger.info(f"Will resume training from specified checkpoint: {checkpoint_path}")
            else:
                # Find the latest checkpoint in the experiment directory
                checkpoint_dirs = sorted(
                    [d for d in finetuned_model_output_dir.glob("checkpoint-*") if d.is_dir()],
                    key=lambda x: int(x.name.split("-")[-1]),
                    reverse=True
                )
                if checkpoint_dirs:
                    latest_checkpoint = checkpoint_dirs[0]
                    resume_from_checkpoint = str(latest_checkpoint)
                    self.logger.info(f"Will resume training from latest checkpoint: {latest_checkpoint}")
                else:
                    self.logger.warning("No checkpoints found in experiment directory. Starting from scratch.")
        
        self.logger.info(f"Training arguments configured. Output directory: {finetuned_model_output_dir}")
        self.logger.debug(f"Training args: {current_training_args}")
        
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.train_ds, 
            eval_dataset=self.eval_ds,   
            data_collator=dc,
            args=SFTConfig(**current_training_args),
        )
        self.logger.info(f"SFTTrainer initialized. Starting training...")
        if continue_experiment: 
            trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:   
            trainer_stats = trainer.train()
        self.logger.info("Training finished.")
        return self.model 

    @staticmethod
    def _sanitize_ollama_model_name(raw_name: str) -> str:
        """
        Sanitizes model names for Ollama tags.
        Ollama model names are lowercase and generally should avoid spaces/symbols.
        """
        sanitized = re.sub(r"[^a-z0-9._-]+", "-", raw_name.lower()).strip("-")
        return sanitized or "model"

    def _build_ollama_model_name(self, gguf_file: pathlib.Path) -> str:
        """
        Builds an Ollama model name from the configured prefix and GGUF quant suffix.
        """
        configured_prefix = str(getattr(self.config, "OLLAMA_MODEL_NAME_PREFIX", "")).strip()
        base_prefix = configured_prefix or self.exp_manager.current_experiment_path.name
        base_prefix = self._sanitize_ollama_model_name(base_prefix)
        quant_suffix = self._sanitize_ollama_model_name(gguf_file.stem.split(".")[-1])
        return f"{base_prefix}-{quant_suffix}"

    def save_ollama_artifacts(self, finetuned_model_dir: pathlib.Path) -> None:
        """
        Exports GGUF files via Unsloth and generates Ollama Modelfiles + helper commands.
        """
        if not getattr(self.config, "ENABLE_OLLAMA_EXPORT", False):
            self.logger.info("Skipping Ollama export because ENABLE_OLLAMA_EXPORT is False.")
            return

        if not hasattr(self.model, "save_pretrained_gguf"):
            raise AttributeError(
                "Model does not expose save_pretrained_gguf(). "
                "Upgrade Unsloth or disable ENABLE_OLLAMA_EXPORT."
            )

        quantization_methods = getattr(self.config, "OLLAMA_GGUF_QUANTIZATION_METHODS", ["q4_k_m"])
        if isinstance(quantization_methods, str):
            quantization_methods = [quantization_methods]
        quantization_methods = [str(m).strip() for m in quantization_methods if str(m).strip()]
        if not quantization_methods:
            self.logger.warning("OLLAMA_GGUF_QUANTIZATION_METHODS is empty. Skipping Ollama export.")
            return

        ollama_export_dir_name = str(getattr(self.config, "OLLAMA_EXPORT_DIR_NAME", "ollama")).strip() or "ollama"
        ollama_export_dir = finetuned_model_dir / ollama_export_dir_name
        ollama_export_dir.mkdir(parents=True, exist_ok=True)

        for quant_method in quantization_methods:
            self.logger.info(f"Saving GGUF for Ollama with quantization '{quant_method}' to: {ollama_export_dir}")
            self.model.save_pretrained_gguf(
                str(ollama_export_dir),
                self.tokenizer,
                quantization_method=quant_method,
            )

        gguf_files = sorted(ollama_export_dir.glob("*.gguf"))
        if not gguf_files:
            raise FileNotFoundError(
                f"Ollama export was requested, but no .gguf files were found in {ollama_export_dir}."
            )

        modelfile_parameters = getattr(self.config, "OLLAMA_MODELFILE_PARAMETERS", {}) or {}
        ollama_namespace = str(getattr(self.config, "OLLAMA_LIBRARY_NAMESPACE", "")).strip().strip("/")

        command_lines = [f"# Run commands from inside: {ollama_export_dir}"]
        created_model_names = []

        for gguf_file in gguf_files:
            model_name = self._build_ollama_model_name(gguf_file)
            modelfile_name = f"Modelfile.{model_name}"
            modelfile_path = ollama_export_dir / modelfile_name

            modelfile_lines = [f"FROM ./{gguf_file.name}"]
            for parameter_name, parameter_value in modelfile_parameters.items():
                if parameter_value is not None:
                    modelfile_lines.append(f"PARAMETER {parameter_name} {parameter_value}")

            modelfile_path.write_text("\n".join(modelfile_lines) + "\n", encoding="utf-8")

            command_lines.append(f"ollama create {model_name} -f {modelfile_name}")
            command_lines.append(f"ollama run {model_name}")
            if ollama_namespace:
                library_model_name = f"{ollama_namespace}/{model_name}"
                command_lines.append(f"ollama cp {model_name} {library_model_name}")
                command_lines.append(f"ollama push {library_model_name}")
            command_lines.append("")
            created_model_names.append(model_name)

        commands_path = ollama_export_dir / "ollama_commands.txt"
        commands_path.write_text("\n".join(command_lines).rstrip() + "\n", encoding="utf-8")
        self.logger.info(
            f"Ollama artifacts saved to: {ollama_export_dir}. "
            f"Generated models: {created_model_names}. Command helper: {commands_path}"
        )

    def save_model_artifacts(self) -> None:
        """
        Saves the finetuned LoRA adapter, tokenizer, and the merged 16-bit model
        to the experiment's finetuned model directory.
        Relies on `train_model` having been successfully executed.
        """
        self.logger.info("Saving model artifacts...")
        if not self.model or not self.tokenizer:
            self.logger.error("Model or tokenizer not available for saving. Was training successful?")
            raise ValueError("Model or tokenizer not available.")

        FastVisionModel.for_inference(self.model)
        self.logger.info("Model converted to inference mode.")
        
        finetuned_model_dir = self.exp_manager.get_finetuned_model_dir(self.config.FINETUNED_MODEL_DIR_NAME)
        final_model_path = finetuned_model_dir / self.config.LORA_CHECKPOINT_NAME 
        tokenizer_path = finetuned_model_dir / "tokenizer"
        merged_model_path = finetuned_model_dir / "merged_16bit"

        self.logger.info(f"Saving final LoRA checkpoint to: {final_model_path}")
        self.model.save_pretrained(str(final_model_path))
        
        self.logger.info(f"Saving tokenizer to: {tokenizer_path}")
        self.tokenizer.save_pretrained(str(tokenizer_path))
        
        self.logger.info(f"Saving merged 16-bit model to: {merged_model_path}")
        self.model.save_pretrained_merged(str(merged_model_path), self.tokenizer)

        try:
            self.save_ollama_artifacts(finetuned_model_dir)
        except Exception as e:
            if getattr(self.config, "FAIL_ON_OLLAMA_EXPORT_ERROR", False):
                raise
            self.logger.error(
                f"Ollama export failed, continuing because FAIL_ON_OLLAMA_EXPORT_ERROR is False. Error: {e}",
                exc_info=True,
            )

        self.logger.info("Model artifacts saving complete.")

    def run_training_pipeline(self, continue_experiment: str | None = None, checkpoint_path: str | None = None) -> pathlib.Path:
        """
        Runs the complete training pipeline: data preparation, model initialization,
        training, and saving artifacts.

        Args:
            continue_experiment (str | None): Path to an existing experiment directory to continue training from.
            checkpoint_path (str | None): Path to a specific checkpoint within the experiment directory to resume from.
                                         If None and continue_experiment is set, will use the latest checkpoint.

        Returns:
            pathlib.Path: The path to the experiment directory where artifacts and logs 
                         for this training run are stored.
        """
        self.logger.info("Starting training pipeline...")
        
        # Step 1: Prepare data splits
        self.prepare_data_splits(self.config.OHS_PROMPT_PATH)
        self.save_data_split_info()
        
        # Step 2: Initialize and prepare model
        self.initialize_and_prepare_model()
        
        # Step 3: Train model
        self.train_model(continue_experiment=continue_experiment, checkpoint_path=checkpoint_path)
        
        # Step 4: Save model artifacts
        self.save_model_artifacts()
        
        self.logger.info("Training pipeline completed successfully.")
        return self.exp_manager.current_experiment_path 
