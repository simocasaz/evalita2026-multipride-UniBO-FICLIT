from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    PreTrainedTokenizer,
)
from transformers.trainer_utils import BestRun
import datasets as ds
import os
from collections.abc import Callable
import wandb


def run_hyperparameter_search(
    model_name: str,
    project_name: str,
    train_dataset: ds.Dataset,
    eval_dataset: ds.Dataset,
    tokenizer: PreTrainedTokenizer,
    hp_space: Callable,
    compute_metrics: Callable,
    model_init: Callable,
    compute_objective: Callable,
    n_trials: int,
    output_dir: str,
    logging_dir: str,
    seed: int = 42,
) -> tuple[BestRun, Trainer]:
    """
    Sets up the Trainer and executes the HPS using Optuna and W&B.

    NOTE: You must pass your tokenized datasets (train_dataset, eval_dataset)
    and your initialized tokenizer to this function.
    """
    # Create dirs for saving and logging if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(logging_dir, exist_ok=True)

    # Initialize wandb with correct project name and entity
    wandb.init(project=project_name, entity=os.getenv("WANDB-ENTITY"))

    # Initialize data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # These settings are shared across all trials unless overridden by hp_space
    training_args = TrainingArguments(
        logging_strategy="steps",  # Set logging strategy to steps
        logging_steps=20,
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=32,  # This is the default, but Optuna will override
        num_train_epochs=10,
        weight_decay=0.01,  # Default, Optuna will override
        learning_rate=2e-5,  # Default, Optuna will override
        warmup_steps=0,
        fp16=True,  # Enables half-precision training (FP16) for speed/memory efficiency
        # Logging & Evaluation Setup (CRITICAL for W&B and HPS)
        evaluation_strategy="steps",
        eval_steps=20,
        save_strategy="steps",
        save_steps=20,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        # W&B INTEGRATION
        report_to=["wandb"],  # Logs all metrics and trial configs to W&B
        disable_tqdm=False,
        seed=seed,
    )

    # 3. Initialize Trainer with Early Stopping Callback
    # Early Stopping is key to stopping models that overfit early.
    trainer = Trainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init,  # Pass the model initialization function
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=data_collator,
    )

    # 4. Execute Hyperparameter Search
    print(f"\n--- Starting Optuna HPS with {project_name} on W&B ---")

    best_run = trainer.hyperparameter_search(
        hp_space=hp_space,  # Your function defining the search space
        compute_objective=compute_objective,  # Your function defining the goal (Macro F1)
        direction="maximize",  # Maximize the Macro F1
        backend="optuna",  # Specify the backend
        n_trials=n_trials,  # Set based on your schedule (10-15 trials for Phase 1)
        # Pass a custom name function for better W&B visibility
        hp_name=lambda trial: f"{model_name}-Trial-{trial.number}",
    )

    print("\n--- HPS Complete. Best Run Found ---")
    print(best_run)

    # --- 5. Final Retrain of the Best Model (WITH OOM FIX) ---

    SAFE_BATCH_SIZE_FOR_RETRAIN = 16

    # Extract the best hyperparameters
    best_params = best_run.hyperparameters
    print(f"\n--- Preparing to retrain with Best Hyperparameters: {best_params} ---")

    # Update training args with the best found parameters
    for n, v in best_params.items():
        setattr(training_args, n, v)

    # --- START: OOM FIX ---
    # Check if the best batch size is larger than our safe limit
    optuna_batch_size = best_params["per_device_train_batch_size"]

    if optuna_batch_size > SAFE_BATCH_SIZE_FOR_RETRAIN:
        # Calculate accumulation steps to preserve the effective batch size
        # (e.g., if Optuna found 32 and safe is 16, this will be 2)
        grad_acc_steps = optuna_batch_size // SAFE_BATCH_SIZE_FOR_RETRAIN

        # Override the TrainingArguments with safe values
        setattr(
            training_args, "per_device_train_batch_size", SAFE_BATCH_SIZE_FOR_RETRAIN
        )
        setattr(training_args, "gradient_accumulation_steps", grad_acc_steps)

        print(
            f"Applying OOM Fix: Effective Batch Size {optuna_batch_size} (Device Batch: {SAFE_BATCH_SIZE_FOR_RETRAIN}, Grad Acc: {grad_acc_steps})"
        )
        # --- END: OOM FIX ---

    # Re-initialize the Trainer with the (now OOM-safe) best parameters
    # This block is the one you asked about. We MUST use model_init
    # to get a fresh, untrained model for the final training run.
    best_trainer = Trainer(
        args=training_args,  # training_args now contains the best params + OOM fix
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init,  # Re-initializes model from scratch
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=data_collator,
    )

    # The function now RETURNS the trainer object (containing the best model yet to be trained)
    # The user is responsible for training and saving the model externally.
    return best_run, best_trainer


def train_save_best_model(best_trainer, save_path):
    best_trainer.train()
    best_trainer.save_model(save_path)
