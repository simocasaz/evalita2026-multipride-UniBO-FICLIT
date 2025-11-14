from transformers import (
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
import os


def run_hyperparameter_search(
    model_name,
    train_dataset,
    eval_dataset,
    wandb_project_name,
    tokenizer,
    hp_space,
    compute_metrics,
    model_init,
    compute_objective,
    n_trials,
    output_dir,
    logging_dir,
):
    """
    Sets up the Trainer and executes the HPS using Optuna and W&B.

    NOTE: You must pass your tokenized datasets (train_dataset, eval_dataset)
    and your initialized tokenizer to this function.
    """

    os.environ["WANDB_PROJECT"] = wandb_project_name

    # 1b. Initialize Data Collator (CRITICAL for dynamic padding)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 2. Define Training Arguments
    # These settings are shared across all trials unless overridden by hp_space
    training_args = TrainingArguments(
        output_dir=output_dir,
        logging_dir=logging_dir,
        per_device_train_batch_size=32,  # This is the default, but Optuna will override
        num_train_epochs=5,  # Default, Optuna will override
        weight_decay=0.01,  # Default, Optuna will override
        learning_rate=2e-5,  # Default, Optuna will override
        warmup_steps=0,
        # Mixed Precision Training
        fp16=True,  # <-- ADDED: Enables half-precision training (FP16) for speed/memory efficiency
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
    print(f"\n--- Starting Optuna HPS with {wandb_project_name} on W&B ---")

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

    # --- 5. Final Retrain of the Best Model ---

    # Extract the best hyperparameters
    best_params = best_run.hyperparameters
    print(f"\n--- Retraining Model with Best Hyperparameters: {best_params} ---")

    # Update training args with the best found parameters
    for n, v in best_params.items():
        setattr(training_args, n, v)

    # Re-initialize the Trainer with the best parameters and a fresh model
    best_trainer = Trainer(
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        model_init=model_init,  # Re-initializes model from scratch
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        data_collator=data_collator,
    )

    # Re-train the model using the optimal settings
    best_trainer.train()

    # The function now RETURNS the trainer object (containing the best model)
    # The user is responsible for saving the model externally.
    return best_run, best_trainer
