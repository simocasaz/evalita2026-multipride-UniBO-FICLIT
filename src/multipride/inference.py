from transformers import Trainer
import pandas as pd


def run_inference(
    model,
    tokenizer,
    test_dataset,
    output_path="predictions.tsv",
    output_file_type="tsv",
):
    """
    Runs the fine-tuned model on the test dataset and saves predictions.
    """
    trainer = Trainer(model=model, tokenizer=tokenizer)
    predictions = trainer.predict(test_dataset)

    preds = predictions.predictions.argmax(-1)

    df = pd.DataFrame({"id": test_dataset["id"], "prediction": preds})

    if output_file_type == "tsv":
        df.to_csv(output_path, index=False, sep="\t")

    else:
        df.to_csv(output_path, index=False)

    print(f"✅ Predictions saved to {output_path}")
