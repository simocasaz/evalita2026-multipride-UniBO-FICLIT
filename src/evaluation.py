import evaluate
import numpy as np


def compute_metrics(eval_preds):
    clf_metrics = evaluate.combine(["f1", "precision", "recall"])
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return clf_metrics.compute(predictions=predictions, references=labels)
