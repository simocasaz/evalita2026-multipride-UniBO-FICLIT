import evaluate
import numpy as np


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    f1 = evaluate.load("f1").compute(
        predictions=predictions, references=labels, average="macro"
    )
    precision = evaluate.load("precision").compute(
        predictions=predictions, references=labels, average="macro", zero_division=0
    )
    recall = evaluate.load("recall").compute(
        predictions=predictions, references=labels, average="macro", zero_division=0
    )
    return {**f1, **precision, **recall}
