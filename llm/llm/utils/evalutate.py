
import os
import torch
import numpy as np
from transformers import glue_compute_metrics as compute_metrics


def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')

# Evaluate the quantized model
def evaluate(model, dataset):
    correct = 0
    total = 0
    preds = []
    labels = []

    with torch.no_grad():
        for batch in dataset:
            inputs = {k: v.unsqueeze(0) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            label = batch["label"].item()
            
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=1).item()
            
            preds.append(prediction)
            labels.append(label)

            if prediction == label:
                correct += 1
            total += 1

    accuracy = correct / total
    metrics = compute_metrics({"predictions": np.array(preds), "label_ids": np.array(labels)})
    
    print(f"Quantized Model Accuracy: {accuracy:.4f}")
    print(f"Quantized Model F1 Score: {metrics['f1']:.4f}")
    return metrics
