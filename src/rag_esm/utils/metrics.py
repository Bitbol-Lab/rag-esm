import numpy as np
import torch

def compute_metrics(eval_pred):
    """
        Compute the metrics for the evaluation of the model.
    """
    logits, labels = eval_pred
    num_pred = np.sum(labels != -100).item()
    num_correct = np.sum(np.argmax(logits, axis=-1) == labels).item()
    cross_entropy = torch.nn.functional.cross_entropy(torch.from_numpy(logits).view(-1, logits.shape[-1]),
                                                        torch.from_numpy(labels).view(-1),
                                                        ignore_index=-100)
    perplexity = torch.exp(cross_entropy)
    
    return {"accuracy": num_correct / num_pred,
            "perplexity": perplexity.item(),
            "predictions_per_sequence": num_pred / logits.shape[0]}