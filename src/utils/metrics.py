"""
Metrics calculation for next-location prediction.
Adapted from the provided metric calculation script.
"""

import torch
import numpy as np
from sklearn.metrics import f1_score


def get_performance_dict(return_dict):
    """Convert raw metric counts to percentages."""
    perf = {
        "correct@1": return_dict["correct@1"],
        "correct@3": return_dict["correct@3"],
        "correct@5": return_dict["correct@5"],
        "correct@10": return_dict["correct@10"],
        "rr": return_dict["rr"],
        "ndcg": return_dict["ndcg"],
        "f1": return_dict["f1"],
        "total": return_dict["total"],
    }

    perf["acc@1"] = perf["correct@1"] / perf["total"] * 100
    perf["acc@5"] = perf["correct@5"] / perf["total"] * 100
    perf["acc@10"] = perf["correct@10"] / perf["total"] * 100
    perf["mrr"] = perf["rr"] / perf["total"] * 100
    perf["ndcg"] = perf["ndcg"] / perf["total"] * 100

    return perf


def calculate_correct_total_prediction(logits, true_y):
    """
    Calculate accuracy@k, MRR, and NDCG metrics.
    
    Args:
        logits: Model predictions (B, num_classes)
        true_y: Ground truth labels (B,)
    
    Returns:
        Metrics array, true labels, top-1 predictions
    """
    top1 = []
    result_ls = []
    
    for k in [1, 3, 5, 10]:
        if logits.shape[-1] < k:
            k = logits.shape[-1]
        prediction = torch.topk(logits, k=k, dim=-1).indices
        
        if k == 1:
            top1 = torch.squeeze(prediction).cpu()

        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum().cpu().numpy()
        result_ls.append(top_k)
    
    # MRR
    result_ls.append(get_mrr(logits, true_y))
    # NDCG
    result_ls.append(get_ndcg(logits, true_y))
    # Total
    result_ls.append(true_y.shape[0])

    return np.array(result_ls, dtype=np.float32), true_y.cpu(), top1


def get_mrr(prediction, targets):
    """Calculate Mean Reciprocal Rank."""
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float()
    rranks = torch.reciprocal(ranks)
    return torch.sum(rranks).cpu().numpy()


def get_ndcg(prediction, targets, k=10):
    """Calculate Normalized Discounted Cumulative Gain."""
    index = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    ranks = (hits[:, -1] + 1).float().cpu().numpy()

    not_considered_idx = ranks > k
    ndcg = 1 / np.log2(ranks + 1)
    ndcg[not_considered_idx] = 0

    return np.sum(ndcg)
