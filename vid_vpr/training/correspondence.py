import torch
import torch.nn.functional as F


def masked_sequence_summary(hidden, masks):
    """Compute one normalized VLM summary per padded sequence."""
    weights = masks.to(device=hidden.device, dtype=torch.float32).unsqueeze(-1)
    summary = (hidden.float() * weights).sum(dim=1)
    summary = summary / weights.sum(dim=1).clamp_min(1.0)
    return F.normalize(summary, p=2, dim=-1)


@torch.no_grad()
def hard_different_label_indices(hidden, masks, labels):
    """Choose the most similar VLM sequence belonging to another place."""
    if labels.ndim != 1 or labels.shape[0] != hidden.shape[0]:
        raise ValueError("labels must contain one value per VLM sequence")
    different_place = labels[:, None].ne(labels[None, :])
    if not different_place.any(dim=1).all():
        raise ValueError("each sample needs at least one different-place VLM candidate")

    summaries = masked_sequence_summary(hidden, masks)
    similarity = summaries @ summaries.T
    similarity.masked_fill_(~different_place, torch.finfo(similarity.dtype).min)
    return similarity.argmax(dim=1)


@torch.no_grad()
def same_label_partner_indices(labels):
    """Return another view of the same place and a mask of valid samples."""
    partners = torch.arange(labels.shape[0], device=labels.device)
    valid = torch.zeros(labels.shape[0], dtype=torch.bool, device=labels.device)
    for label in labels.unique():
        indices = torch.nonzero(labels.eq(label), as_tuple=False).flatten()
        if indices.numel() > 1:
            partners[indices] = indices.roll(1)
            valid[indices] = True
    return partners, valid


def correspondence_hinge_loss(correct, mismatched, labels, margin=0.02):
    """Make correct VLM conditioning more place-consistent than mismatching it."""
    partners, valid = same_label_partner_indices(labels)
    if not valid.any():
        return correct.sum() * 0.0, 0
    positive = F.cosine_similarity(correct, correct[partners], dim=-1)
    counterfactual = F.cosine_similarity(mismatched, correct[partners].detach(), dim=-1)
    loss = F.relu(float(margin) - positive + counterfactual)
    return loss[valid].mean(), int(valid.sum().item())


def descriptor_anchor_loss(current, reference):
    cosine = F.cosine_similarity(current.float(), reference.detach().float(), dim=-1)
    return (1.0 - cosine).clamp_min(0.0).mean()


def assess_rescue(baseline_r1, mode_recalls, thresholds):
    correct_r1 = float(mode_recalls["correct"][1])
    shuffled_r1 = float(mode_recalls["shuffled"][1])
    fixed_r1 = float(mode_recalls["fixed"][1])
    metrics = {
        "baseline_r1": float(baseline_r1),
        "correct_r1": correct_r1,
        "shuffled_r1": shuffled_r1,
        "fixed_r1": fixed_r1,
        "correct_drop": float(baseline_r1) - correct_r1,
        "correct_minus_shuffled": correct_r1 - shuffled_r1,
        "correct_minus_fixed": correct_r1 - fixed_r1,
    }
    checks = {
        "preserves_correct_recall": metrics["correct_drop"]
        <= float(thresholds.get("max_correct_r1_drop", 0.5)),
        "separates_shuffled": metrics["correct_minus_shuffled"]
        >= float(thresholds.get("min_correct_shuffled_gap", 1.0)),
        "separates_fixed": metrics["correct_minus_fixed"]
        >= float(thresholds.get("min_correct_fixed_gap", 0.5)),
    }
    return metrics, checks, all(checks.values())
