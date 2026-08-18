import torch
import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity


def build_metric_objective():
    loss = losses.MultiSimilarityLoss(
        alpha=1.0,
        beta=50,
        base=0.0,
        distance=DotProductSimilarity(),
    )
    miner = miners.MultiSimilarityMiner(
        epsilon=0.1,
        distance=CosineSimilarity(),
    )
    return loss, miner


def metric_loss(loss_fn, miner, descriptors, labels):
    mined_pairs = miner(descriptors, labels)
    return loss_fn(descriptors, labels, mined_pairs), mined_pairs


def cross_attention_alignment_loss(teacher_deltas, student_deltas):
    common_layers = sorted(set(teacher_deltas) & set(student_deltas))
    if not common_layers:
        source = teacher_deltas or student_deltas
        device = next(iter(source.values())).device if source else None
        return torch.tensor(0.0, device=device), 0
    values = [
        F.mse_loss(
            student_deltas[layer].float(),
            teacher_deltas[layer].detach().float(),
        )
        for layer in common_layers
    ]
    return torch.stack(values).mean(), len(common_layers)


ALIGNMENT_TARGETS = {
    "none",
    "global_descriptor",
    "final_local_features",
    "intervention_residual",
}


def distillation_alignment_loss(target, teacher_output, student_output):
    """Compute one of the matched Stage-1 distillation objectives."""
    if target not in ALIGNMENT_TARGETS:
        raise ValueError(
            f"unknown alignment target {target!r}; expected one of "
            f"{sorted(ALIGNMENT_TARGETS)}"
        )
    if target == "none":
        return student_output["descriptor"].sum() * 0.0, 0
    if target == "intervention_residual":
        return cross_attention_alignment_loss(
            teacher_output["xattn_deltas"],
            student_output["xattn_deltas"],
        )
    if target == "global_descriptor":
        teacher = teacher_output["descriptor"].detach().float()
        student = student_output["descriptor"].float()
        if teacher.shape != student.shape:
            raise ValueError(
                "global descriptor alignment requires matching shapes, got "
                f"teacher={tuple(teacher.shape)} student={tuple(student.shape)}"
            )
        return (1.0 - F.cosine_similarity(student, teacher, dim=-1)).mean(), 1

    teacher = teacher_output["feature_tokens"].detach().float()
    student = student_output["feature_tokens"].float()
    if teacher.shape != student.shape:
        raise ValueError(
            "final local feature alignment requires matching shapes, got "
            f"teacher={tuple(teacher.shape)} student={tuple(student.shape)}"
        )
    token_cosine = F.cosine_similarity(student, teacher, dim=-1)
    return (1.0 - token_cosine).mean(), teacher.shape[1]


def descriptor_agreement(teacher, student):
    if teacher.shape[0] < 2:
        return 0.0
    teacher_similarity = teacher.detach() @ teacher.detach().T
    student_similarity = student.detach() @ student.detach().T
    return F.cosine_similarity(
        teacher_similarity.flatten(),
        student_similarity.flatten(),
        dim=0,
    ).item()


def pair_statistics(mined_pairs, teacher_batch_size=None):
    positive_anchor, positive, negative_anchor, negative = mined_pairs
    result = {
        "positive_pairs": int(len(positive_anchor)),
        "negative_pairs": int(len(negative_anchor)),
    }
    if teacher_batch_size is not None:
        result["cross_positive_pairs"] = int(
            ((positive_anchor < teacher_batch_size) != (positive < teacher_batch_size))
            .sum()
            .item()
        )
        result["cross_negative_pairs"] = int(
            ((negative_anchor < teacher_batch_size) != (negative < teacher_batch_size))
            .sum()
            .item()
        )
    return result
