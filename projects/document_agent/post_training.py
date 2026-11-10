"""Small-tensor SFT, DPO and SimPO objectives with explicit supervision boundaries."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _check(logits, labels):
    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError('Expected logits [B,T,V] and labels [B,T]')
    if logits.shape[1] < 2:
        raise ValueError('Causal shift requires at least two positions')
    if labels.dtype != torch.long:
        raise ValueError('Labels must be int64')


def sequence_log_probs(logits, labels, *, length_normalize=False):
    _check(logits, labels)
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~mask, 0)
    if ((safe_labels < 0) | (safe_labels >= logits.shape[-1])).any():
        raise ValueError('Unmasked target outside vocabulary')
    selected = F.log_softmax(shifted_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    selected = torch.where(mask, selected, torch.zeros_like(selected))
    sums = selected.sum(dim=-1)
    lengths = mask.sum(dim=-1)
    if length_normalize:
        sums = sums / lengths.clamp_min(1)
    return sums, lengths


def sft_shift_loss(logits, labels, *, reduction='token'):
    """Token mean and sample mean intentionally assign different sequence weights."""
    _check(logits, labels)
    targets = labels[:, 1:].contiguous()
    shifted_logits = logits[:, :-1, :].contiguous()
    valid = targets.ne(-100)
    if not valid.any():
        return logits.sum() * 0.0
    per_position = F.cross_entropy(shifted_logits.reshape(-1, logits.shape[-1]),
                                  targets.reshape(-1), ignore_index=-100, reduction='none')
    per_position = per_position.reshape_as(targets)
    if reduction == 'token':
        return per_position.sum() / valid.sum()
    if reduction == 'sample':
        counts = valid.sum(dim=-1)
        per_sample = per_position.sum(dim=-1) / counts.clamp_min(1)
        return per_sample[counts > 0].mean()
    if reduction == 'none':
        return per_position
    raise ValueError('Unknown SFT reduction')


def dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, *, beta=0.1):
    if beta <= 0:
        raise ValueError('beta must be positive')
    if not (policy_chosen.shape == policy_rejected.shape == reference_chosen.shape == reference_rejected.shape):
        raise ValueError('All sequence log-prob arrays must have matching shapes')
    reference_margin = (reference_chosen - reference_rejected).detach()
    margin = (policy_chosen - policy_rejected) - reference_margin
    return -F.logsigmoid(beta * margin).mean()


def simpo_loss(chosen_sum, rejected_sum, chosen_lengths, rejected_lengths, *, beta=2.0, gamma=0.5):
    if beta <= 0 or gamma < 0:
        raise ValueError('Invalid SimPO beta or gamma')
    if not (chosen_sum.shape == rejected_sum.shape == chosen_lengths.shape == rejected_lengths.shape):
        raise ValueError('SimPO inputs must have matching shapes')
    if (chosen_lengths <= 0).any() or (rejected_lengths <= 0).any():
        raise ValueError('SimPO needs non-empty responses')
    margin = beta * (chosen_sum / chosen_lengths - rejected_sum / rejected_lengths) - gamma
    return -F.logsigmoid(margin).mean()


def losses_from_logits(policy_chosen, policy_rejected, reference_chosen, reference_rejected,
                       chosen_labels, rejected_labels, *, beta=0.1, simpo_beta=2.0, gamma=0.5):
    pc, lc = sequence_log_probs(policy_chosen, chosen_labels)
    pr, lr = sequence_log_probs(policy_rejected, rejected_labels)
    with torch.no_grad():
        rc, _ = sequence_log_probs(reference_chosen, chosen_labels)
        rr, _ = sequence_log_probs(reference_rejected, rejected_labels)
    return {'dpo':dpo_loss(pc,pr,rc,rr,beta=beta),
            'simpo':simpo_loss(pc,pr,lc,lr,beta=simpo_beta,gamma=gamma),
            'policy_chosen_logps':pc, 'policy_rejected_logps':pr,
            'chosen_lengths':lc, 'rejected_lengths':lr}
