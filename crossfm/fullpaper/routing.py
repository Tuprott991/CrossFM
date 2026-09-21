from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _softmax(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    shifted = values - np.max(values, axis=axis, keepdims=True)
    result = np.exp(shifted)
    return result / np.sum(result, axis=axis, keepdims=True)


def entropy(probability: np.ndarray, axis: int = -1) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0)
    return -np.sum(probability * np.log(probability), axis=axis)


@dataclass(frozen=True, slots=True)
class RoutingResult:
    probability: np.ndarray
    routed_probability: np.ndarray
    weights: np.ndarray
    gate: np.ndarray
    analytical_logits: np.ndarray


def analytical_route(
    response_bank: np.ndarray,
    semantic_logits: np.ndarray,
    statistical_posterior: np.ndarray,
    fallback_probability: np.ndarray,
    *,
    temperature: float = 1.0,
    gate_floor: float = 0.0,
) -> RoutingResult:
    """Posterior-codebook-residual routing with an exact preservation path.

    ``response_bank`` has shape ``[rows, views]``.  ``semantic_logits`` is either
    ``[views]`` or ``[rows, views]``.  ``statistical_posterior`` is a continuous
    distribution over the same views.  A uniform posterior has zero confidence and
    delegates exactly to ``fallback_probability``.
    """

    bank = np.asarray(response_bank, dtype=np.float64)
    if bank.ndim != 2 or bank.shape[1] < 2:
        raise ValueError("response_bank must be [rows, >=2 views]")
    posterior = np.asarray(statistical_posterior, dtype=np.float64)
    if posterior.ndim == 1:
        posterior = np.broadcast_to(posterior, bank.shape)
    if posterior.shape != bank.shape or np.any(posterior < 0):
        raise ValueError("statistical_posterior must match response_bank and be non-negative")
    sums = posterior.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("statistical_posterior rows must have positive mass")
    posterior = posterior / sums
    semantic = np.asarray(semantic_logits, dtype=np.float64)
    if semantic.ndim == 1:
        semantic = np.broadcast_to(semantic, bank.shape)
    if semantic.shape != bank.shape:
        raise ValueError("semantic_logits must have one value per view")
    fallback = np.asarray(fallback_probability, dtype=np.float64).reshape(-1)
    if fallback.shape != (bank.shape[0],):
        raise ValueError("fallback_probability must have one value per row")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    analytical_logits = semantic + np.log(np.clip(posterior, 1e-12, 1.0))
    weights = _softmax(analytical_logits / temperature)
    routed = np.sum(weights * bank, axis=1)
    maximum_entropy = np.log(bank.shape[1])
    confidence = 1.0 - entropy(posterior) / maximum_entropy
    confidence = np.where(confidence <= gate_floor + 1e-12, 0.0, confidence)
    gate = np.clip((confidence - gate_floor) / max(1e-12, 1.0 - gate_floor), 0.0, 1.0)
    probability = fallback.copy()
    active = gate > 0
    probability[active] = (
        (1.0 - gate[active]) * fallback[active] + gate[active] * routed[active]
    )
    return RoutingResult(probability, routed, weights, gate, analytical_logits)


def factorized_view_posterior(
    evidence: np.ndarray,
    *,
    temperature: float = 1.0,
    uniform_when_flat: bool = True,
) -> np.ndarray:
    evidence = np.asarray(evidence, dtype=np.float64)
    if evidence.ndim != 2:
        raise ValueError("evidence must be [rows, views]")
    centered = evidence - np.mean(evidence, axis=1, keepdims=True)
    flat = np.max(np.abs(centered), axis=1) <= 1e-12
    posterior = _softmax(centered / temperature)
    if uniform_when_flat:
        posterior[flat] = 1.0 / evidence.shape[1]
    return posterior


class ARPlusRouter:
    """Small learned correction that can only modify analytical view logits.

    Torch is imported lazily so protocol and dataset tests remain CPU-light.  The
    correction scale is initialized to zero, making the first forward pass exactly
    equal to AR.  The preservation gate is applied after routing and remains exact.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 32, seed: int = 0):
        import torch

        torch.manual_seed(seed)

        class Module(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.net = torch.nn.Sequential(
                    torch.nn.Linear(feature_dim, hidden_dim),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_dim, 1),
                )
                self.residual_scale = torch.nn.Parameter(torch.zeros(()))
                self.log_temperature = torch.nn.Parameter(torch.zeros(()))
                torch.nn.init.zeros_(self.net[-1].weight)
                torch.nn.init.zeros_(self.net[-1].bias)

            def forward(self, base_logits, features, bank, fallback, gate):
                correction = self.residual_scale * self.net(features).squeeze(-1)
                temperature = self.log_temperature.exp().clamp(0.25, 4.0)
                weights = torch.softmax((base_logits + correction) / temperature, dim=-1)
                routed = (weights * bank).sum(-1)
                mixed = (1.0 - gate) * fallback + gate * routed
                probability = torch.where(gate == 0, fallback, mixed)
                return probability, weights, correction

        self.module = Module()

    @property
    def trainable_params(self) -> int:
        return sum(parameter.numel() for parameter in self.module.parameters())

    def predict(
        self,
        *,
        analytical_logits: np.ndarray,
        features: np.ndarray,
        response_bank: np.ndarray,
        fallback_probability: np.ndarray,
        gate: np.ndarray,
        device: str = "cpu",
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        import torch

        self.module.to(device).eval()
        tensors = [
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in (
                analytical_logits, features, response_bank,
                fallback_probability, gate,
            )
        ]
        with torch.inference_mode():
            probability, weights, correction = self.module(*tensors)
        return probability.cpu().numpy(), {
            "weights": weights.cpu().numpy(),
            "correction": correction.cpu().numpy(),
        }

    def fit(
        self,
        *,
        analytical_logits: np.ndarray,
        features: np.ndarray,
        response_bank: np.ndarray,
        fallback_probability: np.ndarray,
        gate: np.ndarray,
        labels: np.ndarray,
        learning_rate: float = 3e-4,
        epochs: int = 100,
        anchor_kl: float = 0.02,
        correction_l2: float = 0.001,
        device: str = "cpu",
    ) -> dict[str, Any]:
        import torch

        self.module.to(device).train()
        base, feats, bank, fallback, gate_tensor, targets = [
            torch.as_tensor(value, dtype=torch.float32, device=device)
            for value in (
                analytical_logits, features, response_bank,
                fallback_probability, gate, labels,
            )
        ]
        anchor_weights = torch.softmax(base, dim=-1).detach()
        optimizer = torch.optim.AdamW(self.module.parameters(), lr=learning_rate)
        best, best_state = float("inf"), None
        for _ in range(epochs):
            optimizer.zero_grad(set_to_none=True)
            probability, weights, correction = self.module(base, feats, bank, fallback, gate_tensor)
            task_loss = torch.nn.functional.binary_cross_entropy(
                probability.clamp(1e-6, 1 - 1e-6), targets,
            )
            kl = torch.sum(
                weights * (
                    torch.log(weights.clamp_min(1e-8))
                    - torch.log(anchor_weights.clamp_min(1e-8))
                ), dim=-1,
            ).mean()
            loss = task_loss + anchor_kl * kl + correction_l2 * correction.square().mean()
            loss.backward()
            optimizer.step()
            numeric = float(loss.detach().cpu())
            if numeric < best:
                best = numeric
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in self.module.state_dict().items()
                }
        if best_state is not None:
            self.module.load_state_dict(best_state)
        return {
            "training_loss": best,
            "trainable_params": self.trainable_params,
            "residual_scale": float(self.module.residual_scale.detach().cpu()),
            "temperature": float(self.module.log_temperature.exp().detach().cpu()),
        }


def routing_features(
    response_bank: np.ndarray,
    semantic_logits: np.ndarray,
    posterior: np.ndarray,
) -> np.ndarray:
    bank = np.asarray(response_bank, dtype=np.float64)
    semantic = np.broadcast_to(np.asarray(semantic_logits, dtype=np.float64), bank.shape)
    posterior = np.broadcast_to(np.asarray(posterior, dtype=np.float64), bank.shape)
    disagreement = np.abs(bank - np.mean(bank, axis=1, keepdims=True))
    uncertainty = -(bank * np.log(np.clip(bank, 1e-8, 1.0)) + (
        1.0 - bank
    ) * np.log(np.clip(1.0 - bank, 1e-8, 1.0)))
    return np.stack((bank, disagreement, uncertainty, semantic, posterior), axis=-1)
