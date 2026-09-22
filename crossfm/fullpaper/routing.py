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


def corrective_route(
    round2_router: RoutingResult,
    router_bank: np.ndarray,
    router_labels: np.ndarray,
    round2_target: RoutingResult,
    target_bank: np.ndarray,
    fallback_probability: np.ndarray,
    *,
    temperature: float = 1.0,
    strength: float = 1.0,
) -> RoutingResult:
    """A label-isolated third beat driven by round-2 residual reliability.

    View corrections are learned only from the router split.  At prediction time
    they are scaled by round-2 confidence and replayed as soft attention; target
    labels are never read.  The original preservation gate remains unchanged, so
    zero statistical evidence still delegates exactly to the LLM fallback.
    """

    router = np.clip(np.asarray(router_bank, dtype=np.float64), 1e-6, 1 - 1e-6)
    labels = np.asarray(router_labels, dtype=np.float64).reshape(-1, 1)
    if router.ndim != 2 or labels.shape[0] != router.shape[0]:
        raise ValueError("router bank and labels must have matching rows")
    target = np.asarray(target_bank, dtype=np.float64)
    if target.shape != round2_target.weights.shape:
        raise ValueError("target bank must match the round-2 view weights")
    view_loss = -np.mean(labels * np.log(router) + (1 - labels) * np.log(1 - router), axis=0)
    base_probability = np.clip(round2_router.probability, 1e-6, 1 - 1e-6)[:, None]
    base_loss = -np.mean(
        labels * np.log(base_probability) + (1 - labels) * np.log(1 - base_probability)
    )
    advantage = base_loss - view_loss
    scale = max(float(np.std(advantage)), 1e-8)
    advantage = (advantage - float(np.mean(advantage))) / scale
    confidence = np.abs(round2_target.probability - 0.5) * 2.0
    correction = float(strength) * confidence[:, None] * advantage[None, :]
    logits = round2_target.analytical_logits + correction
    weights = _softmax(logits / float(temperature))
    routed = np.sum(weights * target, axis=1)
    gate = round2_target.gate.copy()
    fallback = np.asarray(fallback_probability, dtype=np.float64).reshape(-1)
    probability = fallback.copy()
    active = gate > 0
    probability[active] = (
        (1.0 - gate[active]) * fallback[active] + gate[active] * routed[active]
    )
    return RoutingResult(probability, routed, weights, gate, logits)


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


def _probability_logit(probability: np.ndarray) -> np.ndarray:
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-5, 1 - 1e-5)
    return np.log(probability / (1.0 - probability))


class SoftLatentCodebookRouter:
    """Convex predictor with a soft codebook over frozen-model response states.

    The codebook is induced only from response geometry.  Its posterior remains
    continuous in the forward pass and modulates residuals around a global
    linear fusion model.  With one codeword the design matrix is exactly the
    response-bank-plus-LLM matrix used by SMR, which makes K=1 a meaningful
    capacity control rather than a differently parameterized baseline.
    """

    def __init__(
        self,
        *,
        codebook_size: int,
        temperature: float,
        regularization_c: float,
        seed: int,
        use_semantic: bool = True,
        hard_assignments: bool = False,
        shuffle_assignments: bool = False,
    ) -> None:
        if codebook_size < 1:
            raise ValueError("codebook_size must be positive")
        if temperature <= 0 or regularization_c <= 0:
            raise ValueError("temperature and regularization_c must be positive")
        self.codebook_size = int(codebook_size)
        self.temperature = float(temperature)
        self.regularization_c = float(regularization_c)
        self.seed = int(seed)
        self.use_semantic = bool(use_semantic)
        self.hard_assignments = bool(hard_assignments)
        self.shuffle_assignments = bool(shuffle_assignments)

    @staticmethod
    def _validate(
        response_bank: np.ndarray,
        llm_probability: np.ndarray,
        semantic_logits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bank = np.asarray(response_bank, dtype=np.float64)
        llm = np.asarray(llm_probability, dtype=np.float64).reshape(-1)
        semantic = np.asarray(semantic_logits, dtype=np.float64).reshape(-1)
        if bank.ndim != 2 or bank.shape[1] < 2:
            raise ValueError("response_bank must be [rows, >=2 views]")
        if llm.shape != (bank.shape[0],):
            raise ValueError("llm_probability must have one value per row")
        if semantic.shape != (bank.shape[1],):
            raise ValueError("semantic_logits must have one value per view")
        return bank, llm, semantic

    def _semantic_weights(self, semantic_logits: np.ndarray) -> np.ndarray:
        if not self.use_semantic:
            return np.full(len(semantic_logits), 1.0 / len(semantic_logits))
        shifted = semantic_logits - np.max(semantic_logits)
        weights = np.exp(shifted)
        return weights / weights.sum()

    def _code_features(
        self,
        bank: np.ndarray,
        llm: np.ndarray,
        semantic: np.ndarray,
    ) -> np.ndarray:
        semantic_weights = self._semantic_weights(semantic)
        uncertainty = -(bank * np.log(np.clip(bank, 1e-8, 1.0)) + (
            1.0 - bank
        ) * np.log(np.clip(1.0 - bank, 1e-8, 1.0)))
        return np.column_stack((
            _probability_logit(bank),
            _probability_logit(llm),
            bank.mean(axis=1),
            bank.std(axis=1),
            bank.max(axis=1) - bank.min(axis=1),
            bank @ semantic_weights,
            uncertainty @ semantic_weights,
        ))

    @staticmethod
    def _base_features(bank: np.ndarray, llm: np.ndarray) -> np.ndarray:
        return np.column_stack((bank, llm))

    def _posterior_from_scaled(
        self, scaled_features: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        squared_distance = self.kmeans_.transform(scaled_features) ** 2
        logits = -squared_distance / (self.temperature * self.distance_scale_)
        posterior = _softmax(logits)
        if self.hard_assignments:
            posterior = np.eye(self.codebook_size)[np.argmax(posterior, axis=1)]
        return posterior, np.min(squared_distance, axis=1)

    def _design(self, base: np.ndarray, posterior: np.ndarray) -> np.ndarray:
        if self.codebook_size == 1:
            return base
        centered = posterior - self.code_usage_[None, :]
        interactions = [centered[:, [index]] * base for index in range(self.codebook_size)]
        return np.column_stack((base, posterior[:, :-1], *interactions))

    def fit(
        self,
        *,
        response_bank: np.ndarray,
        llm_probability: np.ndarray,
        labels: np.ndarray,
        semantic_logits: np.ndarray,
    ) -> dict[str, Any]:
        from sklearn.cluster import KMeans
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        bank, llm, semantic = self._validate(
            response_bank, llm_probability, semantic_logits,
        )
        targets = np.asarray(labels, dtype=np.int8).reshape(-1)
        if targets.shape != (bank.shape[0],) or len(np.unique(targets)) != 2:
            raise ValueError("labels must be binary and have one value per row")
        if self.codebook_size > len(targets):
            raise ValueError("codebook_size cannot exceed the training rows")
        code_features = self._code_features(bank, llm, semantic)
        self.scaler_ = StandardScaler().fit(code_features)
        scaled = self.scaler_.transform(code_features)
        self.kmeans_ = KMeans(
            n_clusters=self.codebook_size,
            n_init=10,
            algorithm="lloyd",
            random_state=self.seed,
        ).fit(scaled)
        squared_distance = self.kmeans_.transform(scaled) ** 2
        self.distance_scale_ = max(
            float(np.median(np.min(squared_distance, axis=1))), 1e-8,
        )
        posterior, nearest_distance = self._posterior_from_scaled(scaled)
        self.code_usage_ = np.mean(posterior, axis=0)
        training_posterior = posterior
        if self.shuffle_assignments:
            permutation = np.random.default_rng(self.seed + 77).permutation(len(posterior))
            training_posterior = posterior[permutation]
        design = self._design(self._base_features(bank, llm), training_posterior)
        self.classifier_ = LogisticRegression(
            C=self.regularization_c,
            max_iter=2000,
            random_state=self.seed,
            solver="lbfgs",
        ).fit(design, targets)
        self.ood_distance_threshold_ = max(
            float(np.quantile(nearest_distance, 0.99)), 1e-8,
        )
        usage_entropy = entropy(self.code_usage_)
        return {
            "code_usage": self.code_usage_.tolist(),
            "code_usage_perplexity": float(np.exp(usage_entropy)),
            "minimum_code_mass": float(np.min(self.code_usage_)),
            "classifier_params": int(
                self.classifier_.coef_.size + self.classifier_.intercept_.size
            ),
            "codebook_values": int(self.kmeans_.cluster_centers_.size),
            "ood_distance_threshold": self.ood_distance_threshold_,
        }

    def predict(
        self,
        *,
        response_bank: np.ndarray,
        llm_probability: np.ndarray,
        semantic_logits: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        bank, llm, semantic = self._validate(
            response_bank, llm_probability, semantic_logits,
        )
        scaled = self.scaler_.transform(self._code_features(bank, llm, semantic))
        posterior, nearest_distance = self._posterior_from_scaled(scaled)
        probability = self.classifier_.predict_proba(
            self._design(self._base_features(bank, llm), posterior)
        )[:, 1]
        distance_ratio = nearest_distance / self.ood_distance_threshold_
        ood_confidence = np.ones(len(bank), dtype=np.float64)
        shifted = distance_ratio > 1.0
        ood_confidence[shifted] = np.exp(1.0 - distance_ratio[shifted])
        ood_confidence[distance_ratio >= 4.0] = 0.0
        return probability, {
            "posterior": posterior,
            "ood_confidence": ood_confidence,
            "nearest_distance": nearest_distance,
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
