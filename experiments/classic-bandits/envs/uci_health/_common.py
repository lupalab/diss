"""
Shared vLLM Infrastructure for MedGemma Human Simulators

This module contains FULLY SHARED functions that work across all datasets:
- vLLM model loading
- Sampling parameters
- k-NN retrieval logic
- Probability mapping utilities

These functions are dataset-agnostic and should NOT be modified per dataset.
"""

from __future__ import annotations

import sklearn.neighbors as skl_neighs
import torch as th


# ============================================================================
# k-NN Retrieval (FULLY SHARED)
# ============================================================================
def get_knn_indices(
    query_features: th.Tensor,
    training_features: th.Tensor,
    n_neighbors: int,
    active_feature_indices: list[int] | None = None,
) -> th.Tensor:
    """Get k-nearest neighbor indices using sklearn.

    This is dataset-agnostic - works on any feature tensor.

    Args:
        query_features: Query feature tensor [N, F] or [F]
        training_features: Training feature tensor [M, F]
        n_neighbors: Number of neighbors to retrieve
        active_feature_indices: Optional list of feature indices to use (for masking)

    Returns:
        Tensor of neighbor indices [N, k] or [k] if query is 1D
    """
    # Handle single query (1D tensor)
    if query_features.dim() == 1:
        query_features = query_features[None, :]
        squeeze_output = True
    else:
        squeeze_output = False
    # Use only active features if specified
    if active_feature_indices is not None:
        query_features = query_features[:, active_feature_indices]
        training_features = training_features[:, active_feature_indices]
    # Find k-nearest neighbors
    knn_indices = (
        skl_neighs.NearestNeighbors()
        .fit(training_features.numpy(force=True))
        .kneighbors(
            query_features.numpy(force=True),
            n_neighbors=n_neighbors,
            return_distance=False,
        )
    )
    knn_indices = th.as_tensor(knn_indices, dtype=th.long, device="cpu")
    if squeeze_output:
        knn_indices = knn_indices.squeeze(0)
    return knn_indices


# ============================================================================
# Probability Mapping (FULLY SHARED)
# ============================================================================
def map_risk_confidence_to_probability(
    risk: th.Tensor,
    confidence: th.Tensor,
    risk_scale: int = 4,
    confidence_scale: int = 4,
) -> th.Tensor:
    """Map risk and confidence scores to probability [0, 1].

    This mapping is task-agnostic - works for any risk assessment task
    using ordinal scales.

    Formula:
        base_prob = risk / risk_scale
        confidence_weight = (confidence / confidence_scale) * 0.5 + 0.5
        probability = base_prob * confidence_weight

    Args:
        risk: Tensor of risk values (0 to risk_scale)
        confidence: Tensor of confidence values (0 to confidence_scale)
        risk_scale: Maximum risk value (default: 4)
        confidence_scale: Maximum confidence value (default: 4)

    Returns:
        Tensor of probabilities [0, 1]

    Examples:
        risk=4, conf=4 → 1.0 * 1.0 = 1.00 (very high risk, very confident)
        risk=0, conf=4 → 0.0 * 1.0 = 0.00 (very low risk, very confident)
        risk=2, conf=2 → 0.5 * 0.75 = 0.375 (medium risk, medium confidence)
        risk=4, conf=0 → 1.0 * 0.5 = 0.50 (high risk, not confident → reduced)
    """
    # Normalize risk to [0, 1]
    base_prob = risk.float() / float(risk_scale)
    # Normalize confidence to [0.5, 1.0] (never fully discount)
    # Low confidence → 0.5 weight, High confidence → 1.0 weight
    confidence_weight = (confidence.float() / float(confidence_scale)) * 0.5 + 0.5
    # Combine: high risk + low confidence → reduced probability
    probability = base_prob * confidence_weight
    return probability


# ============================================================================
# Feature Masking Utilities (FULLY SHARED)
# ============================================================================
def apply_feature_mask(
    features: th.Tensor,
    mask: th.Tensor | None,
    missing_value: float = float("nan"),
) -> th.Tensor:
    """Apply feature mask to features tensor.

    Args:
        features: Feature tensor [N, F] or [F]
        mask: Binary mask tensor [N, F] or [F] (1=active, 0=masked)
              If None, all features are active
        missing_value: Value to use for masked features (default: nan)

    Returns:
        Masked feature tensor (same shape as input)
    """
    if mask is None:
        return features
    return th.where(mask.to(dtype=th.bool), features, missing_value)


def get_active_feature_indices(mask: th.Tensor) -> list[int]:
    """Get indices of active features from mask.

    Args:
        mask: Binary mask tensor [F] (1=active, 0=masked)

    Returns:
        List of active feature indices
    """
    return th.argwhere(mask == 1).flatten().tolist()


# ============================================================================
# Debugging Utilities (FULLY SHARED)
# ============================================================================
def print_vllm_debug_info(
    predictions: list[float],
    confidences: list[float],
    json_errors: int,
    total_samples: int,
    sample_outputs: list[str] | None = None,
) -> None:
    """Print debugging information for vLLM outputs.

    Args:
        predictions: List of prediction values
        confidences: List of confidence values
        json_errors: Number of JSON parsing errors
        total_samples: Total number of samples
        sample_outputs: Optional list of sample output texts
    """
    print("\n" + "=" * 80)
    print("DEBUGGING INFO")
    print("=" * 80)
    # Error rate
    error_pct = (json_errors / total_samples) * 100
    print(f"\nJSON parsing errors: {json_errors}/{total_samples} ({error_pct:.1f}%)")
    # Prediction distribution
    print("\nPredictions:")
    print(f"  Unique values: {len(set(predictions))}")
    if predictions:
        print(f"  Range: [{min(predictions):.2f}, {max(predictions):.2f}]")
        most_common = max(set(predictions), key=predictions.count)
        count = predictions.count(most_common)
        print(f"  Most common: {most_common} (appears {count} times)")
    # Confidence distribution
    print("\nConfidence:")
    print(f"  Unique values: {len(set(confidences))}")
    if confidences:
        print(f"  Range: [{min(confidences):.2f}, {max(confidences):.2f}]")
    # Sample outputs
    if sample_outputs:
        print("\n" + "-" * 80)
        print("SAMPLE OUTPUTS:")
        print("-" * 80)
        for i, output in enumerate(sample_outputs[:5]):
            print(f"\nSample {i+1}:")
            print(output[:500])  # First 500 chars
            print("-" * 80)
    print("\n" + "=" * 80)
    if json_errors > 0:
        print(f"\n⚠️  JSON errors: {json_errors}/{total_samples} ({error_pct:.1f}%)")
    print(f"\n✅ All {total_samples} samples processed!")
