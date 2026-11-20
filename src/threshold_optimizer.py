import numpy as np
from sklearn.metrics import confusion_matrix


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    retention_cost: float = 50.0,
    customer_ltv: float = 600.0,
    offer_success_rate: float = 0.35,
) -> dict:
    """
    Finds the probability threshold that maximizes net business savings
    compared to taking no retention action.

    Args:
        y_true: Binary ground truth labels (1 = Churn, 0 = Retain).
        y_proba: Predicted churn probabilities from XGBoost/Scikit-Learn.
        retention_cost: Cost of sending a retention incentive ($).
        customer_ltv: Revenue lost if a customer churns ($).
        offer_success_rate: Probability that an targeted churner stays.

    Returns:
        Dictionary containing optimal threshold, net savings, and metrics.
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    max_net_savings = -np.inf
    best_metrics = {}

    for thresh in thresholds:
        y_pred = (y_proba >= thresh).astype(int)
        _tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        # Targeted customers = TP + FP (everyone flagged as churn)
        total_targeted = tp + fp
        campaign_cost = total_targeted * retention_cost

        # Revenue saved from true churners who accept the retention offer
        saved_revenue = tp * offer_success_rate * customer_ltv

        # Net financial gain compared to doing nothing
        net_savings = saved_revenue - campaign_cost

        if net_savings > max_net_savings:
            max_net_savings = net_savings
            best_threshold = round(float(thresh), 2)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            best_metrics = {
                "optimal_threshold": best_threshold,
                "net_savings_usd": round(float(net_savings), 2),
                "recall": round(float(recall), 4),
                "precision": round(float(precision), 4),
                "customers_targeted": int(total_targeted),
            }

    return best_metrics


if __name__ == "__main__":
    # Example usage during model evaluation
    rng = np.random.default_rng(42)
    sample_true = rng.binomial(1, 0.27, size=1000)
    sample_proba = np.clip(sample_true * 0.6 + rng.normal(0.2, 0.2, size=1000), 0.01, 0.99)

    result = find_optimal_threshold(sample_true, sample_proba)
    print("Cost-Optimized Operating Point:", result)