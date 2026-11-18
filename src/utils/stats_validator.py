from typing import Any, Sequence
import json
import re
import math
import numpy as np

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def _betacf(a: float, b: float, x: float, max_iter: int = 100, eps: float = 1e-12) -> float:
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        del_h = d * c
        h *= del_h
        if abs(del_h - 1.0) < eps:
            break
    return h


def _ibeta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    else:
        return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_pvalue(t: float, df: float) -> float:
    x = df / (df + t * t)
    return float(_ibeta(df / 2.0, 0.5, x))


def _t_ppf_975(df: float) -> float:
    z = 1.959963984540054
    return float(z + (z**3 + z) / (4.0 * df) + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2))


def validate_experiment(
    baseline_values: Sequence[float],
    experiment_values: Sequence[float],
) -> dict[str, Any]:
    """Compare experiment results with a baseline using Welch's t-test."""
    baseline = np.asarray(baseline_values, dtype=float)
    experiment = np.asarray(experiment_values, dtype=float)

    if baseline.ndim != 1 or experiment.ndim != 1:
        raise ValueError("Both inputs must be one-dimensional lists of numbers.")
    if len(baseline) < 2 or len(experiment) < 2:
        raise ValueError("Each group must contain at least two values.")
    if not np.isfinite(baseline).all() or not np.isfinite(experiment).all():
        raise ValueError("Values must be finite numbers.")
    if np.var(baseline, ddof=1) == 0 and np.var(experiment, ddof=1) == 0:
        raise ValueError("At least one group must contain some variation.")

    baseline_variance = float(np.var(baseline, ddof=1))
    experiment_variance = float(np.var(experiment, ddof=1))
    baseline_mean_variance = baseline_variance / len(baseline)
    experiment_mean_variance = experiment_variance / len(experiment)

    mean_difference = float(np.mean(experiment) - np.mean(baseline))

    # Welch's degrees of freedom for the confidence interval.
    df_numerator = (baseline_mean_variance + experiment_mean_variance) ** 2
    df_denominator = (
        baseline_mean_variance**2 / (len(baseline) - 1)
        + experiment_mean_variance**2 / (len(experiment) - 1)
    )
    degrees_of_freedom = df_numerator / df_denominator

    standard_error = float(np.sqrt(
        baseline_mean_variance + experiment_mean_variance
    ))

    if HAVE_SCIPY:
        test_result = stats.ttest_ind(
            experiment,
            baseline,
            equal_var=False,
        )
        p_value = float(test_result.pvalue)
        test_statistic = float(test_result.statistic)
        critical_value = float(stats.t.ppf(0.975, degrees_of_freedom))
    else:
        test_statistic = mean_difference / standard_error if standard_error > 0 else 0.0
        p_value = _t_pvalue(test_statistic, degrees_of_freedom)
        critical_value = _t_ppf_975(degrees_of_freedom)

    confidence_interval = [
        float(mean_difference - critical_value * standard_error),
        float(mean_difference + critical_value * standard_error),
    ]

    # Cohen's d uses the pooled standard deviation.
    pooled_variance = (
        (len(baseline) - 1) * baseline_variance
        + (len(experiment) - 1) * experiment_variance
    ) / (len(baseline) + len(experiment) - 2)
    effect_size = mean_difference / np.sqrt(pooled_variance)

    is_significant = p_value < 0.05
    if is_significant:
        summary = "The experiment shows a statistically significant difference from the baseline."
    else:
        summary = "The experiment does not show a statistically significant difference from the baseline."

    return {
        "p_value": p_value,
        "is_significant": is_significant,
        "confidence_interval_95": confidence_interval,
        "effect_size_cohens_d": float(effect_size),
        "mean_difference": mean_difference,
        "test_statistic": test_statistic,
        "summary": summary,
        "n_baseline": int(len(baseline)),
        "n_treatment": int(len(experiment)),
    }


def compute_statistics_from_output(output: str) -> dict[str, Any]:
    match = re.search(
        r"^RESULTS_JSON:\s*(\{.*\})\s*$",
        output or "",
        re.MULTILINE,
    )
    if not match:
        return {}

    try:
        data = json.loads(match.group(1))
        return validate_experiment(data["baseline"], data["treatment"])
    except (ValueError, KeyError, TypeError):
        return {}