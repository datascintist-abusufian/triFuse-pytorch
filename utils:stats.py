import numpy as np
import pandas as pd
from scipy import stats
from sklearn.utils import resample


def paired_t_test(data1, data2, alpha=0.05):
    """
    Perform paired t-test with effect size and confidence intervals.
    
    Args:
        data1: Array of values for method 1
        data2: Array of values for method 2
    
    Returns:
        Dictionary with test results
    """
    diff = np.array(data1) - np.array(data2)
    n = len(diff)
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    se = std_diff / np.sqrt(n)
    
    t_stat = mean_diff / se
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))
    
    # Confidence interval
    t_critical = stats.t.ppf(1 - alpha/2, df)
    ci_low = mean_diff - t_critical * se
    ci_high = mean_diff + t_critical * se
    
    # Effect size (Cohen's d)
    cohens_d = mean_diff / (std_diff if std_diff > 0 else 1e-8)
    
    return {
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        't_stat': t_stat,
        'df': df,
        'p_value': p_value,
        'ci_low': ci_low,
        'ci_high': ci_high,
        'cohens_d': cohens_d,
        'significant': p_value < alpha
    }


def holm_correction(p_values, alpha=0.05):
    """
    Perform Holm-Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values
        alpha: Family-wise error rate
    
    Returns:
        Adjusted p-values
    """
    p_array = np.array(p_values)
    n = len(p_array)
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]
    
    adjusted = np.ones(n)
    for i in range(n):
        adjusted_p = sorted_p[i] * (n - i)
        adjusted_p = min(adjusted_p, 1.0)
        adjusted[sorted_indices[i]] = adjusted_p
    
    return adjusted


def bootstrap_confidence_interval(data, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for mean."""
    means = []
    for _ in range(n_bootstrap):
        sample = resample(data)
        means.append(np.mean(sample))
    
    lower = np.percentile(means, (1 - ci) / 2 * 100)
    upper = np.percentile(means, (1 + ci) / 2 * 100)
    
    return lower, upper


def wilcoxon_test(data1, data2):
    """Perform Wilcoxon signed-rank test."""
    diff = np.array(data1) - np.array(data2)
    diff = diff[diff != 0]  # Remove zeros
    if len(diff) == 0:
        return {'statistic': 0, 'p_value': 1.0, 'significant': False}
    
    statistic, p_value = stats.wilcoxon(diff)
    return {
        'statistic': statistic,
        'p_value': p_value,
        'significant': p_value < 0.05
    }


def compute_summary_stats(data):
    """Compute summary statistics."""
    data = np.array(data)
    return {
        'mean': data.mean(),
        'std': data.std(ddof=1),
        'min': data.min(),
        'max': data.max(),
        'median': np.median(data),
        'q1': np.percentile(data, 25),
        'q3': np.percentile(data, 75),
        'n': len(data)
    }