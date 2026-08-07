import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.stats import paired_t_test, holm_correction, bootstrap_confidence_interval


def load_results(method_dir, dataset):
    """Load evaluation results for a method."""
    df = pd.read_csv(os.path.join(method_dir, f'{dataset}_results.csv'))
    return df


def compare_methods(method1_dir, method2_dir, dataset, output_file):
    """Compare two methods statistically."""
    df1 = load_results(method1_dir, dataset)
    df2 = load_results(method2_dir, dataset)
    
    metrics = ['dice', 'volume_error', 'hd95', 'assd']
    results = []
    p_values = []
    
    for metric in metrics:
        data1 = df1[metric].values
        data2 = df2[metric].values
        
        # Paired t-test
        test_result = paired_t_test(data1, data2)
        results.append({
            'metric': metric,
            'mean1': data1.mean(),
            'mean2': data2.mean(),
            'mean_diff': test_result['mean_diff'],
            'ci_low': test_result['ci_low'],
            'ci_high': test_result['ci_high'],
            'cohens_d': test_result['cohens_d'],
            'p_value': test_result['p_value'],
            'significant': test_result['significant']
        })
        p_values.append(test_result['p_value'])
    
    # Holm correction
    adjusted_p = holm_correction(p_values)
    for i, result in enumerate(results):
        result['p_value_adjusted'] = adjusted_p[i]
    
    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    
    return df_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method1', type=str, required=True)
    parser.add_argument('--method2', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True, choices=['acdc', 'mscmrseg'])
    parser.add_argument('--output', type=str, default='statistical_analysis.csv')
    args = parser.parse_args()
    
    results = compare_methods(args.method1, args.method2, args.dataset, args.output)
    
    print('\nStatistical Comparison Results:')
    print('='*60)
    for _, row in results.iterrows():
        print(f'\n{row["metric"]}:')
        print(f'  Method 1: {row["mean1"]:.4f}')
        print(f'  Method 2: {row["mean2"]:.4f}')
        print(f'  Difference: {row["mean_diff"]:.4f} (95% CI: [{row["ci_low"]:.4f}, {row["ci_high"]:.4f}])')
        print(f'  Cohen\'s d: {row["cohens_d"]:.4f}')
        print(f'  p-value: {row["p_value"]:.6f} (adjusted: {row["p_value_adjusted"]:.6f})')
        print(f'  Significant: {row["significant"]}')
    print('='*60)


if __name__ == '__main__':
    main()