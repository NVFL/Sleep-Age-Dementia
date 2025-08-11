import numpy as np
import pandas as pd


def save_and_return_df(data: pd.DataFrame, filename: str = 'results.csv') -> pd.DataFrame:
    """
    Adds 'difference' and 'absolute_error' columns to the DataFrame provided,
    before saving it as a CSV file in the current directory and returning the updated DataFrame.

    Parameters:

        - data (pd.DataFrame): DataFrame with 'pred' and 'true' columns.
        - filename (str): Name of the CSV file to save. Default is 'data.csv'.

    Returns:

        - pd.DataFrame: Updated DataFrame with added columns.
    """
    if data['age_bin'].dtype == 'object':
        data['age_bin'] = data['age_bin'].apply(lambda x: f"{float(x.split(',')[0][1:])}-{float(x.split(',')[1][:-1])}")
    data['age_bin'] = data['age_bin'].astype('category')

    data['difference'] = data['pred_age'] - data['true_age']
    data['absolute_error'] = abs(data['true_age'] - data['pred_age'])
    data.to_csv(filename, index=False)
    return data


def compute_weighted_stats(df):
    # First, compute mean and std per person
    bins = [-np.inf, 50, 55, 60, 65, 70, 75, 80, np.inf]
    df['age_bin'] = pd.cut(df['true_age'], bins=np.sort(bins), right=False, include_lowest=False, ordered=True)
    person_stats = df.groupby(['age_bin', 'ids'], observed=True)['difference'].agg(
        person_mean='mean',
        person_std='std'
    ).reset_index()

    # Then, compute weighted mean and combined std per age_bin group
    def weighted_mean_std(sub_df):
        x = sub_df['person_mean']
        sigma = sub_df['person_std']

        # Remove entries with std <= 0 or NaN
        valid = sigma > 0
        x = x[valid]
        sigma = sigma[valid]

        if len(x) == 0:
            return pd.Series({
                'group_weighted_mean': np.nan,
                'group_combined_std': np.nan
            })

        weights = 1 / sigma**2
        weighted_mean = np.sum(x * weights) / np.sum(weights)
        combined_std = np.sqrt(1 / np.sum(weights))

        return pd.Series({
            'group_weighted_mean': weighted_mean,
            'group_combined_std': combined_std
        })

    group_stats = person_stats.groupby('age_bin').apply(weighted_mean_std).reset_index()

    return group_stats


def interval_to_str(interval):
    left = str(interval.left) if interval.left != -np.inf else "-inf"
    right = str(interval.right) if interval.right != np.inf else "inf"
    return f"{left}-{right}"
