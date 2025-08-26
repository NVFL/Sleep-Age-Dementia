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


def compute_weighted_stats_age(df):
    """
    Computes the weighted mean and combined standard deviation of prediction errors
    grouped by age bins. First calculates per-person mean and standard deviation,
    then aggregates these statistics within each age bin.

    Parameters:
    
        - df (pd.DataFrame): DataFrame containing 'true_age', 'difference', and 'ids' columns.

    Returns:
    
        - pd.DataFrame: DataFrame with one row per age bin, containing: 'age_bin', 'group_weighted_mean', and 'group_combined_std'
    """
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
    """
    Converts a pandas Interval object into a string representation.

    Parameters:
    
        - interval (pd.Interval): Interval object with 'left' and 'right' boundaries.

    Returns:
    
        - str: String representation of the interval in the format 'left-right',
               where '-inf' and 'inf' are used for infinite bounds.
    """
    left = str(interval.left) if interval.left != -np.inf else "-inf"
    right = str(interval.right) if interval.right != np.inf else "inf"
    return f"{left}-{right}"

def compute_weighted_stats_dementia(df):
    """
    Computes the weighted mean and combined standard deviation of risk scores
    grouped by age bins. First calculates per-person mean and standard deviation,
    then aggregates these statistics within each age bin.

    Parameters:
    
        - df (pd.DataFrame): DataFrame containing 'Age Bin', 'Likelihood', and 'ID' columns.

    Returns:
    
        - pd.DataFrame: DataFrame with one row per age bin, containing: 'Age Bin', 'group_weighted_mean', and 'group_combined_std'
    """
    # First, compute mean and std per person
    person_stats = df.groupby(['Age Bin', 'ID'], observed=True)['Likelihood'].agg(
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

    group_stats = person_stats.groupby('Age Bin').apply(weighted_mean_std).reset_index()

    return group_stats

def interpret_prediction(label, adjusted_prob):
    """
    Interprets the prediction outcome based on the label and adjusted probability.

    Parameters:
    
        - label (str): The risk category label (e.g., 'Red', 'Amber', 'Green').
        - adjusted_prob (float): The adjusted probability score associated with the prediction.

    Returns:
    
        - str: A qualitative interpretation of the prediction.
    """
    if label == 'Red':
        if adjusted_prob >= 0:
            return "Highly Likely"
        else:
            return "Somewhat Likely"
    elif label == 'Amber':
        if adjusted_prob >= 0:
            return "Somewhat Likely"
        else:
            return "Not Likely"
    else:
        return "Not Likely"
