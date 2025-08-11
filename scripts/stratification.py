import numpy as np
import pandas as pd

from sklearn.metrics import recall_score # type: ignore
from imblearn.metrics import specificity_score # type: ignore
from sklearn.metrics import precision_score # type: ignore
from sklearn.metrics import f1_score # type: ignore

def calc_strat_stats(y_true, y_proba, resolution=0.05, ignore_amber=False):
    """
    Calculate statistics based on stratified risk levels and find optimal thresholds.

    Args:
    - y_true: true labels (0 or 1).
    - y_proba: Predicted probabilities for the positive class.
    - resolution: Step size for threshold search.
    - ignore_amber: If True, only consider Green and Red categories. In this case, the probability will be used for ROC AUC instead of the predicted labels.

    Returns:
    - DataFrame with columns: green, amber, and metrics
    """

    # Define thresholds
    green_thresholds = np.arange(0.05, 0.55, resolution)
    amber_thresholds = 1 - green_thresholds  # Inverse relation for simplification

    results = []
    for green in green_thresholds:
        for amber in amber_thresholds:
            # Ensure proper threshold logic
            if green < amber:
                # Stratify based on thresholds
                stratified = np.select(
                    [y_proba <= green,
                     (y_proba > green) & (y_proba <= amber)],
                    ['Green', 'Amber'],
                    default='Red'
                )

                if ignore_amber:
                    
                    # only consider Green and Red
                    green_and_red_idx = (stratified == 'Green') | (stratified == 'Red')
                    stratified, y_proba_i, y_true_i = map(
                        lambda x: x[green_and_red_idx],
                        [stratified, y_proba, y_true]
                    )

                    # set total_count as validation set size
                    total_count = len(y_true)

                    if len(y_true_i) == 0:
                        continue
                    
                    if len(np.unique(y_true_i)) < 2:
                        continue

                    y_pred_i = np.where(stratified == 'Red', 1, 0)  # Consider Red as positive

                else:
                    y_true_i = y_true
                    y_pred_i = np.where(stratified == 'Red', 1, 0)  # Consider Red as positive
                    total_count = len(stratified)

                # Ensure binary classification
                if len(np.unique(y_true_i)) == 2:  # Check if it's binary

                    # Calculate metrics
                    sensitivity = recall_score(y_true_i, y_pred_i)
                    specificity = specificity_score(y_true_i, y_pred_i)
                    precision = precision_score(y_true_i, y_pred_i)
                    f1 = f1_score(y_true_i, y_pred_i)
                    youdens = sensitivity + specificity - 1

                    # Count number of samples in each group

                    green_count = np.sum(stratified == 'Green')
                    red_count = np.sum(stratified == 'Red')

                    # Calculate total number of samples and ratio

                    green_ratio = green_count/total_count if total_count > 0 else 0
                    red_ratio = red_count/total_count if total_count > 0 else 0

                    # Calculate amber count and ratio

                    amber_count = total_count - (green_count + red_count)
                    amber_ratio = amber_count / total_count if total_count > 0 else 0

                    # Debugging: print the counts for each category
                    print(f"Green threshold: {green}, Amber threshold: {amber}")
                    print(f"Total count: {total_count}, Green count: {green_count}, Amber count: {amber_count}, Red count: {red_count}")
                    
                    # Compile results
                    results.append({'green': green,
                                    'amber': amber,
                                    'sens': sensitivity,
                                    'spec': specificity,
                                    'prec': precision,
                                    'f1': f1,
                                    'J': youdens,
                                    'green_count': green_count,
                                    'amber_count': amber_count,
                                    'red_count': red_count,
                                    'green_ratio': green_ratio,
                                    'amber_ratio': amber_ratio,
                                    'red_ratio': red_ratio
                                    })

    # Convert to DataFrame and return
    results_df = pd.DataFrame(results).sort_values(by='J', ascending=False)
    return results_df

def find_optimal_threshold(mean_results, max_diff_ratio=0.5, max_amber_ratio=0.5):
    """
    Find the optimal threshold based on a balance between the highest Youden's J statistic,
    with additional constraints on the difference between the green and amber thresholds and the amber ratio.

    Args:
    - mean_results: DataFrame containing the metrics (J, Amber ratio, etc.).
    - max_diff_ratio: Maximum allowed value for difference between green and amber thresholds.
    - max_amber_ratio: Maximum allowed value for percentage of samples assigned amber label.

    Returns:
    - Dictionary containing the optimal thresholds, including Youden's J, max_diff_ratio, and max_amber_ratio.
    """
    # Filter based on constraints:

    # max_diff_ratio and max_amber_ratio constraint
    filtered_results = mean_results[
        (abs(mean_results['green'] - mean_results['amber']) <= max_diff_ratio) &
        (np.isclose(mean_results['green'], 1 - mean_results['amber'])) &
         (abs(mean_results[('amber_ratio', 'mean')]) <= max_amber_ratio)
         ]

    # If no rows meet the criteria, return None or an appropriate message
    if filtered_results.empty:
        print("No valid thresholds found based on the given constraints.")
        return None

    # Find row with highest objective value
    best_row = filtered_results.loc[filtered_results[('J', 'mean')].idxmax()]
    
    # Return the optimal thresholds along with the metrics
    optimal_threshold = {
        'green': best_row['green'],
        'amber': best_row['amber'],
        'sens_mean': best_row[('sens', 'mean')],
        'sens_sd': best_row[('sens', 'std')],
        'spec_mean': best_row[('spec', 'mean')],
        'spec_sd': best_row[('spec', 'std')],
        'prec_mean': best_row[('prec', 'mean')],
        'prec_sd': best_row[('prec', 'std')],
        'f1_mean': best_row[('f1', 'mean')],
        'f1_sd': best_row[('f1', 'std')],
        'J_mean': best_row[('J', 'mean')],
        'J_sd': best_row[('J', 'std')],
        'green_ratio_mean': best_row[('green_ratio', 'mean')],
        'green_ratio_sd': best_row[('green_ratio', 'std')],
        'amber_ratio_mean': best_row[('amber_ratio', 'mean')],
        'amber_ratio_sd': best_row[('amber_ratio', 'std')],
        'red_ratio_mean': best_row[('red_ratio', 'mean')],
        'red_ratio_sd': best_row[('red_ratio', 'std')],
        }
    return optimal_threshold

def evaluate_on_test_set(scores, optimal_thresholds, ignore_amber=True):

    y_true = scores['state']
    y_proba = scores['probability']

    # Get the optimal green and amber thresholds
    green_threshold = optimal_thresholds['green']
    amber_threshold = optimal_thresholds['amber']

    # Stratify test set based on thresholds
    stratified = np.select(
        [y_proba <= green_threshold, (y_proba > green_threshold) & (y_proba <= amber_threshold)], ['Green', 'Amber'], default='Red')

    if ignore_amber:
    
        # only consider Green and Red
        green_and_red_idx = (stratified == 'Green') | (stratified == 'Red')
        stratified, y_proba_i, y_true_i = map(
            lambda x: x[green_and_red_idx],
            [stratified, y_proba, y_true])

        # set total_count as validation set size
        total_count = len(y_true)

        if len(y_true_i) == 0:
            print("No samples left after filtering Amber; skipping evaluation.")
            return None
                        
        if len(np.unique(y_true_i)) < 2:
            print("Not enough classes present after filtering Amber; skipping evaluation.")
            return None

        y_pred_i = np.where(stratified == 'Red', 1, 0)  # Consider Red as positive

    else:
        y_true_i = y_true
        y_pred_i = np.where(stratified == 'Red', 1, 0)  # Consider Red as positive
        total_count = len(stratified)

    # Ensure binary classification
    if len(np.unique(y_true_i)) == 2:  # Check if it's binary

        # Calculate metrics on the test set
        sensitivity = recall_score(y_true_i, y_pred_i)
        specificity = specificity_score(y_true_i, y_pred_i)
        precision = precision_score(y_true_i, y_pred_i)
        f1 = f1_score(y_true_i, y_pred_i)
        youdens = sensitivity + specificity - 1

        # Count number of samples in each group

        green_count = np.sum(stratified == 'Green')
        red_count = np.sum(stratified == 'Red')

        # Calculate total number of samples and ratio

        green_ratio = green_count/total_count if total_count > 0 else 0
        red_ratio = red_count/total_count if total_count > 0 else 0

        # Calculate amber count and ratio

        amber_count = total_count - (green_count + red_count)
        amber_ratio = amber_count / total_count if total_count > 0 else 0

        # Debugging: print the counts for each category
        print(f"Green threshold: {green_threshold}, Amber threshold: {amber_threshold}")
        print(f"Total count: {total_count}, Green count: {green_count}, Amber count: {amber_count}, Red count: {red_count}")
        print(f"Ratios: Green ratio: {green_ratio}, Amber ratio: {amber_ratio}, Red ratio: {red_ratio}")
        print(f"Sensitivity: {sensitivity:.4f}")
        print(f"Specificity: {specificity:.4f}")
        print(f"Precision: {precision:.4f}")
                        
        # Compile results
        final_results = {'green': green_threshold,
                         'amber': amber_threshold,
                        'sens': sensitivity,
                                         'spec': specificity,
                                         'prec': precision,
                                         'f1': f1,
                                         'J': youdens,
                                         'green_count': green_count,
                                         'amber_count': amber_count,
                                         'red_count': red_count,
                                         'green_ratio': green_ratio,
                                         'amber_ratio': amber_ratio,
                                         'red_ratio': red_ratio
                                         }
                
    return final_results

def assign_traffic_light(y_proba, green_threshold=0.25, amber_threshold=0.75):
    """
    Assign traffic light labels based on predicted probabilities.

    Args:
    - y_proba: Predicted probabilities for the positive class.
    - green_threshold: Probability threshold for the 'Green' label.
    - amber_threshold: Probability threshold for the 'Amber' label.

    Returns:
    - traffic_lights: Array of traffic light labels ('Green', 'Amber', 'Red').
    """
    traffic_lights = np.empty(len(y_proba), dtype=object)

    for i, proba in enumerate(y_proba):
        if proba <= green_threshold:
            traffic_lights[i] = 'Green'
        elif proba <= amber_threshold:
            traffic_lights[i] = 'Amber'
        else:
            traffic_lights[i] = 'Red'

    return traffic_lights
