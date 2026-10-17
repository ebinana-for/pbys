import numpy as np


def age_precision(preds, labels, x=5):
    """
    - 'labels': the actual age of the person
    - 'preds': the predicted age of the person
    - 'x': the precision range, default is 5 years.

    Returns the dataset precision of the age prediction and the number of samples used for the calculation.
    """

    if len(preds) != len(labels):
        raise ValueError("Predictions and labels must have the same length.")
    # Calculate the absolute difference between the predicted and actual age
    age_diff = np.abs(labels - preds)
    # Count the number of samples within the precision range
    within_precision = len(age_diff[age_diff <= x])
    # Calculate the total number of samples
    total_samples = len(age_diff)
    # Calculate the precision
    precision = within_precision / total_samples if total_samples > 0 else 0

    return precision, total_samples


def MAE(preds, labels):
    """
    - 'labels': the actual age of the person
    - 'preds': the predicted age of the person

    Returns the Mean Absolute Error (MAE) of the age prediction.
    """

    # Calculate the absolute error
    abs_error = np.abs(labels - preds)
    # Calculate the Mean Absolute Error
    mae = abs_error.mean() if len(abs_error) > 0 else 0

    return mae
