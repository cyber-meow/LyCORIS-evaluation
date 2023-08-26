import pandas as pd
from contextlib import contextmanager


@contextmanager
def display_all_rows():
    prev_option = pd.get_option("display.max_rows")
    pd.set_option("display.max_rows", None)
    yield
    pd.set_option("display.max_rows", prev_option)


@contextmanager
def display_all_columns():
    prev_option = pd.get_option("display.max_columns")
    pd.set_option("display.max_columns", None)
    yield
    pd.set_option("display.max_columns", prev_option)


def detect_systematically_worse(df, metric_triplets, threshold):
    """
    Returns the subframe containing rows for which all specified metric means
    are below the given threshold.

    Parameters:
    - df: Input DataFrame with multi-level columns
    - metric_triplets: List of triplets defining the metrics of interest
    - threshold: Value to compare the metric means against

    Returns:
    - subframe: DataFrame containing rows where all specified metric means
                are below the threshold
    """

    # Create a list to hold the conditions for each metric triplet
    conditions = []

    # Iterate through the list of metric triplets
    for triplet in metric_triplets:
        # Create a multi-index tuple for mean columns
        mean_column = triplet + ('mean',)

        # Check if this column exists in the DataFrame
        if mean_column not in df.columns:
            print(f"Column {mean_column} does not exist in the DataFrame.")
            continue

        # Create a condition for this metric
        condition = df[mean_column] < threshold
        conditions.append(condition)

    # Combine conditions with the "and" operator (all conditions must be True)
    combined_condition = pd.DataFrame(conditions).all()

    # Filter the DataFrame based on the combined condition
    subframe = df[combined_condition]

    return subframe
