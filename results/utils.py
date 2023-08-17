import os
import pandas as pd


def filter_dataframe(df, keywords, steps=None, categories=None):
    """
    Filters the dataframe based on folder name keywords, optional step value, and category.

    Parameters:
    - df (pd.DataFrame): Input dataframe.
    - keywords (list): List of keywords to search in folder names.
    - steps (int or list of ints, optional): One or more of [10, 30, 50]. Default is None.
    - categories (str or list of strs, optional): Specific categories. Default is None.

    Returns:
    - pd.DataFrame: Filtered dataframe.
    """

    if not keywords:
        raise ValueError("Provide at least one keyword to filter by.")

    # Filter by keywords
    keyword_condition = df['Folder'].apply(
        lambda x: any(keyword in x for keyword in keywords))
    df_filtered = df[keyword_condition]

    # If step is provided, further filter based on step value(s)
    if steps is not None:
        valid_steps = [10, 30, 50]

        # Ensure step values are in valid_steps
        if isinstance(steps, int):
            if steps not in valid_steps:
                raise ValueError(
                    "Invalid step value. It should be one of [10, 30, 50].")
            steps = [steps]  # Convert to list for consistent handling below

        # Handle invalid values in step list
        for s in steps:
            if s not in valid_steps:
                raise ValueError(
                    "Invalid step value in list. All should be one of [10, 30, 50]."
                )

        # Create filter conditions for each step value
        step_conditions = []
        for s in steps:
            if s == 10:
                step_conditions.append(
                    df_filtered['Folder'].apply(lambda x: os.path.basename(
                        x.split(os.path.sep)[0]).endswith("000010")))
            elif s == 30:
                step_conditions.append(
                    df_filtered['Folder'].apply(lambda x: os.path.basename(
                        x.split(os.path.sep)[0]).endswith("000030")))
            elif s == 50:
                step_conditions.append(df_filtered['Folder'].apply(
                    lambda x: not (os.path.basename(x.split(os.path.sep)[
                        0]).endswith("000010") or os.path.basename(
                            x.split(os.path.sep)[0]).endswith("000030"))))

        # Combine all step conditions with OR operation
        combined_step_condition = pd.concat(step_conditions,
                                            axis=1).any(axis=1)

        df_filtered = df_filtered[combined_step_condition]

    # If category is provided, filter based on it
    if categories:
        if isinstance(categories, str):
            categories = [categories]
        category_conditions = []
        for category in categories:
            if category == "all":
                # Only consider folders with a single level
                category_conditions.append(df_filtered['Folder'].apply(
                    lambda x: len(x.split(os.path.sep)) == 1))
            else:
                # Consider any subpath of folder names matching the category exactly
                category_conditions.append(
                    df_filtered['Folder'].apply(lambda x: any(
                        os.path.join(*x.split(os.path.sep)[i:]) == category
                        for i in range(
                            len(x.split(os.path.sep)) - len(
                                category.split(os.path.sep)) + 1))))

        # Combine all step conditions with OR operation
        combined_category_condition = pd.concat(category_conditions,
                                                axis=1).any(axis=1)
        df_filtered = df_filtered[combined_category_condition]

    return df_filtered


def extract_common_prefix(folder_name):
    """
    Extract common prefix by considering variations in folder naming conventions.
    """
    all_levels = folder_name.split(os.path.sep)
    parts = all_levels[0].split('-')
    if parts[-1] in ['a', 'b', 'c']:
        first_part = '-'.join(parts[:-1])
    else:
        first_part = '-'.join(parts[:-2]+[parts[-1]])
    return os.path.join(first_part, *all_levels[1:])


def group_and_aggregate(df):
    """
    Groups the dataframe by the common prefix in the 'Folder' column (ignoring the a/b/c distinctions).
    Computes mean and standard deviation for each group and each score column.

    Parameters:
    - df (pd.DataFrame): Input dataframe.

    Returns:
    - pd.DataFrame: Aggregated dataframe.
    """

    # Extract common prefix
    df = df.copy()
    df['CommonPrefix'] = df['Folder'].apply(extract_common_prefix)

    # Group by common prefix and compute aggregates
    mean_df = df.groupby('CommonPrefix').mean()
    std_df = df.groupby('CommonPrefix').std()
    count_df = df.groupby('CommonPrefix').size()

    # Calculate standard error
    stderr_df = std_df.div(count_df ** 0.5, axis=0)

    # Rename columns for clarity
    mean_df.columns = [f"{col}_mean" for col in mean_df.columns]
    std_df.columns = [f"{col}_std" for col in std_df.columns]
    stderr_df.columns = [f"{col}_stderr" for col in stderr_df.columns]

    # Combine mean and std dataframes
    agg_df = pd.concat(
        [mean_df, std_df, stderr_df], axis=1).reset_index()

    # Add the count column
    agg_df['count'] = count_df.values

    return agg_df


def extract_step(folder_name):
    """
    Extract step from folder name.
    If it ends with a, b, c, step is 50.
    Otherwise, extract the numerical part, which represents the step.
    """
    if folder_name[-1] in ['a', 'b', 'c']:
        return 50
    else:
        # Extract the last digits from the string
        return int(folder_name.split('-')[-1])


def extract_folder_modified(folder_name):
    """
    Extract the portion after 'exp' and before any step indicators.
    """
    # Find the start of 'exp'
    start_idx = folder_name.find('exp')
    if start_idx == -1:
        return None

    # Extract portion after 'exp'
    main_part = folder_name[start_idx:]

    # Remove step part
    if main_part[-1] in ['a', 'b', 'c']:
        return main_part
    else:
        return '-'.join(main_part.split('-')[:-1])


def join_dataframes(config_mapping, metrics):
    """
    Join config_mapping and metrics dataframes based on specified logic.
    """
    # Create a new column in metrics for step
    metrics = metrics.copy()
    metrics['step'] = metrics['Folder'].apply(extract_step)

    # Create a new column in metrics for matching without step and after "exp"
    metrics['Folder_modified'] = metrics['Folder'].apply(
        extract_folder_modified)

    # Join dataframes on config_name and Folder_modified
    result = pd.merge(metrics,
                      config_mapping,
                      left_on='Folder_modified',
                      right_on='config_name',
                      how='inner')

    # Drop the temporary Folder_modified column
    result.drop('Folder_modified', axis=1, inplace=True)
    result.drop('config_name', axis=1, inplace=True)

    return result


def aggregate_metrics(df, group_fields, metric_fields):
    """
    Aggregate metrics based on group_fields.
    Compute mean and stderr for the metric_fields.

    Parameters:
    - df: Input dataframe
    - group_fields: List of fields to group by
    - metric_fields: List of metric fields to compute the mean and stderr

    Returns:
    - aggregated_df: A dataframe with means and stderrs
    """
    # Group by fields and compute mean
    mean_df = df.groupby(group_fields, dropna=False)[
        metric_fields].mean().reset_index()

    # Group by fields and compute standard error
    stderr_df = df.groupby(group_fields, dropna=False)[
        metric_fields].std().reset_index()
    count_df = df.groupby(group_fields, dropna=False)[
        metric_fields].count().reset_index()

    for metric in metric_fields:
        stderr_df[metric] = stderr_df[metric] / count_df[metric].pow(0.5)

    # Merge mean and stderr dataframes
    for metric in metric_fields:
        mean_df[f"{metric}_mean"] = mean_df[metric]
        mean_df[f"{metric}_stderr"] = stderr_df[metric]
        mean_df.drop(metric, axis=1, inplace=True)

    return mean_df
