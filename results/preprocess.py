import pandas as pd


def extract_multiindex_components(col_name):
    components = col_name.split('(')

    # For the metric, it's always the first component
    metric = components[0].strip()
    if metric == 'Text Similarity':
        components.insert(1, 'CLIP)')
    elif metric == 'Style Loss':
        components.insert(1, 'Vgg19)')
    encoder_mapping = {
        'clip-L-14': 'CLIP',
        'dinov2-l-fb': 'DINOv2',
        'convnextv2-l': 'ConvNeXt V2',
    }

    # Check how many parenthesis to determine the case
    if len(components) == 2:  # Metric + Architecture
        arch = components[1].split(')')[0]
        if arch in encoder_mapping:
            arch = encoder_mapping[arch]
        return (metric, arch, '')
    elif len(components) == 3:  # Metric + Architecture + Condition
        arch = components[1].split(')')[0]
        if arch in encoder_mapping:
            arch = encoder_mapping[arch]
        condition = components[2].split(')')[0]
        return (metric, arch, condition)
    else:  # Only metric
        return (metric, '', '')


def to_multiindex(df):
    # Extracting components for MultiIndex
    components = [
        extract_multiindex_components(col) for col in df.columns
    ]
    # Create MultiIndex
    multiindex = pd.MultiIndex.from_tuples(
        components, names=['', 'Architecture', 'Condition'])
    # Assign MultiIndex to dataframe columns
    df.columns = multiindex
    return df


def filter_columns(df, triplets):
    """
    Filter dataframe columns based on a list of triplets
        (metric, architecture, condition).

    Parameters:
    - df: DataFrame with a MultiIndex on columns.
    - triplets: List of tuples, where each tuple is
        (metric, architecture, condition).

    Returns:
    - Filtered DataFrame
    """

    # Extract the columns that match the given triplets
    columns_to_keep = [('Folder', '', '')
                       ] + [col for col in df.columns if col in triplets]

    # Return the filtered dataframe
    return df[columns_to_keep]


def extract_config_step(folder_name):
    parts = folder_name.split('-')
    if parts[0] == 'v15':
        del parts[0]
    if parts[-1] in ['a', 'b', 'c']:
        config = '-'.join(parts[:-1])
        seed = parts[-1]
        step = 50
    else:
        config = '-'.join(parts[:-2])
        seed = parts[-2]
        step = int(parts[-1])
    return config, seed, step


def get_folder_element(folder_str, target):
    components = folder_str.split('/')
    if target == 'Config':
        return extract_config_step(components[0])[0]
    elif target == 'Seed':
        return extract_config_step(components[0])[1]
    elif target == 'Step':
        return extract_config_step(components[0])[2]
    elif target == 'Category':
        return components[1]
    elif target == 'Class':
        return components[2]
    elif target == 'Subclass':
        try:
            return components[3]
        except IndexError:
            return 'none'
    else:
        raise ValueError('Unsupported target type')


def process_folder_path(df):
    targets = ['Config', 'Seed', 'Step', 'Category', 'Class', 'Subclass']
    for idx, target in enumerate(targets):
        df.insert(idx+1, target,
                  df["Folder"].apply(get_folder_element, target=target))
        if target != 'Step':
            df[target] = df[target].astype('category')
    df.drop(level=0, columns='Folder', inplace=True)
    return df


def compute_scaled_ranks(df, groupby_columns, score_metrics, distance_metrics):
    """
    Compute scaled ranks within groups defined by groupby_columns.

    Parameters:
    - df: DataFrame containing the data.
    - groupby_columns: List of column names used to group the data.
    - score_metrics: List of metrics (at the 'Metric' level in the multiindex)
        where a higher score is better.
    - distance_metrics: List of metrics
        (at the 'Metric' level in the multiindex)
        where a lower score is better.

    Returns:
    - DataFrame with scores in specified columns replaced by scaled ranks.
    """

    def rank_within_group(group):
        """Computes the scaled rank for each column in the group."""
        group_size = len(group)

        # Standard rank function, where highest score gets highest rank
        standard_ranking = group.rank(ascending=False) / group_size

        # Distance metric rank function, where lowest score gets highest rank
        distance_ranking = group.rank(ascending=True) / group_size

        # Apply the appropriate ranking method based on the column's presence
        # in distance_metrics
        for column in group.columns:
            if column[0] in distance_metrics:
                group[column] = distance_ranking[column]
            elif column[0] in score_metrics:
                group[column] = standard_ranking[column]

        return group

    # Group by specified columns and apply the ranking function
    df_ranked = df.groupby(groupby_columns).apply(rank_within_group)

    # Drop groupby columns from index (since they are duplicated in the result)
    df_ranked.reset_index(drop=True, inplace=True)

    return df_ranked


def join_with_config(metrics, config_mapping):
    """
    Join config_mapping and metrics dataframes based on specified logic.
    """
    # Join dataframes on config_name and Folder_modified
    merged_df = pd.merge(metrics,
                         config_mapping,
                         on=[('Config', '', '')],
                         how='inner')
    all_columns = merged_df.columns.tolist()
    df1_columns = metrics.columns.tolist()
    df2_columns = [col for col in all_columns if col not in df1_columns]
    new_column_order = df1_columns[:1] + df2_columns + df1_columns[1:]
    merged_df = merged_df[new_column_order]

    return merged_df


def load_and_preprocess_metrics(
        metric_file, config_file, metrics_to_include=None,
        score_metrics=None, distances_metrics=None):
    if score_metrics is None:
        score_metrics = ['Text Similarity',
                         'Image Similarity', 'Vendi', 'vendi']
    if distances_metrics is None:
        distance_metrics = ['Squared Centroid Distance', 'Style Loss']
    df_metrics = pd.read_csv(metric_file)
    df_metrics = to_multiindex(df_metrics)
    if metrics_to_include is not None:
        df_metrics = filter_columns(df_metrics, metrics_to_include)
    df_metrics = process_folder_path(df_metrics)
    groupby_cols = ['Category', 'Class', 'Subclass']
    df_ranked = compute_scaled_ranks(
        df_metrics, groupby_cols, score_metrics, distance_metrics)
    config_mapping = pd.read_csv(config_file)
    config_mapping = to_multiindex(config_mapping)
    category_configs = ['Config', 'Algo', 'Preset']
    for target in category_configs:
        config_mapping[target] = config_mapping[target].astype('category')
    return join_with_config(df_ranked, config_mapping)
