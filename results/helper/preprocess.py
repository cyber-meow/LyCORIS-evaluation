import pandas as pd
from functools import partial


SCORE_METRICS = ['Text Similarity', 'Image Similarity', 'Vendi', 'vendi']
DISTANCE_METRICS = ['Squared Centroid Distance', 'Style Loss']


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
        for arch_name in encoder_mapping:
            arch = arch.replace(arch_name, encoder_mapping[arch_name])
        return (metric, arch, '')
    elif len(components) == 3:  # Metric + Architecture + Condition
        arch = components[1].split(')')[0]
        for arch_name in encoder_mapping:
            arch = arch.replace(arch_name, encoder_mapping[arch_name])
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
    exp_parts = folder_name.split('exp')
    folder_name = 'exp' + exp_parts[1]
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
    if components[0].isdigit():
        if target == 'Weight':
            return float('0.' + components[0])
        else:
            del components[0]
    if target == 'Weight':
        return 1
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
            return ''
    else:
        raise ValueError('Unsupported target type')


def process_folder_path(df, keywords_to_exclude=None):
    targets = ['Config', 'Seed', 'Step', 'Weight',
               'Category', 'Class', 'Subclass']
    for idx, target in enumerate(targets):
        df.insert(idx+1, target,
                  df["Folder"].apply(get_folder_element, target=target))
        if target != 'Step':
            df[target] = df[target].astype('category')
    df.drop(level=0, columns='Folder', inplace=True)
    # Check if any of the keywords to exclude are in x

    def exclude_keywords(x):
        return not any(keyword in x for keyword in keywords_to_exclude)

    if keywords_to_exclude is not None:
        df = df[df['Config'].apply(exclude_keywords)]
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
        standard_ranking = (group.rank(ascending=True)-1) / (group_size-1)

        # Distance metric rank function, where lowest score gets highest rank
        distance_ranking = (group.rank(ascending=False)-1) / (group_size-1)

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
        metric_file, config_file,
        metrics_to_include=None, keywords_to_exclude=None,
        rank=True, score_metrics=None, distances_metrics=None):
    if score_metrics is None:
        score_metrics = SCORE_METRICS
    if distances_metrics is None:
        distance_metrics = DISTANCE_METRICS
    df_metrics = pd.read_csv(metric_file)
    df_metrics = df_metrics[df_metrics['Folder'].apply(lambda x: 'exp' in x)]
    existing_columns = list(df_metrics.columns)
    df_metrics = df_metrics[
        existing_columns[:1] + sorted(existing_columns[1:])]
    df_metrics = to_multiindex(df_metrics)
    if metrics_to_include is not None:
        df_metrics = filter_columns(df_metrics, metrics_to_include)
    df_metrics = process_folder_path(df_metrics, keywords_to_exclude)
    if rank:
        groupby_cols = ['Category', 'Class', 'Subclass']
        df_ranked = compute_scaled_ranks(
            df_metrics, groupby_cols, score_metrics, distance_metrics)
    else:
        df_ranked = df_metrics
    config_mapping = pd.read_csv(config_file)
    config_mapping = to_multiindex(config_mapping)
    category_configs = ['Config', 'Algo', 'Preset']
    for target in category_configs:
        config_mapping[target] = config_mapping[target].astype('category')
    return join_with_config(df_ranked, config_mapping)


def compute_additional_attributes(row, multiindex=False):
    algo = row['Algo']
    dim = row['Dim']
    alpha = row['Alpha']
    factor = row['Factor']
    lr = row['Lr']
    lrs = [1e-4, 5e-4, 1e-3]
    if multiindex:
        algo = algo.item()
        dim = dim.item()
        alpha = alpha.item()
        factor = factor.item()
        lr = lr.item()
    if algo == 'full':
        size = 4
        scale = 1
        lrs = [5e-7, 1e-6, 5e-6]
    elif algo == 'lokr':
        size = [12, 8, 4].index(factor) + 1
        scale = 1
        # lrs = [5e-4, 1e-3, 5e-3]
    elif algo == 'loha':
        if dim == 4:
            size = 2
        elif dim == 16:
            size = 3
        scale = alpha / dim
        # lrs = [5e-4, 1e-3, 5e-3]
    elif algo == 'lora':
        if dim == 8:
            size = 2
        elif dim == 32:
            size = 3
        scale = alpha / dim
        # lrs = [5e-5, 1e-4, 5e-4]
    lr_scale = lrs.index(lr) + 1
    column_names = ['Capacity', 'Scale', 'Lr Level']
    result = pd.Series({
        column_names[0]: size,
        column_names[1]: scale,
        column_names[2]: lr_scale})
    return result


def transform_attributes(df, drop=True,
                         drop_scale=False, multiindex=False):
    compute_add = partial(
        compute_additional_attributes, multiindex=multiindex)
    new_columns = df.apply(compute_add, axis=1)
    if multiindex:
        existing_levels = df.columns.nlevels
        multi_index = [(col, *[''] * (existing_levels-1))
                       for col in new_columns.columns]
        multi_index = pd.MultiIndex.from_tuples(
            multi_index, names=[None] * existing_levels)
        new_columns.columns = multi_index
    X = pd.concat([df, new_columns], axis=1)
    if drop:
        if multiindex:
            X = X.drop(columns=['Lr', 'Dim', 'Alpha', 'Factor'], level=0)
        else:
            X = X.drop(columns=['Lr', 'Dim', 'Alpha', 'Factor'])
    if drop_scale:
        if multiindex:
            X = X.drop(columns=['Scale'], level=0)
        else:
            X = X.drop(columns=['Scale'])
    return X
