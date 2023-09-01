import pandas as pd


SCORE_METRICS = ['Text Similarity', 'Image Similarity', 'Vendi', 'vendi']
DISTANCE_METRICS = ['Squared Centroid Distance', 'Style Loss']


def aggregate_metrics_aux(df, group_fields,
                          metric_fields=None, compute_std=True):
    """
    Aggregate metrics based on group_fields.
    Compute mean and optionally stderr for the metric_fields.

    Parameters:
    - df: Input dataframe
    - group_fields: List of fields to group by
    - metric_fields: List of metric fields to compute the mean and stderr
    - compute_std: Whether to compute stderr

    Returns:
    - aggregated_df: A dataframe with means and optionally stderrs
    """

    if metric_fields is None:
        metric_fields = SCORE_METRICS + DISTANCE_METRICS

    # Function to compute mean and optionally stderr for each group
    def aggregate_group(group):
        result = {}
        for column in group.columns:
            metric = column[0]
            if metric not in metric_fields:
                continue
            # Calculate the mean
            mean_value = group[column].mean()

            # Use multiindex to store these calculated values
            result[column + ('mean',)] = mean_value

            # Optionally calculate stderr
            if compute_std:
                std_value = group[column].std()
                stderr_value = std_value / (len(group[column]) ** 0.5)
                result[column + ('stderr',)] = stderr_value

        return pd.Series(result,
                         index=pd.MultiIndex.from_tuples(result.keys()))

    # Group by given fields and then apply the aggregate_group function
    aggregated_df = df.groupby(group_fields, dropna=False)
    aggregated_df = aggregated_df.apply(aggregate_group).reset_index()

    # Create a new MultiIndex with an additional 'Statistics' level
    new_columns = pd.MultiIndex.from_tuples(
        aggregated_df.columns,
        names=[*df.columns.names, 'Statistics']
    )
    aggregated_df.columns = new_columns

    aggregated_df.reset_index(drop=True, inplace=True)

    if not compute_std:
        aggregated_df.columns = aggregated_df.columns.droplevel(
            level='Statistics')

    return aggregated_df


def aggregate_subclass(df, metric_fields=None):
    df_with_subclass = df[df['Subclass'] != '']
    if len(df_with_subclass) == 0:
        return df
    group_fields = [
        'Config', 'Algo', 'Preset', 'Lr', 'Dim', 'Alpha', 'Factor', 'Seed',
        'Step', 'Category', 'Class'
    ]
    aggregated_df = aggregate_metrics_aux(df_with_subclass,
                                          group_fields,
                                          metric_fields,
                                          compute_std=False)
    df_without_subclass = df[df['Subclass'] == ''].drop(
        columns='Subclass', level=0)
    df_agg_subclass = pd.concat([aggregated_df, df_without_subclass])
    df_agg_subclass.sort_values(by='Config').reset_index(drop=True,
                                                         inplace=True)
    return df_agg_subclass


def aggregate_metrics(df,
                      level,
                      group_seeds=False,
                      input_agg_subclass=False,
                      metric_fields=None):
    assert level in ['All', 'Category', 'Class']
    if input_agg_subclass:
        df_agg_subclass = df
    else:
        df_agg_subclass = aggregate_subclass(df)
    if level == 'Class' and not group_seeds:
        return df_agg_subclass
    group_fields = [
        'Config', 'Algo', 'Preset', 'Lr', 'Dim', 'Alpha', 'Factor', 'Seed',
        'Step', 'Category', 'Class'
    ]
    if group_seeds:
        group_fields.remove('Seed')
    if level in ['Category', 'All']:
        group_fields.remove('Class')
    if level == 'All':
        group_fields.remove('Category')
    aggregated_df = aggregate_metrics_aux(df_agg_subclass, group_fields,
                                          metric_fields, compute_std=True)
    return aggregated_df


def filter_config_and_aggregate(data_frame,
                                columns_of_interest,
                                extra_condition=None,
                                algos=None,
                                default_keywords=None):
    # Filter by keywords in 'Config'
    if extra_condition is not None:
        for key in extra_condition:
            data_frame = data_frame[data_frame[key] == extra_condition[key]]
    if algos is None:
        algos = ['lora', 'loha', 'lokr']
    if default_keywords is None:
        default_keywords = ["1001", "1011", "1021", "1031"]
    keyword_condition = data_frame['Config'].apply(
        lambda x: any(keyword in x for keyword in default_keywords))
    df_filtered = data_frame[keyword_condition]

    if len(columns_of_interest) != 0:
        col_diff_filtered = []

        for algo in algos:
            algo_row = df_filtered[df_filtered['Algo'] == algo].iloc[0]
            algo_subframe = data_frame[data_frame['Algo'] == algo]
            for col in columns_of_interest:
                default = algo_row[col].item()
                algo_subframe_filtered = algo_subframe[
                    pd.notna(algo_subframe[col])]
                col_diff_filtered.append(
                    algo_subframe_filtered[
                        algo_subframe_filtered[col] != default])

        df_to_concat = pd.concat(col_diff_filtered).drop_duplicates(
        ).reset_index(drop=True)

        # Concatenate df_filtered and df_to_concat
        result = pd.concat([df_filtered, df_to_concat], ignore_index=True)
    else:
        result = df_filtered
    result = aggregate_metrics(result, level='All', group_seeds=True)

    return result
