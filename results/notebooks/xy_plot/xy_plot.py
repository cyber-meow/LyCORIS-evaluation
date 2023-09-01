import matplotlib.pyplot as plt
import seaborn as sns


def plot_metrics_xy(data_frame,
                    metric_1,
                    metric_2,
                    color_by,
                    shape_by,
                    shape_mapping=None,
                    legend=True,
                    step=None,
                    save_name=None):

    # Ensure 'steps' is a list

    # Unique values for color_by and shape_by columns
    if color_by == 'Algo':
        unique_colors = ['lora', 'loha', 'lokr', 'full']
    else:
        unique_colors = sorted(data_frame[color_by].unique())
    if shape_by == 'Preset':
        unique_shapes = ['attn-only', 'attn-mlp', 'full']
    else:
        unique_shapes = sorted(data_frame[shape_by].unique())

    # Create a dictionary for mapping unique colors and shapes to
    # plotting markers and colors
    color_palettes = sns.color_palette("husl", len(unique_colors))
    color_map = {key: color_palettes[i] for i, key in enumerate(unique_colors)}
    shape_map = {key: marker for marker, key in zip('os^v<>', unique_shapes)}
    linestyle_map = {
        key: linestyle
        for key, linestyle in zip(unique_shapes, ['-', '--', '-.', ':'])
    }

    if step is not None:
        data_frame = data_frame[data_frame['Step'] == step]

    for (color_val,
         shape_val) in set(zip(data_frame[color_by], data_frame[shape_by])):
        color = color_map[color_val]
        marker = shape_map[shape_val]
        linestyle = linestyle_map[shape_val]
        filter_conditions = (data_frame[color_by] == color_val) & (
            data_frame[shape_by] == shape_val)

        x = data_frame.loc[filter_conditions, metric_1 + ('mean', )]
        y = data_frame.loc[filter_conditions, metric_2 + ('mean', )]
        x_err = data_frame.loc[filter_conditions, metric_1 + ('stderr', )]
        y_err = data_frame.loc[filter_conditions, metric_2 + ('stderr', )]

        eb = plt.errorbar(x,
                          y,
                          xerr=x_err,
                          yerr=y_err,
                          fmt=marker,
                          color=color,
                          capsize=4,
                          ms=8,
                          mew=1)
        eb[-1][0].set_linestyle(linestyle)
        eb[-1][1].set_linestyle(linestyle)

    handles = []

    # Plot legend for color
    for color_val, color in color_map.items():
        handle, = plt.plot([], [], 'o', color=color, label=color_val)
        handles.append(handle)

    # Plot legend for shape
    for shape_val, marker in shape_map.items():
        if shape_mapping is not None:
            shape_val = shape_mapping[shape_val]
        handle, = plt.plot([], [], marker, color='gray', label=shape_val)
        handles.append(handle)

    if legend:
        plt.legend(title=f'{color_by} / {shape_by}', loc="best")

    if metric_2[0] == 'Squared Centroid Distance':
        metric_2 = ('Centroid Distance', '', metric_2[2])
    plt.xlabel(f'{metric_1[0]}-{metric_1[2]}')
    plt.ylabel(f'{metric_2[0]}-{metric_2[2]}')
    plt.grid(True)
    if save_name is not None:
        plt.savefig(save_name)
    return handles
