import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_histograms(df, columns, n_cols, n_rows=None, bins=None):
    """
    Create histogram subplots for specified columns in a DataFrame.

    Args:
    - df: pandas DataFrame
        The DataFrame containing the fields to plot.
    - columns: list
        List of column names in df for which histograms are to be plotted.
    - n_rows: int, optional
        Number of rows for subplot layout. Default is calculated based on len(columns).
    - n_cols: int, optional
        Number of columns for subplot layout. Default is 2.
    - bins: int, optional
        Number of bins for the histograms. Default is None (automatic binning).

    Returns:
    - fig, axes: matplotlib Figure and Axes objects
        A figure with histogram subplots for each provided column
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    # Check if columns is a list
    if not isinstance(columns, list):
        raise TypeError("Input 'columns' must be a list of column names.")

    # Check if columns exist in df
    invalid_columns = [col for col in columns if col not in df.columns]
    if invalid_columns:
        raise ValueError(f"The following columns do not exist in the DataFrame: {', '.join(invalid_columns)}")

    if n_rows is None:
        n_rows = (len(columns) + n_cols - 1) // n_cols if n_cols else 1 # Calculate number of rows required

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    axes = axes.flatten() if n_rows > 1 else np.array([axes])  # Ensure axes is always iterable

    for i, col in enumerate(columns):
        ax = axes[i]
        sns.histplot(df[col], kde=True, ax=ax, bins=bins)
        ax.set_title(f'Distribution of {col}')
        ax.set_xlabel(col)
        ax.set_ylabel('Frequency')

    # Remove any unused subplots
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig, axes

# Example usage:
# Assuming `df` is your DataFrame and `filteredcols` or `fundingcols` are lists of columns
# Replace with actual data and column names as per your use case

# Example 1: Automatic layout and bins
# fig, axes = plot_histograms(df, filteredcols)

# Example 2: Specifying layout and bins
# fig, axes = plot_histograms(df, fundingcols, n_cols=3, bins=10)

# Show the plots
# plt.show()
