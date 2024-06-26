import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

    
def plot_histograms(df, columns, n_cols, n_rows=None, kde=False, bins='auto'):
    """
    Create histogram subplots for specified columns in a DataFrame.

    Args:
    - df: pandas DataFrame
        The DataFrame containing the fields to plot.
    - columns: list
        List of column names in df for which histograms are to be plotted.
    - n_cols: int
        Number of columns for subplot layout.
    - n_rows: int, optional
        Number of rows for subplot layout. Default is calculated based on len(columns).
    - kde: bool, optional
        Whether a KDE plots should be included. Default is False
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
    
    # Calculate number of rows required
    if n_rows is None:
        n_rows = (len(columns) + n_cols - 1) // n_cols if n_cols else 1 

    # Initialize figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Ensure axes always iterable
    axes = axes.flatten() if n_rows > 1 else np.array(axes).flatten()  
    
    for i, col in enumerate(columns):
        sns.histplot(df[col], kde=kde, ax=axes[i], bins=bins)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Remove any unused subplots
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig, axes



def plot_boxplots(df, columns, n_cols, n_rows=None):
    """
    Create boxplot subplots for specified columns in a DataFrame.

    Args:
    - df: pandas DataFrame
        The DataFrame containing the fields to plot.
    - columns: list
        List of column names in df for which boxplots are to be plotted.
    - n_cols: int
        Number of columns for subplot layout.
    - n_rows: int, optional
        Number of rows for subplot layout. Default is calculated based on len(columns).

    Returns:
    - fig, axes: matplotlib Figure and Axes objects
        A figure with boxplot subplots for each provided column
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
    
    # Calculate number of rows required
    if n_rows is None:
        n_rows = (len(columns) + n_cols - 1) // n_cols if n_cols else 1 

    # Initialize figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Ensure axes always iterable
    axes = axes.flatten() if n_rows > 1 else np.array(axes).flatten()   

    for i, col in enumerate(columns):
        sns.boxplot(y=df[col], ax=axes[i])
        axes[i].set_title(f'Boxplot of {col}')
        axes[i].set_ylabel('Value')

    # Remove any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig, axes


# Function to filter out values outside the IQR
def filter_iqr(df, column):
    """
    Filter out records outside the IQR of a given column

    Args:
    - df: pandas DataFrame
        The DataFrame to filter.
    - columns: string
        Column name in df to drive filtering.
    Returns:
    - df_filtered: pandas DataFrame
        Dataframe with records removed which were outside IQR range for given column
    """
    # Check if df is a DataFrame
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")
    
    # Check if column exists in df
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    df_filtered = df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    return df_filtered


def plot_IQRhistograms(df, columns, n_cols, n_rows=None, kde=False, bins='auto'):
    """
    Create histogram subplots for specified columns in a DataFrame, excluding records which fall outside its interquartile range.

    Args:
    - df: pandas DataFrame
        The DataFrame containing the fields to plot.
    - columns: list
        List of column names in df for which histograms are to be plotted.
    - n_cols: int
        Number of columns for subplot layout.
    - n_rows: int, optional
        Number of rows for subplot layout. Default is calculated based on len(columns).
    - kde: bool, optional
        Whether a KDE plots should be included. Default is False
    - bins: int, optional
        Number of bins for the histograms. Default is None (automatic binning).

    Returns:
    - fig, axes: matplotlib Figure and Axes objects
        A figure with histogram subplots for each provided column, excluding records outside IQR
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
    
    # Calculate number of rows required
    if n_rows is None:
        n_rows = (len(columns) + n_cols - 1) // n_cols if n_cols else 1 

    # Initialize figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Ensure axes always iterable
    axes = axes.flatten() if n_rows > 1 else np.array(axes).flatten()  
    
    for i, col in enumerate(columns):
        df_filtered = filter_iqr(df, col)
        sns.histplot(df_filtered[col], kde=kde, ax=axes[i], bins=bins)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')

    # Remove any unused subplots
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to avoid overlap
    plt.tight_layout()

    return fig, axes


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_freq_dist(df, columns, n_cols, n_rows=None):
    """
    Create frequency distribution subplots for specified categorical columns in a DataFrame.

    Args:
    - df: pandas DataFrame
        The DataFrame containing the fields to plot.
    - columns: list
        List of categorical column names in df for which frequency distributions are to be plotted.
    - n_cols: int
        Number of columns for subplot layout.
    - n_rows: int, optional
        Number of rows for subplot layout. Default is calculated based on len(columns).

    Returns:
    - fig, axes: matplotlib Figure and Axes objects
        A figure with frequency distribution subplots for each provided column.
    """
    # Validate input DataFrame and columns
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input 'df' must be a pandas DataFrame.")

    if not isinstance(columns, list):
        raise TypeError("Input 'columns' must be a list of column names.")

    # Validate existence of columns in DataFrame
    invalid_columns = [col for col in columns if col not in df.columns]
    if invalid_columns:
        raise ValueError(f"The following columns do not exist in the DataFrame: {', '.join(invalid_columns)}")

    # Calculate number of rows if not provided
    if n_rows is None:
        n_rows = (len(columns) + n_cols - 1) // n_cols
   
    # Initialize figure and axes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))

    # Ensure axes always iterable
    axes = axes.flatten() if n_rows > 1 else np.array(axes).flatten()  

    # Plotting each column's frequency distribution
    for i, col in enumerate(columns):
        sns.countplot(data=df, x=col, ax=axes[i])
        axes[i].set_title(f'Frequency Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].tick_params(axis='x', rotation=45)  # Rotate labels for readability

    # Remove unused subplots
    for j in range(len(columns), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout to prevent overlap
    plt.tight_layout()

    return fig, axes
