# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "requests",
#   "pathlib",
#   "matplotlib"
# ]
# ///
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import json
import requests
import sys

# Set your custom proxy URL for the OpenAI API
openai_api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Load your AIPROXY_TOKEN environment variable
openai_api_key = os.environ.get("AIPROXY_TOKEN")

def perform_generic_analysis(dataframe):
    """
    Perform basic analysis on the given DataFrame, including summary statistics and correlation analysis.

    Parameters:
        dataframe (pd.DataFrame): The input dataset for analysis.

    Returns:
        tuple: A tuple containing:
            - summary (dict): Dataset summary with column details, data types, missing values, and summary statistics.
            - correlation_matrix (dict or None): Correlation matrix for numeric columns, or None if no numeric columns exist.
    """
    summary = {
        'columns': list(dataframe.columns),
        'data_types': dict(dataframe.dtypes),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'summary_statistics': dataframe.describe().to_dict()
    }

    numeric_columns = dataframe.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr().to_dict() if not numeric_columns.empty else None

    return summary, correlation_matrix

def detect_outliers_z_score(dataframe, threshold=3):
    """
    Detect outliers in the dataset using the Z-score method.

    Parameters:
        dataframe (pd.DataFrame): The input dataset for analysis.
        threshold (float): The Z-score threshold for outlier detection. Defaults to 3.

    Returns:
        tuple: A tuple containing:
            - dataframe (pd.DataFrame): The original dataframe with an additional 'outliers' column.
            - outliers (pd.Series): Count of outliers in each numeric column.
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("No numeric columns for outlier detection.")
        return None, None

    z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
    dataframe['outliers'] = (z_scores > threshold).any(axis=1).astype(int)
    outliers = (z_scores > threshold).sum(axis=0)

    return dataframe, outliers

def perform_time_series_analysis(dataframe):
    """
    Perform time series analysis if a time-related column is detected in the dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset for analysis.

    Returns:
        pd.Series or None: A time series object based on a detected 'date' column or None if no time column exists.
    """
    date_column = next((col for col in dataframe.columns if 'date' in col.lower() or 'time' in col.lower()), None)

    if date_column is None:
        print("No date column found for time series analysis.")
        return None

    dataframe[date_column] = pd.to_datetime(dataframe[date_column])
    dataframe.set_index(date_column, inplace=True)

    # Placeholder for additional time-series specific processing
    return dataframe

def dynamic_cluster_analysis(dataframe, max_clusters=5):
    """
    Perform clustering analysis dynamically based on the dataset's numeric columns.

    Parameters:
        dataframe (pd.DataFrame): The input dataset for clustering.
        max_clusters (int): Maximum number of clusters. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with an additional 'cluster' column indicating cluster membership.
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("No numeric columns for clustering.")
        return None

    # Dynamically adjust clustering parameters
    n_features = numeric_data.shape[1]
    n_clusters = min(max_clusters, max(2, n_features))

    # Simple K-means implementation
    centroids = numeric_data.sample(n_clusters, random_state=42).values

    def assign_clusters(data, centroids):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    for _ in range(100):
        clusters = assign_clusters(numeric_data.values, centroids)
        new_centroids = np.array([
            numeric_data.values[clusters == i].mean(axis=0) if (clusters == i).any() else centroids[i]
            for i in range(n_clusters)
        ])

        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids

    dataframe['cluster'] = clusters
    return dataframe

def create_histograms(dataframe, bins=10):
    """
    Create histograms for all numeric columns in the dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset for visualization.
        bins (int): Number of bins for the histogram. Defaults to 10.

    Returns:
        list: A list of matplotlib Figure objects for the histograms.
    """
    histograms = []
    numeric_columns = dataframe.select_dtypes(include=[np.number])

    for col in numeric_columns.columns:
        plt.figure()
        sns.histplot(dataframe[col], bins=bins, kde=True, color='skyblue')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)
        plt.tight_layout()
        histograms.append(plt.gcf())
        plt.close()

    return histograms

def create_boxplots(dataframe):
    """
    Create boxplots for all numeric columns in the dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset for visualization.

    Returns:
        list: A list of matplotlib Figure objects for the boxplots.
    """
    boxplots = []
    numeric_columns = dataframe.select_dtypes(include=[np.number])

    for col in numeric_columns.columns:
        plt.figure()
        sns.boxplot(x=dataframe[col], color='orange')
        plt.title(f"Boxplot of {col}")
        plt.xlabel(col)
        plt.grid(True)
        plt.tight_layout()
        boxplots.append(plt.gcf())
        plt.close()

    return boxplots

def create_correlation_heatmap(dataframe):
    """
    Create a heatmap for visualizing the correlation matrix of numeric columns in the dataset.

    Parameters:
        dataframe (pd.DataFrame): The input dataset for visualization.

    Returns:
        matplotlib.figure.Figure: A matplotlib Figure object for the heatmap.
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return plt.gcf()

def generate_story(data_summary, analysis_results, charts, advanced_analysis_results):
    """
    Generate a textual summary of the dataset analysis using AI API.

    Parameters:
        data_summary (dict): Summary of dataset including columns, data types, etc.
        analysis_results (dict): Analytical results such as correlation matrix and outliers.
        charts (list): List of charts generated during the analysis.
        advanced_analysis_results (dict): Advanced analysis like clustering and time series insights.

    Returns:
        str: The AI-generated story or a failure message.
    """
    prompt = f"""
    Write a comprehensive analysis report based on the following information:
    Dataset Summary: {data_summary}
    Analysis Results: {analysis_results}
    Advanced Analysis Results: {advanced_analysis_results}
    """

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 700
    }

    response = requests.post(openai_api_url, headers=headers, json=data)
    if response.status_code == 200:
        return response.json().get('choices', [{}])[0].get('message', {}).get('content', "AI generation failed.")
    else:
        print(f"Error: {response.status_code}\n{response.text}")
        return "AI generation failed."

def create_readme(ai_story, charts, summary):
    """
    Generate a README file summarizing the dataset analysis and include visualizations.

    Parameters:
        ai_story (str): The AI-generated story summarizing the analysis.
        charts (list): List of matplotlib Figure objects to include in the README.
        summary (dict): Summary of the dataset.
    """
    with open('README.md', 'w') as f:
        f.write("# Dataset Analysis Report\n\n")
        f.write("## Summary of Dataset\n\n")
        f.write(f"Columns: {', '.join(summary['columns'])}\n")
        f.write(f"Data Types: {json.dumps(summary['data_types'], indent=2)}\n")
        f.write(f"Missing Values: {json.dumps(summary['missing_values'], indent=2)}\n")
        f.write(f"Summary Statistics: {json.dumps(summary['summary_statistics'], indent=2)}\n")
        f.write("\n## Analysis Results\n")
        f.write(ai_story)
        f.write("\n## Charts\n")

        for chart in charts:
            chart_file = f"chart_{charts.index(chart)}.png"
            chart.savefig(chart_file)
            f.write(f"![Chart {charts.index(chart)}]({chart_file})\n")

def analyze_csv(input_file):
    """
    Perform a comprehensive analysis of a CSV file including basic and advanced analysis.

    Parameters:
        input_file (str): Path to the input CSV file.
    """
    dataframe = pd.read_csv(input_file, encoding='latin1')

    summary, correlation_matrix = perform_generic_analysis(dataframe)
    advanced_analysis_results = {
        'outlier_and_anomaly_detection': detect_outliers_z_score(dataframe)[1].to_dict(),
        'time_series_forecast': perform_time_series_analysis(dataframe),
        'cluster_analysis': dynamic_cluster_analysis(dataframe)['cluster'].value_counts().to_dict()
    }

    ai_story = generate_story(
        summary,
        {'correlation_matrix': correlation_matrix},
        [],
        advanced_analysis_results
    )

    charts = create_histograms(dataframe)
    charts += create_boxplots(dataframe)
    charts.append(create_correlation_heatmap(dataframe))

    create_readme(ai_story, charts, summary)
    print("Analysis complete. Check README.md and chart files.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <dataset.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not Path(input_file).is_file():
        print(f"File {input_file} not found.")
        sys.exit(1)

    analyze_csv(input_file)



