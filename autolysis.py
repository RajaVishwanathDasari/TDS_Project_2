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
    """Perform basic analysis including summary statistics and outlier detection."""
    summary = {
        'columns': list(dataframe.columns),
        'data_types': dict(dataframe.dtypes),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'summary_statistics': dataframe.describe().to_dict()
    }

    # Dynamically choose further analysis based on data types
    numeric_columns = dataframe.select_dtypes(include=[np.number])
    if not numeric_columns.empty:
        correlation_matrix = numeric_columns.corr().to_dict()
    else:
        correlation_matrix = None

    return summary, correlation_matrix


def detect_outliers_z_score(dataframe, threshold=3):
    """Detect outliers using Z-score method."""
    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("No numeric columns for outlier detection.")
        return None

    z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
    outliers = (z_scores > threshold).sum(axis=0)
    dataframe['outliers'] = (z_scores > threshold).any(axis=1).astype(int)

    return dataframe, outliers


def perform_time_series_analysis(dataframe):
    """Performs time series analysis if time-related data is found."""
    date_column = None
    for col in dataframe.columns:
        if 'date' in col.lower() or 'time' in col.lower():  # Search for columns with "date" or "time"
            date_column = col
            break

    if date_column is None:
        print("No date column found for time series analysis.")
        return None

    dataframe[date_column] = pd.to_datetime(dataframe[date_column])
    dataframe.set_index(date_column, inplace=True)

    if 'target_column' in dataframe.columns:
        ts_data = dataframe['target_column']
        return ts_data.mean()  # Basic time series forecast: Mean of the target_column

    return None


def dynamic_cluster_analysis(dataframe, max_clusters=5):
    """Dynamically perform clustering analysis based on number of features and data distribution."""
    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("No numeric columns for clustering.")
        return None

    # Dynamically adjust clustering based on the number of columns
    n_features = numeric_data.shape[1]
    if n_features <= 2:
        n_clusters = 2  # Simple clustering if there are only 2 features
    elif n_features <= 5:
        n_clusters = 3  # Moderate complexity
    else:
        n_clusters = max_clusters  # Use the maximum clusters defined

    # Min-Max scaling
    min_vals = numeric_data.min()
    max_vals = numeric_data.max()
    scaled_data = (numeric_data - min_vals) / (max_vals - min_vals)

    # Basic K-means clustering manually (without sklearn)
    centroids = scaled_data.sample(n_clusters, random_state=42).values
    def assign_clusters(data, centroids):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    for _ in range(100):
        clusters = assign_clusters(scaled_data.values, centroids)
        new_centroids = []
        for i in range(n_clusters):
            cluster_data = scaled_data.values[clusters == i]
            if cluster_data.size > 0:
                new_centroids.append(cluster_data.mean(axis=0))
            else:
                new_centroids.append(centroids[i])

        new_centroids = np.array(new_centroids)

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids

    dataframe['cluster'] = clusters
    return dataframe[['cluster']]


def create_histograms(dataframe, bins=10):
    """Create histograms for numeric columns in the dataframe."""
    histograms = []
    numeric_columns = dataframe.select_dtypes(include=[np.number])

    for col in numeric_columns.columns:
        plt.figure()
        sns.histplot(dataframe[col], bins=bins)
        plt.title(f"Histogram of {col}")
        histograms.append(plt.gcf())
        plt.close()

    return histograms


def create_boxplots(dataframe):
    """Create boxplots for numeric columns in the dataframe."""
    boxplots = []
    numeric_columns = dataframe.select_dtypes(include=[np.number])

    for col in numeric_columns.columns:
        plt.figure()
        sns.boxplot(data=dataframe, x=col)
        plt.title(f"Boxplot of {col}")
        boxplots.append(plt.gcf())
        plt.close()

    return boxplots


def create_correlation_heatmap(dataframe):
    """Create a correlation heatmap for numeric columns."""
    numeric_data = dataframe.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    plt.figure()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.close()
    return plt.gcf()


def generate_story(data_summary, analysis_results, charts, advanced_analysis_results):
    """Generate a detailed story using the AI API based on the dataset analysis."""
    prompt = f"""
    Write a comprehensive analysis report based on the following information:
    1. Dataset summary: columns, data types, missing values, and summary statistics.
    2. Analytical insights: correlation matrix, outlier details, and any patterns or anomalies.
    3. Advanced analysis results:
        - Outlier and Anomaly Detection: {advanced_analysis_results['outlier_and_anomaly_detection']}
        - Time Series Forecast: {advanced_analysis_results['time_series_forecast']}
        - Cluster Analysis: {advanced_analysis_results['cluster_analysis']}
    4. Visualizations: include descriptions of the provided charts.

    Data Summary: {data_summary}
    Analysis Results: {analysis_results}
    Advanced Analysis Results: {advanced_analysis_results}
    Charts: {charts}
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
    """Generate a README file with detailed analysis and charts."""
    with open('README.md', 'w') as f:
        f.write("# Dataset Analysis Report\n\n")
        f.write(f"## Summary of Dataset\n\n")
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
    """Main function to perform the analysis on the provided CSV file."""
    dataframe = pd.read_csv(input_file, encoding='latin1')

    # Get columns and send to LLM for analysis suggestion
    columns = list(dataframe.columns)
    prompt = f"Here are the columns in the dataset: {columns}. Please suggest which analyses should be performed based on the available data."

    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "gpt-4",
        "messages": [
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500
    }

    response = requests.post(openai_api_url, headers=headers, json=data)
    if response.status_code != 200:
        print(f"Error: {response.status_code}\n{response.text}")
        sys.exit(1)

    # Parse the LLM response to get the analysis suggestions
    analysis_suggestions = response.json().get('choices', [{}])[0].get('message', {}).get('content', '')
    print(f"LLM suggested analyses: {analysis_suggestions}")

    # Perform suggested analysis
    summary, correlation_matrix = perform_generic_analysis(dataframe)
    advanced_analysis_results = {
        'outlier_and_anomaly_detection': perform_outlier_detection(dataframe),
        'time_series_forecast': perform_time_series_analysis(dataframe),
        'cluster_analysis': dynamic_cluster_analysis(dataframe),
    }

    # Generate AI story with basic and advanced analysis results
    ai_story = generate_story(
        summary,
        {'correlation_matrix': correlation_matrix},
        [],  # Empty chart list initially, populated later
        advanced_analysis_results  # Include advanced analysis results
    )

    # Create charts
    charts = create_histograms(dataframe)
    charts += create_boxplots(dataframe)
    charts.append(create_correlation_heatmap(dataframe))

    # Generate README with detailed analysis and charts
    create_readme(ai_story, charts, summary)

    print("Analysis complete. Check README.md and chart files.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    if not Path(input_file).is_file():
        print(f"File {input_file} not found.")
        sys.exit(1)

    analyze_csv(input_file)



