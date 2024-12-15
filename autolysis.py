# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "seaborn",
#   "requests",
#   "sys",
#   "pathlib",
#   "json",
#   "os"
# ]
# ///
import os
import pandas as pd
import numpy as np
import seaborn as sns
import requests
import sys
from pathlib import Path
import json

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

    numeric_columns = dataframe.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr().to_dict() if not numeric_columns.empty else None

    if not numeric_columns.empty:
        Q1 = numeric_columns.quantile(0.25)
        Q3 = numeric_columns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_columns < (Q1 - 1.5 * IQR)) | (numeric_columns > (Q3 + 1.5 * IQR))).sum().to_dict()
    else:
        outliers = None

    return summary, correlation_matrix, outliers

def perform_outlier_and_anomaly_detection(dataframe):
    """Perform outlier and anomaly detection using Z-score."""
    numeric_data = dataframe.select_dtypes(include=[np.number])
    
    # Calculate Z-scores for anomaly detection
    z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
    outliers = (z_scores > 3).sum(axis=0)  # Flag rows with Z-score > 3 as anomalies
    dataframe['outliers'] = (z_scores > 3).any(axis=1).astype(int)
    
    return dataframe, outliers

def perform_regression_analysis(dataframe):
    """Perform basic linear regression using numpy (X and Y must be numeric)."""
    X = dataframe.dropna().select_dtypes(include=[np.number]).drop(columns=['target_column'], errors='ignore')
    y = dataframe['target_column'] if 'target_column' in dataframe.columns else None
    
    regression_results = None
    if y is not None:
        X = np.c_[np.ones(len(X)), X]  # Add intercept (bias term)
        beta = np.linalg.inv(X.T @ X) @ X.T @ y  # Normal equation: (X^T * X)^(-1) * X^T * y
        regression_results = dict(zip(['Intercept'] + X.columns.tolist(), beta))  # Save coefficients
    
    return regression_results

def perform_time_series_analysis(dataframe, date_column='date'):
    """Perform basic time series analysis and forecasting using mean."""
    # Assume the 'date_column' is already in datetime format
    dataframe[date_column] = pd.to_datetime(dataframe[date_column])
    dataframe.set_index(date_column, inplace=True)
    
    # If there is a target_column, use it for time series forecasting (simple mean prediction)
    ts_data = dataframe['target_column'] if 'target_column' in dataframe.columns else None
    if ts_data is not None:
        forecast = ts_data.mean()  # Basic forecast: the mean of the data
        return forecast
    return None

def perform_cluster_analysis(dataframe):
    """Perform basic clustering using K-means (without sklearn)."""
    X = dataframe.select_dtypes(include=[np.number]).dropna()
    K = 3  # Number of clusters

    # Randomly initialize centroids
    centroids = X.sample(K).values
    prev_centroids = centroids.copy()
    
    # Iterative process to assign clusters
    for _ in range(100):  # Max iterations
        distances = np.linalg.norm(X.values[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        new_centroids = np.array([X[clusters == i].mean(axis=0) for i in range(K)])
        
        if np.all(new_centroids == prev_centroids):  # If centroids don't change, break early
            break
        prev_centroids = new_centroids
    
    dataframe['cluster'] = clusters
    return dataframe

def perform_geographic_analysis(dataframe, lat_col='latitude', lon_col='longitude'):
    """Perform basic geographic analysis using K-means (without sklearn)."""
    coords = dataframe[[lat_col, lon_col]].dropna()
    K = 3  # Number of clusters for geographic analysis

    # Randomly initialize centroids
    centroids = coords.sample(K).values
    prev_centroids = centroids.copy()
    
    # Iterative process to assign clusters
    for _ in range(100):  # Max iterations
        distances = np.linalg.norm(coords.values[:, np.newaxis] - centroids, axis=2)
        clusters = np.argmin(distances, axis=1)
        new_centroids = np.array([coords[clusters == i].mean(axis=0) for i in range(K)])
        
        if np.all(new_centroids == prev_centroids):  # If centroids don't change, break early
            break
        prev_centroids = new_centroids
    
    dataframe['geo_cluster'] = clusters
    return dataframe

def generate_story(data_summary, analysis_results, charts, advanced_analysis_results):
    """Generate a detailed story using the AI API based on the dataset analysis."""
    data_summary = {
        'columns': list(data_summary['columns']),
        'data_types': {k: str(v) for k, v in data_summary['data_types'].items()},
        'missing_values': data_summary['missing_values'],
        'summary_statistics': data_summary['summary_statistics']
    }

    analysis_results = {
        'correlation_matrix': analysis_results.get('correlation_matrix', None),
        'outliers': analysis_results.get('outliers', None)
    }

    advanced_analysis_results = {
        'outlier_and_anomaly_detection': advanced_analysis_results.get('outlier_and_anomaly_detection', None),
        'regression_analysis': advanced_analysis_results.get('regression_analysis', None),
        'time_series_forecast': advanced_analysis_results.get('time_series_forecast', None),
        'cluster_analysis': advanced_analysis_results.get('cluster_analysis', None),
        'geographic_analysis': advanced_analysis_results.get('geographic_analysis', None)
    }

    charts = [str(chart) for chart in charts]

    prompt = f"""
    Write a comprehensive analysis report based on the following information:
    1. Dataset summary: columns, data types, missing values, and summary statistics.
    2. Analytical insights: correlation matrix, outlier details, and any patterns or anomalies.
    3. Advanced analysis results:
        - Outlier and Anomaly Detection: {advanced_analysis_results['outlier_and_anomaly_detection']}
        - Regression Analysis: {advanced_analysis_results['regression_analysis']}
        - Time Series Forecast: {advanced_analysis_results['time_series_forecast']}
        - Cluster Analysis: {advanced_analysis_results['cluster_analysis']}
        - Geographic Analysis: {advanced_analysis_results['geographic_analysis']}
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

def create_histograms(dataframe, numerical_cols):
    """Generate and save histograms for numerical columns using seaborn."""
    charts = []
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(dataframe[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        chart_path = f"{col}_histogram.png"
        plt.savefig(chart_path, dpi=100)
        plt.close()
        charts.append(chart_path)
    return charts

def create_boxplots(dataframe, numerical_cols):
    """Generate a boxplot for numerical columns using seaborn."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=dataframe[numerical_cols], orient='h')
    plt.title('Boxplot for Outlier Detection')
    plt.xlabel('Values')
    boxplot_path = 'outliers_boxplot.png'
    plt.savefig(boxplot_path, dpi=100)
    plt.close()
    return boxplot_path

def create_correlation_heatmap(correlation_matrix):
    """Generate a heatmap for the correlation matrix using seaborn."""
    if correlation_matrix:
        plt.figure(figsize=(10, 8))
        sns.heatmap(pd.DataFrame(correlation_matrix), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix Heatmap')
        heatmap_path = 'correlation_matrix.png'
        plt.savefig(heatmap_path, dpi=100)
        plt.close()
        return heatmap_path
    return None

def analyze_csv(input_file):
    """Main function to perform the analysis on the provided CSV file."""
    dataframe = pd.read_csv(input_file, encoding='latin1')

    summary, correlation_matrix, outliers = perform_generic_analysis(dataframe)

    advanced_analysis_results = {
        'outlier_and_anomaly_detection': perform_outlier_and_anomaly_detection(dataframe),
        'regression_analysis': perform_regression_analysis(dataframe),
        'time_series_forecast': perform_time_series_analysis(dataframe),
        'cluster_analysis': perform_cluster_analysis(dataframe),
        'geographic_analysis': perform_geographic_analysis(dataframe)
    }

    ai_story = generate_story(summary, {'correlation_matrix': correlation_matrix, 'outliers': outliers}, [], advanced_analysis_results)

    numerical_cols = dataframe.select_dtypes(include=[np.number]).columns

    charts = create_histograms(dataframe, numerical_cols)
    charts.append(create_boxplots(dataframe, numerical_cols))
    if correlation_matrix:
        charts.append(create_correlation_heatmap(correlation_matrix))

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
