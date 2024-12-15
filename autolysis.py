# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "seaborn",
#   "requests",
#   "pathlib"
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
    try:
        # Ensure that 'target_column' exists in the dataframe
        if 'target_column' not in dataframe.columns:
            return "Error: 'target_column' not found in the dataset."
        
        # Prepare X (features) and y (target)
        X = dataframe.dropna().select_dtypes(include=[np.number]).drop(columns=['target_column'], errors='ignore')
        y = dataframe['target_column']
        
        if X.empty or len(X) != len(y):
            return "Error: The feature matrix X or the target vector y is empty or has mismatched lengths."

        # Add intercept (bias term)
        X = np.c_[np.ones(len(X)), X]  # Adding a column of ones for the intercept term
        
        # Perform linear regression using the normal equation: (X^T * X)^(-1) * X^T * y
        beta = np.linalg.inv(X.T @ X) @ X.T @ y
        
        # Save coefficients (including intercept)
        regression_results = dict(zip(['Intercept'] + X.columns.tolist(), beta))
        
        return regression_results

    except KeyError as e:
        return f"Error: Missing required column in dataframe: {str(e)}"
    
    except ValueError as e:
        return f"Error: Value error during regression: {str(e)}"
    
    except np.linalg.LinAlgError as e:
        return f"Error: Linear algebra error during regression, possibly due to a singular matrix: {str(e)}"
    
    except Exception as e:
        return f"An unexpected error occurred during regression analysis: {str(e)}"


def perform_time_series_analysis(dataframe):
    """Perform basic time series analysis and forecasting using mean."""
    # Try to automatically detect a date column
    date_column = None
    for col in dataframe.columns:
        if 'date' in col.lower() or 'time' in col.lower():  # Search for columns with "date" or "time" in the name
            date_column = col
            break

    if date_column is None:
        print("No date column found in the dataset. Skipping time series analysis.")
        return None  # Return None if no date column is found

    # Convert the detected column to datetime format
    dataframe[date_column] = pd.to_datetime(dataframe[date_column])

    # Proceed with the time series analysis if a valid date column is found
    dataframe.set_index(date_column, inplace=True)
    ts_data = dataframe['target_column'] if 'target_column' in dataframe.columns else None
    if ts_data is not None:
        forecast = ts_data.mean()  # Basic forecast: the mean of the data
        return forecast
    return None



def perform_cluster_analysis(dataframe, n_clusters=3, max_iter=100):
    """Perform cluster analysis using KMeans without sklearn and add cluster labels to the dataframe."""
    # Select only numeric columns for clustering
    numerical_cols = dataframe.select_dtypes(include=[np.number])

    # If there are no numeric columns, return None
    if numerical_cols.empty:
        print("No numeric columns found for clustering.")
        return None

    # Normalize the data using min-max scaling
    min_vals = numerical_cols.min()
    max_vals = numerical_cols.max()
    scaled_data = (numerical_cols - min_vals) / (max_vals - min_vals)

    # Initialize centroids randomly by selecting random data points
    centroids = scaled_data.sample(n_clusters, random_state=42).values

    # Function to assign clusters
    def assign_clusters(data, centroids):
        distances = np.linalg.norm(data[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    # K-means algorithm (basic implementation)
    for _ in range(max_iter):
        # Assign clusters
        clusters = assign_clusters(scaled_data.values, centroids)

        # Recalculate centroids
        new_centroids = np.array([scaled_data.values[clusters == i].mean(axis=0) for i in range(n_clusters)])

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Add cluster labels to the dataframe
    dataframe['cluster'] = clusters

    return dataframe[['cluster']]  # Return only the 'cluster' column for analysis purposes

def perform_geographic_analysis(dataframe, lat_col='latitude', lon_col='longitude'):
    """Perform basic geographic analysis using K-means (without sklearn)."""
    try:
        # Check if the necessary columns exist
        if lat_col not in dataframe.columns or lon_col not in dataframe.columns:
            return f"Error: Required columns '{lat_col}' and '{lon_col}' not found in dataset"
        
        # Drop rows with missing values in the latitude or longitude columns
        coords = dataframe[[lat_col, lon_col]].dropna()
        
        if coords.empty:
            return "Error: No valid latitude/longitude data available after removing missing values"
        
        K = 3  # Number of clusters for geographic analysis
        
        # Randomly initialize centroids
        centroids = coords.sample(K).values
        prev_centroids = centroids.copy()
        
        # Iterative process to assign clusters
        for _ in range(100):  # Max iterations
            distances = np.linalg.norm(coords.values[:, np.newaxis] - centroids, axis=2)
            clusters = np.argmin(distances, axis=1)
            new_centroids = np.array([coords[clusters == i].mean(axis=0) for i in range(K)])
            
            # If centroids don't change, break early
            if np.all(new_centroids == prev_centroids):
                break
            prev_centroids = new_centroids
        
        # Add a new column for clusters in the dataframe
        dataframe['geo_cluster'] = clusters
        return dataframe

    except Exception as e:
        # Handle any unexpected errors and return a message
        return f"An error occurred during geographic analysis: {str(e)}"


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

def create_histograms(dataframe, bins=10):
    """Create histograms using seaborn."""
    histograms = []
    numeric_columns = dataframe.select_dtypes(include=[np.number])

    for col in numeric_columns.columns:
        plot = sns.histplot(dataframe[col], bins=bins)
        plot.set(title=f"Histogram of {col}")
        histograms.append(plot)

    return histograms

def create_boxplots(dataframe):
    """Create boxplots using seaborn."""
    boxplots = []
    numeric_columns = dataframe.select_dtypes(include=[np.number])

    for col in numeric_columns.columns:
        plot = sns.boxplot(data=dataframe, x=col)
        plot.set(title=f"Boxplot of {col}")
        boxplots.append(plot)

    return boxplots

def create_correlation_heatmap(dataframe):
    """Create a correlation heatmap using seaborn."""
    numeric_data = dataframe.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    heatmap = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    heatmap.set(title="Correlation Heatmap")
    return heatmap

def analyze_csv(input_file):
    """Main function to perform the analysis on the provided CSV file."""
    dataframe = pd.read_csv(input_file, encoding='latin1')

    # Perform basic analysis
    summary, correlation_matrix, outliers = perform_generic_analysis(dataframe)

    # Perform advanced analysis
    advanced_analysis_results = {
        'outlier_and_anomaly_detection': perform_outlier_and_anomaly_detection(dataframe),
        'regression_analysis': perform_regression_analysis(dataframe),
        'time_series_forecast': perform_time_series_analysis(dataframe),
        'cluster_analysis': perform_cluster_analysis(dataframe),
        'geographic_analysis': perform_geographic_analysis(dataframe)
    }

    # Generate AI story with basic and advanced analysis results
    ai_story = generate_story(
        summary,
        {'correlation_matrix': correlation_matrix, 'outliers': outliers},
        [],  # Empty chart list initially, populated later
        advanced_analysis_results  # Include advanced analysis results
    )

    # Extract numerical columns for visualizations
    numerical_cols = dataframe.select_dtypes(include=[np.number]).columns

    # Create charts
    charts = create_histograms(dataframe, numerical_cols)
    charts.append(create_boxplots(dataframe, numerical_cols))
    if correlation_matrix:
        charts.append(create_correlation_heatmap(correlation_matrix))

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
