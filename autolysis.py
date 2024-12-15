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
import requests
import sys
import matplotlib.pyplot as plt
from pathlib import Path
import json
import seaborn as sns

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
    
    # Ensure there are numeric columns before calculating Z-scores
    if numeric_data.empty:
        print("No numeric columns found for anomaly detection.")
        return dataframe, None
    
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
        new_centroids = []
        for i in range(n_clusters):
            # Ensure we don't calculate the mean of an empty slice
            cluster_data = scaled_data.values[clusters == i]
            if cluster_data.size > 0:  # Check if there's data for the cluster
                new_centroids.append(cluster_data.mean(axis=0))
            else:
                # If no data for this cluster, retain the previous centroid or handle it accordingly
                new_centroids.append(centroids[i])

        # Convert new centroids to a numpy array
        new_centroids = np.array(new_centroids)

        # Check for convergence (if centroids do not change)
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    # Add cluster labels to the dataframe
    dataframe['cluster'] = clusters

    return dataframe[['cluster']]  # Return only the 'cluster' column for analysis purposes


def generate_dynamic_prompt(data_summary, advanced_analysis_results):
    """Generate dynamic prompts based on dataset analysis results."""
    prompt = f"""
    Dataset Summary:
    Columns: {data_summary['columns']}
    Data Types: {json.dumps(data_summary['data_types'], indent=2)}
    Missing Values: {json.dumps(data_summary['missing_values'], indent=2)}
    Summary Statistics: {json.dumps(data_summary['summary_statistics'], indent=2)}

    Advanced Analysis:
    Outlier Detection: {advanced_analysis_results.get('outlier_and_anomaly_detection')}
    Regression Results: {advanced_analysis_results.get('regression_analysis')}
    Time Series Forecast: {advanced_analysis_results.get('time_series_forecast')}
    Cluster Analysis: {advanced_analysis_results.get('cluster_analysis')}
    Geographic Analysis: {advanced_analysis_results.get('geographic_analysis')}
    
    Based on the analysis above, suggest the next steps for exploration.
    """

    return prompt

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
            # Save each figure to a file (PNG) and link to them in the README
            chart_file = f"chart_{charts.index(chart)}.png"
            chart.savefig(chart_file)
            f.write(f"![Chart {charts.index(chart)}]({chart_file})\n")


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

    # Generate dynamic prompt based on advanced analysis results
    dynamic_prompt = generate_dynamic_prompt(summary, advanced_analysis_results)

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


