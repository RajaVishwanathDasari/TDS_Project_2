# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "httpx",
#   "pandas",
#   "numpy",
#   "requests",
#   "pathlib",
#   "openai",
#   "seaborn",
#   "matplotlib",
#   "scikit-learn"
# ]
# ///
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import requests
from pathlib import Path
import sys

# Set your OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")  # Ensure you set your OpenAI API key
openai.api_key = openai_api_key

openai_api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def perform_generic_analysis(dataframe):
    """
    Perform basic analysis of the dataframe, including summary statistics, missing values, 
    data types, and correlation analysis for numeric columns.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.

    Returns:
    tuple: Summary statistics and correlation matrix as dictionaries.
    """
    summary = {
        'columns': list(dataframe.columns),
        'data_types': dict(dataframe.dtypes),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'summary_statistics': dataframe.describe(include='all').to_dict()
    }

    # Correlation matrix for numeric columns
    numeric_columns = dataframe.select_dtypes(include=[np.number])
    correlation_matrix = numeric_columns.corr().to_dict() if not numeric_columns.empty else None

    return summary, correlation_matrix

def detect_outliers_zscore(dataframe, threshold=3):
    """
    Detect outliers in numeric columns using the Z-score method.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.
    threshold (float): The Z-score threshold to identify outliers.

    Returns:
    tuple: Updated dataframe with outlier column and a summary of detected outliers.
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("No numeric columns for outlier detection.")
        return dataframe, {}

    z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
    outlier_summary = (z_scores > threshold).sum(axis=0).to_dict()
    dataframe['outliers'] = (z_scores > threshold).any(axis=1).astype(int)

    return dataframe, outlier_summary

def perform_time_series_analysis(dataframe, target_column, date_column):
    """
    Perform basic time series analysis such as detecting trends and seasonality 
    using time-based grouping.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.
    target_column (str): The column to analyze trends.
    date_column (str): The column representing dates.

    Returns:
    dict: Summary of trends and grouped statistics.
    """
    try:
        dataframe[date_column] = pd.to_datetime(dataframe[date_column])
        dataframe.sort_values(by=date_column, inplace=True)

        # Aggregate target_column by month and year for trend analysis
        dataframe['year_month'] = dataframe[date_column].dt.to_period('M')
        grouped = dataframe.groupby('year_month')[target_column].mean().reset_index()

        # Identify trends (e.g., overall increase/decrease)
        trend = "increasing" if grouped[target_column].iloc[-1] > grouped[target_column].iloc[0] else "decreasing"

        return {
            'trend': trend,
            'monthly_averages': grouped.to_dict(orient='records')
        }
    except Exception as e:
        print(f"Time series analysis failed: {e}")
        return None

def dynamic_cluster_analysis(dataframe, max_clusters=5):
    """
    Perform dynamic clustering analysis using K-means, with the number of clusters determined by data complexity.
    This version does not use sklearn.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.
    max_clusters (int): Maximum number of clusters.

    Returns:
    pd.DataFrame: DataFrame with assigned clusters.
    """
    
    # Select numeric columns
    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        print("No numeric columns for clustering.")
        return None

    # Normalize data
    scaled_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())

    # Determine the number of clusters (max_clusters parameter)
    n_clusters = min(len(scaled_data), max_clusters)
    
    # Initialize random centroids
    centroids = scaled_data.sample(n=n_clusters, random_state=42).values
    
    # Function to calculate the Euclidean distance between points and centroids
    def calculate_distance(X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    # Function to assign clusters based on closest centroids
    def assign_clusters(X, centroids):
        distances = calculate_distance(X, centroids)
        return np.argmin(distances, axis=1)
    
    # K-means clustering loop
    prev_centroids = np.zeros(centroids.shape)
    max_iterations = 300
    tolerance = 1e-4
    for _ in range(max_iterations):
        # Assign clusters based on current centroids
        clusters = assign_clusters(scaled_data.values, centroids)
        
        # Update centroids
        new_centroids = np.array([scaled_data.values[clusters == i].mean(axis=0) for i in range(n_clusters)])
        
        # Check for convergence (if centroids don't change significantly)
        if np.all(np.abs(new_centroids - prev_centroids) < tolerance):
            break
        
        prev_centroids = new_centroids

    # Add cluster labels to the dataframe
    dataframe['cluster'] = clusters

    return dataframe[['cluster']]
def create_histograms(dataframe, bins=10):
    """
    Create histograms for numeric columns with labels and annotations.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.
    bins (int): Number of bins for histograms.

    Returns:
    list: Matplotlib figures for histograms.
    """
    histograms = []
    numeric_columns = dataframe.select_dtypes(include=[np.number])

    for col in numeric_columns.columns:
        plt.figure()
        sns.histplot(dataframe[col], bins=bins, kde=True, color='skyblue')
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(axis='y', linestyle='--', alpha=0.7)

        # Annotate significant points
        plt.annotate(f"Mean: {dataframe[col].mean():.2f}", xy=(0.7, 0.9), xycoords='axes fraction')
        plt.annotate(f"Max: {dataframe[col].max():.2f}", xy=(0.7, 0.85), xycoords='axes fraction')
        histograms.append(plt.gcf())
        plt.close()

    return histograms

def create_correlation_heatmap(dataframe):
    """
    Create a labeled correlation heatmap for numeric columns.

    Parameters:
    dataframe (pd.DataFrame): The dataset to analyze.

    Returns:
    Matplotlib figure: Correlation heatmap.
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)

    # Add key insights as annotations
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j and abs(corr_matrix.iloc[i, j]) > 0.7:
                plt.text(j, i, "High Corr", ha='center', va='center', color='red', fontsize=9)

    plt.tight_layout()
    return plt.gcf()

def generate_story(data_summary, analysis_results, charts, advanced_analysis_results):
    """
    Generate a detailed narrative using LLM API based on dataset analysis, with iterative refinements.

    Parameters:
    data_summary (dict): Summary statistics of the dataset.
    analysis_results (dict): Results of various analyses.
    charts (list): Visualization charts.
    advanced_analysis_results (dict): Advanced analyses results.

    Returns:
    str: Generated narrative.
    """
    # Define the functions the AI can call
    functions = [
        {
            "name": "get_data_summary",
            "description": "Returns the dataset summary statistics.",
            "parameters": {
                "summary": data_summary
            }
        },
        {
            "name": "get_analysis_insights",
            "description": "Returns analysis results like correlations and outliers.",
            "parameters": {
                "analysis": analysis_results
            }
        },
        {
            "name": "get_advanced_analysis",
            "description": "Returns advanced analysis like time-series and clustering insights.",
            "parameters": {
                "advanced_analysis": advanced_analysis_results
            }
        },
        {
            "name": "describe_charts",
            "description": "Describes key observations from charts.",
            "parameters": {
                "charts_info": [str(chart) for chart in charts]  # Convert chart objects to strings or meaningful descriptions
            }
        }
    ]

    # Prepare the API request prompt
    prompt = f"""
    You are a data analysis assistant. Based on the following data analysis, generate a comprehensive report:
    1. Dataset summary
    2. Insights from various analyses
    3. Advanced analysis results (e.g., trends, clusters, outliers)
    4. Observations from the visualizations provided.

    The functions to call for specific insights are as follows:
    """

    # Request to the OpenAI API with function calling enabled
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # Use the GPT-4o-mini model
            messages=[
                {"role": "system", "content": "You are an assistant that helps generate detailed reports from data analysis."},
                {"role": "user", "content": prompt},
            ],
            functions=functions,  # Pass the available functions
            function_call="auto",  # Let GPT-4o-mini automatically decide which function to call
        )

        if response.status_code == 200:
            narrative = response['choices'][0]['message']['content']
            return narrative
        else:
            print(f"Error: {response.status_code}\n{response.text}")
            return "AI generation failed."

    except Exception as e:
        print(f"Error occurred: {e}")
        return "AI generation failed."


def analyze_csv(input_file):
    """
    Main function to analyze a CSV file and generate a detailed report.

    Parameters:
    input_file (str): Path to the CSV file.
    """
    dataframe = pd.read_csv(input_file, encoding='latin1')

    # Perform analysis
    summary, correlation_matrix = perform_generic_analysis(dataframe)
    dataframe, outlier_summary = detect_outliers_zscore(dataframe)

    advanced_analysis_results = {
        'time_series_analysis': perform_time_series_analysis(dataframe, 'target_column', 'date_column'),
        'cluster_analysis': dynamic_cluster_analysis(dataframe),
        'outlier_detection': outlier_summary
    }

    # Create visualizations
    charts = create_histograms(dataframe)
    charts.append(create_correlation_heatmap(dataframe))

    # Generate narrative
    ai_story = generate_story(summary, {'correlation_matrix': correlation_matrix}, charts, advanced_analysis_results)

    print(ai_story)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    analyze_csv(input_file)





