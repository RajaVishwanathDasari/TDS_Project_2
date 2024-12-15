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
openai_api_key = os.environ.get("AIPROXY_TOKEN")  # Ensure you set your OpenAI API key
openai.api_key = openai_api_key

openai_api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

def perform_generic_analysis(dataframe):
    """
    Perform basic analysis of the dataframe, including summary statistics, missing values, 
    data types, and correlation analysis for numeric columns.
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
    Perform basic time series analysis such as detecting trends and seasonality using time-based grouping.
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
    """
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
    
    def calculate_distance(X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def assign_clusters(X, centroids):
        distances = calculate_distance(X, centroids)
        return np.argmin(distances, axis=1)
    
    prev_centroids = np.zeros(centroids.shape)
    max_iterations = 300
    tolerance = 1e-4
    for _ in range(max_iterations):
        clusters = assign_clusters(scaled_data.values, centroids)
        new_centroids = np.array([scaled_data.values[clusters == i].mean(axis=0) for i in range(n_clusters)])
        if np.all(np.abs(new_centroids - prev_centroids) < tolerance):
            break
        prev_centroids = new_centroids

    dataframe['cluster'] = clusters
    return dataframe[['cluster']]

def create_histograms(dataframe, bins=10):
    """
    Create histograms for numeric columns with labels and annotations.
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
        plt.annotate(f"Mean: {dataframe[col].mean():.2f}", xy=(0.7, 0.9), xycoords='axes fraction')
        plt.annotate(f"Max: {dataframe[col].max():.2f}", xy=(0.7, 0.85), xycoords='axes fraction')
        histograms.append(plt.gcf())
        plt.close()

    return histograms

def create_correlation_heatmap(dataframe):
    """
    Create a labeled correlation heatmap for numeric columns.
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])
    corr_matrix = numeric_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i != j and abs(corr_matrix.iloc[i, j]) > 0.7:
                plt.text(j, i, "High Corr", ha='center', va='center', color='red', fontsize=9)

    plt.tight_layout()
    return plt.gcf()

def generate_story(data_summary, analysis_results, charts, advanced_analysis_results):
    """
    Generate a dynamic, customizable narrative using LLM API based on dataset analysis, with iterative refinements.
    """
    # Dynamically assemble prompt components based on available analysis
    prompt_components = []

    if data_summary:
        prompt_components.append(f"Dataset Summary: {data_summary}")
    if analysis_results:
        prompt_components.append(f"Analysis Insights: {analysis_results}")
    if advanced_analysis_results:
        prompt_components.append(f"Advanced Analysis Results: {advanced_analysis_results}")
    if charts:
        prompt_components.append(f"Charts Insights: {charts}")

    # Formulate prompt
    prompt = "\n\n".join(prompt_components)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",  # Adjust the model based on the complexity required
            messages=[
                {"role": "system", "content": "You are an assistant that generates detailed reports based on dynamic data analysis."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1500,
        )
        
        narrative = response['choices'][0]['message']['content']
        return narrative
    except Exception as e:
        print(f"Error occurred: {e}")
        return "AI generation failed."

def analyze_csv(input_file):
    """
    Main function to analyze a CSV file and generate a detailed report.
    """
    dataframe = pd.read_csv(input_file, encoding='latin1')

    # Perform analysis dynamically based on input configurations (could be extended)
    analysis_config = {
        'generic_analysis': True,
        'outlier_detection': True,
        'time_series_analysis': {'target_column': 'target_column', 'date_column': 'date_column'},
        'clustering': True
    }

    summary, correlation_matrix = perform_generic_analysis(dataframe)
    outlier_summary = None
    if analysis_config['outlier_detection']:
        dataframe, outlier_summary = detect_outliers_zscore(dataframe)
        
    advanced_analysis_results = {}
    if analysis_config['time_series_analysis']:
        advanced_analysis_results['time_series_analysis'] = perform_time_series_analysis(dataframe, 
                                                                                         analysis_config['time_series_analysis']['target_column'], 
                                                                                         analysis_config['time_series_analysis']['date_column'])
    if analysis_config['clustering']:
        advanced_analysis_results['cluster_analysis'] = dynamic_cluster_analysis(dataframe)

    # Create visualizations
    charts = create_histograms(dataframe)
    charts.append(create_correlation_heatmap(dataframe))

    # Generate narrative dynamically based on selected analyses
    ai_story = generate_story(summary, {'correlation_matrix': correlation_matrix, 'outlier_summary': outlier_summary}, charts, advanced_analysis_results)

    print(ai_story)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    analyze_csv(input_file)





