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
from pathlib import Path
import sys

# Set your OpenAI API key
openai_api_key = os.environ.get("OPENAI_API_KEY")  # Ensure you set your OpenAI API key
openai.api_key = openai_api_key

# Set the base URL for the OpenAI API to the proxy server
openai.api_base = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

# Function to suggest types of analysis to perform based on dataset structure
def suggest_analysis(dataframe):
    """
    Query OpenAI to suggest the analysis techniques based on the dataset.
    """
    prompt = f"Based on a dataset with {len(dataframe.columns)} columns, what analysis techniques should be performed? Options could include summary statistics, outlier detection, clustering, time series analysis, etc."
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an assistant that helps determine which analyses should be done on a dataset."},
                  {"role": "user", "content": prompt}],
        max_tokens=300,
        functions=[
            {
                "name": "suggest_analysis_types",
                "description": "Suggest the types of analyses that should be performed on the dataset.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["columns"]
                }
            }
        ]
    )

    return response['choices'][0]['message']['content']


# Function to generate visualizations based on user input or system's suggestion
def suggest_visualizations(dataframe):
    """
    Query OpenAI to suggest which visualizations to create based on dataset analysis.
    """
    prompt = f"Given a dataset with {len(dataframe.columns)} columns, suggest visualizations such as histograms, scatter plots, or correlation heatmaps to display important patterns."
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an assistant that helps decide what visualizations should be created."},
                  {"role": "user", "content": prompt}],
        max_tokens=300,
        functions=[
            {
                "name": "suggest_visualization_types",
                "description": "Suggest which types of visualizations (e.g., histograms, scatter plots, correlation heatmaps) should be generated.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columns": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["columns"]
                }
            }
        ]
    )

    return response['choices'][0]['message']['content']


# Function to generate a narrative report based on analyses
def generate_report(data_summary, analysis_results, charts, advanced_analysis_results):
    """
    Generate a detailed narrative report based on the analysis results.
    """
    prompt = f"Based on the analysis of the dataset, write a detailed report. Include summaries, insights from analysis, charts, and advanced findings."
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini", 
        messages=[{"role": "system", "content": "You are an assistant that writes detailed reports based on data analysis."},
                  {"role": "user", "content": prompt}],
        max_tokens=1500,
        functions=[
            {
                "name": "generate_report",
                "description": "Generate a detailed report incorporating analysis results and visual insights.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                        "analysis_results": {"type": "string"},
                        "charts": {"type": "array", "items": {"type": "string"}},
                        "advanced_analysis_results": {"type": "string"},
                    },
                    "required": ["summary", "analysis_results"]
                }
            }
        ]
    )

    return response['choices'][0]['message']['content']


# Function to detect outliers in the dataset
def detect_outliers(dataframe, threshold=3):
    """
    Perform outlier detection using the Z-score method.
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return dataframe, {}

    z_scores = np.abs((numeric_data - numeric_data.mean()) / numeric_data.std())
    outlier_summary = (z_scores > threshold).sum(axis=0).to_dict()
    dataframe['outliers'] = (z_scores > threshold).any(axis=1).astype(int)

    return dataframe, outlier_summary


# Function to perform clustering on the dataset
def perform_clustering(dataframe):
    """
    Perform clustering using a dynamic method without sklearn (e.g., K-means implementation).
    """
    numeric_data = dataframe.select_dtypes(include=[np.number])

    if numeric_data.empty:
        return dataframe, {}

    # Normalize data
    scaled_data = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
    
    # Initialize centroids (for simplicity)
    centroids = scaled_data.sample(n=3, random_state=42).values

    def calculate_distance(X, centroids):
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def assign_clusters(X, centroids):
        distances = calculate_distance(X, centroids)
        return np.argmin(distances, axis=1)
    
    prev_centroids = np.zeros(centroids.shape)
    for _ in range(300):
        clusters = assign_clusters(scaled_data.values, centroids)
        new_centroids = np.array([scaled_data.values[clusters == i].mean(axis=0) for i in range(3)])
        if np.all(np.abs(new_centroids - prev_centroids) < 1e-4):
            break
        prev_centroids = new_centroids

    dataframe['cluster'] = clusters
    return dataframe[['cluster']], clusters


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


# Main function to analyze a CSV file and generate a detailed report
def analyze_csv(input_file):
    """
    Main function to analyze a CSV file and generate a detailed report.
    """
    dataframe = pd.read_csv(input_file, encoding='latin1')

    # Step 1: Call OpenAI to suggest which analysis methods to perform on the dataset
    analysis_suggestions = suggest_analysis(dataframe)
    print(f"Suggested Analysis: {analysis_suggestions}")

    # Step 2: Perform analysis based on OpenAI's suggestions
    analysis_config = {
        'outlier_detection': 'outlier' in analysis_suggestions.lower(),
        'clustering': 'clustering' in analysis_suggestions.lower(),
        'time_series_analysis': 'time series' in analysis_suggestions.lower(),
        'visualizations': 'visualizations' in analysis_suggestions.lower()
    }

    # Perform outlier detection if suggested
    outlier_summary = None
    if analysis_config['outlier_detection']:
        dataframe, outlier_summary = detect_outliers(dataframe)

    # Perform clustering if suggested
    cluster_results = None
    if analysis_config['clustering']:
        cluster_results, clusters = perform_clustering(dataframe)

    # Step 3: Generate visualizations if suggested
    charts = []
    if analysis_config['visualizations']:
        charts = create_histograms(dataframe)

    # Step 4: Generate the final report
    ai_report = generate_report(
        data_summary="Summary of the dataset",
        analysis_results={"outlier_summary": outlier_summary, "clusters": cluster_results},
        charts=charts,
        advanced_analysis_results={"clustering_results": cluster_results}
    )

    print(ai_report)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    analyze_csv(input_file)






