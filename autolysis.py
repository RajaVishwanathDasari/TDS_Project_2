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
from sklearn.cluster import KMeans
# Set your OpenAI API key
openai_api_key = os.environ.get("AIPROXY_TOKEN")  # AIPROXY_TOKEN is the AI proxy token set in environment
openai.api_key = openai_api_key

openai_api_url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"  #This proxy url is where the above token works

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
    Handles missing values (NaNs) by filling them with the column mean.

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

    # Fill NaN values with the column means
    numeric_data_filled = numeric_data.apply(lambda col: col.fillna(col.mean()), axis=0)

    # Normalize data (min-max scaling)
    scaled_data = (numeric_data_filled - numeric_data_filled.min(axis=0)) / (numeric_data_filled.max(axis=0) - numeric_data_filled.min(axis=0))

    # Ensure the number of rows in the dataframe matches the number of rows in scaled_data
    if len(scaled_data) != len(dataframe):
        # Ensure we are not dropping rows by checking the lengths
        print(f"Warning: Length mismatch between dataframe ({len(dataframe)}) and scaled data ({len(scaled_data)}).")
        # Align rows by matching index
        dataframe = dataframe.iloc[:len(scaled_data)]

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=min(len(scaled_data), max_clusters), random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)

    # Ensure that cluster labels match the original dataframe's length
    dataframe['cluster'] = np.concatenate([cluster_labels, np.full(len(dataframe) - len(cluster_labels), np.nan)])

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
            "description": "Returns the dataset summary statistics including basic descriptive statistics and data types.",
            "parameters": {
                "summary": data_summary
            }
        },
        {
            "name": "get_analysis_insights",
            "description": "Returns analysis results, including correlations between variables and any detected outliers.",
            "parameters": {
                "analysis": analysis_results
            }
        },
        {
            "name": "get_advanced_analysis",
            "description": "Returns advanced analysis results such as time-series trends, clustering insights, and outlier detection summaries.",
            "parameters": {
                "advanced_analysis": advanced_analysis_results
            }
        },
        {
            "name": "describe_charts",
            "description": "Provides a narrative for the charts by describing key patterns, trends, and outliers observed in the visualizations.",
            "parameters": {
                "charts_info": [str(chart) for chart in charts]  # Convert chart objects to strings or meaningful descriptions
            }
        }
    ]

    # Updated prompt for clearer structure and expectations from the LLM
    prompt = f"""
    You are a data analysis assistant. Your task is to generate a comprehensive, structured report that integrates the following components:

    1. **Dataset Summary**:
        Provide a summary of the dataset including the key statistics (such as mean, median, and standard deviation), data types of each column, and any missing values in the dataset.
        
    2. **Analysis Insights**:
        Present insights derived from the dataset analysis. This includes a correlation matrix that identifies relationships between numeric variables and any significant outliers detected based on Z-scores or other methods.

    3. **Advanced Analysis**:
        Offer deeper insights derived from advanced analyses such as:
            - Time-series analysis (e.g., trends, seasonality, etc.),
            - Clustering results (including insights on groupings),
            - Outlier detection summaries (showing which values were identified as outliers and their impact on the dataset).

    4. **Visualizations**:
        Describe key insights from the charts provided. For each chart, explain what the visual representation reveals about the dataset. Focus on significant patterns, anomalies, trends, and correlations that are visible in the charts.

    You should provide a narrative that logically connects each of these components. Begin with the dataset summary and then proceed to the analysis insights. Make sure to describe how the advanced analysis results build on the basic analysis, leading naturally into the interpretation of the visualizations. The goal is to create a cohesive, easy-to-understand report that highlights both high-level trends and specific insights from the dataset.
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
