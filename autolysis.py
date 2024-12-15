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
# Import necessary libraries
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set up Matplotlib backend for non-interactive use (important for headless environments)
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import sys
from pathlib import Path
import json

# Set the OpenAI API URL and authentication token (ensure the token is set securely in the environment)
openai_api_url = "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
openai_api_key = os.environ.get("AIPROXY_TOKEN")  # Load the token from the environment variables

def generate_story(data_summary, analysis_results, charts):
    """
    Generates a detailed story about the dataset's analysis using the OpenAI API.
    
    Parameters:
    - data_summary: A dictionary containing summary of the data (columns, types, etc.)
    - analysis_results: A dictionary containing analysis results (correlation matrix, outliers)
    - charts: A list of chart file paths
    
    Returns:
    - A string with the AI-generated story based on the provided data and analysis.
    """
    
    # Convert data types to basic Python types to ensure compatibility with JSON
    data_summary = {
        'columns': list(data_summary['columns']),
        'data_types': dict(data_summary['data_types']),
        'missing_values': dict(data_summary['missing_values']),
        'summary_statistics': {k: v.to_dict() if isinstance(v, pd.DataFrame) else v for k, v in data_summary['summary_statistics'].items()}
    }

    # Prepare analysis results for the AI prompt
    analysis_results = {
        'correlation_matrix': analysis_results.get('correlation_matrix', None),
        'outliers': analysis_results.get('outliers', None)
    }

    # Convert charts (file paths) to a simple list of strings
    charts = [str(chart) for chart in charts]

    # Create the AI prompt that includes data summary, analysis results, and charts
    prompt = f"""
    Write a detailed story about the analysis of a dataset. The story should describe:
    1. A brief summary of the data: what the data consists of, column names, types, and any relevant patterns or issues.
    2. The analysis performed: including steps like correlation, outlier detection, and any other relevant analysis.
    3. Insights discovered: any patterns, anomalies, trends, or key takeaways from the data.
        - Describe the key correlations found in the data and their implications.
        - Discuss the outliers detected and what they mean for the data analysis.
    4. Visualizations: describe the visualizations and what insights they provide.
        - Discuss the histograms, boxplot, and correlation heatmap, and how they help illustrate the findings.
    5. Implications: what actions can be taken based on the insights? What do these findings mean for decision-making?
    
    Data Summary: {data_summary}
    
    Analysis Results:
    {analysis_results}

    Charts: {charts}
    """
    
    # Set the request headers with authorization token
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    # Prepare the data payload for the API request
    data = {
        "model": "gpt-4o-mini",  # Select the model for the OpenAI API
        "messages": [
            {"role": "system", "content": "You are a data analysis assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 500  # Limit the response to avoid excessive tokens
    }

    # Send a POST request to the OpenAI API
    response = requests.post(openai_api_url, headers=headers, json=data)
    
    # Check if the response is successful and return the AI-generated story
    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content'].strip()
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return "AI generation failed."

def perform_generic_analysis(dataframe):
    """
    Performs basic analysis on the given DataFrame, including dataset summary, correlation matrix, and outlier detection.

    Parameters:
    - dataframe: The input DataFrame to analyze.

    Returns:
    - A tuple containing:
        - A dictionary with dataset summary (columns, data types, missing values, etc.)
        - The correlation matrix of numerical columns
        - A summary of detected outliers (based on IQR method)
    """
    
    # Generate summary statistics for the dataset
    summary = {
        'columns': list(dataframe.columns),
        'data_types': dict(dataframe.dtypes),
        'missing_values': dataframe.isnull().sum().to_dict(),
        'summary_statistics': dataframe.describe()  # Get basic descriptive statistics
    }
    
    # Select only numeric columns for correlation analysis and outlier detection
    numeric_columns = dataframe.select_dtypes(include=[np.number])
    
    # Compute correlation matrix for numeric columns
    correlation_matrix = numeric_columns.corr() if numeric_columns.shape[1] > 1 else None

    # Detect outliers using the IQR method
    if numeric_columns.shape[1] > 0:
        Q1 = numeric_columns.quantile(0.25)
        Q3 = numeric_columns.quantile(0.75)
        IQR = Q3 - Q1
        outliers = ((numeric_columns < (Q1 - 1.5 * IQR)) | (numeric_columns > (Q3 + 1.5 * IQR))).sum()  # Count outliers
    else:
        outliers = None

    return summary, correlation_matrix, outliers

def create_histograms(dataframe, numerical_cols, output_folder):
    """
    Creates histograms for the numerical columns in the DataFrame and saves them to the specified folder.

    Parameters:
    - dataframe: The DataFrame containing the data.
    - numerical_cols: A list of column names for numerical data.
    - output_folder: The folder where the generated histogram images will be saved.

    Returns:
    - A list of file paths to the saved histograms.
    """
    
    charts = []  # Initialize list to store chart paths
    
    # Create histograms for each numerical column
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(dataframe[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        chart_path = os.path.join(output_folder, f"{col}_histogram.png")
        plt.savefig(chart_path, dpi=100)
        plt.close()
        charts.append(chart_path)  # Append chart path to list
    
    return charts

def create_boxplots(dataframe, numerical_cols, output_folder):
    """
    Creates a boxplot to visualize outliers and saves it to the specified folder.

    Parameters:
    - dataframe: The DataFrame containing the data.
    - numerical_cols: A list of column names for numerical data.
    - output_folder: The folder where the generated boxplot image will be saved.

    Returns:
    - The file path of the saved boxplot image.
    """
    
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=dataframe[numerical_cols])
    plt.title('Outlier Detection - Boxplot')
    boxplot_path = os.path.join(output_folder, 'outliers_boxplot.png')
    plt.savefig(boxplot_path, dpi=100)
    plt.close()
    return boxplot_path

def create_correlation_heatmap(correlation_matrix, output_folder):
    """
    Creates a heatmap to visualize the correlation matrix and saves it to the specified folder.

    Parameters:
    - correlation_matrix: The correlation matrix to visualize.
    - output_folder: The folder where the generated heatmap image will be saved.

    Returns:
    - The file path of the saved heatmap image, or None if the correlation matrix is empty.
    """
    
    if correlation_matrix is not None:
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title('Correlation Matrix')
        heatmap_path = os.path.join(output_folder, 'correlation_matrix.png')
        plt.savefig(heatmap_path, dpi=100)
        plt.close()
        return heatmap_path
    return None

def create_readme(ai_story, charts, summary, output_folder):
    """
    Creates a README file summarizing the analysis, including AI-generated insights and visualizations.

    Parameters:
    - ai_story: The AI-generated narrative summarizing the analysis.
    - charts: A list of chart file paths to include in the README.
    - summary: The dataset summary.
    - output_folder: The folder where the README file will be saved.
    """
    
    with open(os.path.join(output_folder, 'README.md'), 'w', encoding='utf-16') as readme_file:
        readme_file.write("# Automated Data Analysis Report\n\n")
        readme_file.write("## Dataset Summary\n")
        readme_file.write(f"Columns: {summary['columns']}\n")
        readme_file.write(f"Data Types: {summary['data_types']}\n")
        readme_file.write(f"Missing Values: {summary['missing_values']}\n")
        readme_file.write("Summary Statistics:\n")
        readme_file.write(f"{summary['summary_statistics']}\n\n")

        readme_file.write("## AI-Generated Story\n")
        readme_file.write(ai_story + "\n\n")
        
        readme_file.write("## Data Visualizations\n")
        for chart in charts:
            readme_file.write(f"![{chart}]({chart})\n")

def analyze_csv(input_file):
    """
    Performs the entire analysis process for a given CSV file, including:
    - Data analysis
    - AI story generation
    - Visualization creation
    - README generation
    
    Parameters:
    - input_file: The path to the CSV file to analyze.
    """
    
    # Use the current directory as the output folder
    output_folder = os.getcwd()

    # Load dataset with UTF-16 encoding (ensure correct handling of non-ASCII characters)
    dataframe = pd.read_csv(input_file, encoding='latin1')

    # Step 1: Perform generic analysis
    summary, correlation_matrix, outliers = perform_generic_analysis(dataframe)

    # Step 2: Generate AI-powered story based on analysis
    ai_story = generate_story(summary, {'correlation_matrix': correlation_matrix, 'outliers': outliers}, [])

    # Step 3: Create visualizations (histograms, boxplots, correlation matrix)
    numerical_cols = dataframe.select_dtypes(include=["float64", "int64"]).columns
    charts = []

    # Create histograms for numerical columns
    charts.extend(create_histograms(dataframe, numerical_cols, output_folder))

    # Create boxplot for outliers
    charts.append(create_boxplots(dataframe, numerical_cols, output_folder))

    # Create correlation heatmap if applicable
    if correlation_matrix is not None:
        charts.append(create_correlation_heatmap(correlation_matrix, output_folder))

    # Step 4: Generate README file with analysis summary, AI insights, and visualizations
    create_readme(ai_story, charts, summary, output_folder)
    print(f"Analysis complete. Check {output_folder}/README.md, chart files")

# Main execution block
if __name__ == "__main__":
    # Ensure CSV filename is passed as an argument
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <dataset.csv>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    if not Path(input_file).is_file():
        print(f"File {input_file} not found.")
        sys.exit(1)

    analyze_csv(input_file)

