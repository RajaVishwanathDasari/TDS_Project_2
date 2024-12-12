import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import sys
from datetime import datetime
import json

# Retrieve the API token from environment variable
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Ensure the token is set
if AIPROXY_TOKEN is None:
    raise ValueError("AIPROXY_TOKEN environment variable is not set.")

# Initialize OpenAI API with the retrieved token
openai.api_key = AIPROXY_TOKEN

def load_data(file_path):
    """Load and clean data from a CSV file."""
    try:
        data = pd.read_csv(file_path)
        print("Data loaded successfully")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_with_llm(data):
    """Send data analysis request to GPT-4o-Mini via OpenAI API."""
    # Convert the data to a JSON format or any other structure that works
    data_json = data.head().to_json(orient='records')
    
    prompt = f"""
    Analyze the following dataset and provide insights, correlations, and potential trends:
    {data_json}
    """

    try:
        # Request LLM (GPT-4o-Mini) for analysis
        response = openai.Completion.create(
            model="gpt-4o-mini",  # Using GPT-4o-Mini model
            prompt=prompt,
            max_tokens=500,
            temperature=0.7
        )
        analysis_results = response.choices[0].text.strip()
        return analysis_results
    except openai.error.AuthenticationError as e:
        print(f"Authentication Error: {e}")
        return "Authentication failed, please check your API token."
    except Exception as e:
        print(f"Error in LLM request: {e}")
        return "Analysis not available."

def summarize_statistics(data):
    """Generate summary statistics for the dataset."""
    summary_stats = data.describe()
    print("\nSummary Statistics:")
    print(summary_stats)

def count_missing_values(data):
    """Count missing values in the dataset."""
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    print("\nMissing Values:")
    print(pd.DataFrame({'Missing Values': missing_values, 'Percentage': missing_percentage}))

def plot_correlation_matrix(data):
    """Generate and save correlation matrix plot."""
    plt.figure(figsize=(10, 8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Correlation Matrix")
    plt.savefig("correlation_matrix.png")
    print("Correlation matrix saved.")

def detect_outliers(data):
    """Detect outliers using IQR method."""
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)))
    print("\nOutliers detected (True indicates outlier):")
    print(outliers)

def generate_histogram(data):
    """Generate histograms for each numerical column."""
    plt.figure(figsize=(10, 6))
    for column in data.select_dtypes(include=[np.number]).columns:
        plt.hist(data[column].dropna(), bins=20, alpha=0.7, label=column)
    plt.title("Histograms of Numerical Columns")
    plt.legend()
    plt.savefig("histograms.png")
    print("Histograms saved.")

def generate_box_plots(data):
    """Generate box plots for each numerical column."""
    plt.figure(figsize=(10, 6))
    for column in data.select_dtypes(include=[np.number]).columns:
        sns.boxplot(x=data[column], color='lightblue')
        plt.title(f"Boxplot of {column}")
        plt.savefig(f"{column}_boxplot.png")
        print(f"Boxplot for {column} saved.")

def generate_narrative(data, analysis_results):
    """Generate a markdown narrative for the analysis."""
    narrative = "# Data Analysis Report\n\n"
    narrative += "## Descriptive Statistics\n"
    narrative += data.describe().to_markdown() + "\n"
    
    narrative += "## LLM Insights\n"
    narrative += analysis_results + "\n"

    # Adding some basic visualizations as images
    narrative += "## Visualizations\n"
    narrative += "### Correlation Matrix\n![Correlation Matrix](correlation_matrix.png)\n"
    narrative += "### Histograms\n![Histograms](histograms.png)\n"

    return narrative

def save_results(narrative):
    """Save the narrative and visualizations to disk."""
    with open("analysis_report.md", "w") as f:
        f.write(narrative)
    print("Analysis report saved as 'analysis_report.md'.")

def run_analysis(file_path):
    """Main function to load, analyze, visualize, and generate report."""
    # Load data
    data = load_data(file_path)
    if data is None:
        return
    
    # Perform generic analysis
    summarize_statistics(data)
    count_missing_values(data)
    plot_correlation_matrix(data)
    detect_outliers(data)
    generate_histogram(data)
    generate_box_plots(data)

    # Send the data for analysis to the LLM
    analysis_results = analyze_with_llm(data)
    
    # Generate a comprehensive narrative
    narrative = generate_narrative(data, analysis_results)
    
    # Save everything
    save_results(narrative)

if __name__ == "__main__":
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py <csv_file_name>")
        sys.exit(1)

    # Get the CSV file name from the command-line argument
    csv_file_name = sys.argv[1]

    # Construct the full path to the dataset (assuming the file is in the current directory)
    dataset_path = os.path.join(os.getcwd(), csv_file_name)

    # Check if the dataset file exists
    if not os.path.isfile(dataset_path):
        print(f"Error: The file '{csv_file_name}' does not exist.")
        sys.exit(1)

    # Run the analysis with the dataset path
    run_analysis(dataset_path)
