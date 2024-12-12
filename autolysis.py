import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import openai
import numpy as np

# Set up the OpenAI API key from environment variable
openai.api_key = os.getenv("AIPROXY_TOKEN")

def analyze_data(df):
    """
    Perform basic analysis on the dataset: Summary statistics, missing values, correlations, and outliers.
    Returns a dictionary of analysis results.
    """
    analysis = {}

    # Summary statistics
    analysis['summary'] = df.describe(include='all')

    # Missing values
    analysis['missing_values'] = df.isnull().sum()

    # Correlation matrix
    analysis['correlation_matrix'] = df.corr()

    # Check for outliers using IQR
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).sum()
    analysis['outliers'] = outliers

    return analysis

def generate_visualizations(df, analysis):
    """
    Generate visualizations from the analysis: Correlation heatmap and histogram of numerical columns.
    """
    # Create a correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(analysis['correlation_matrix'], annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png', format='png')
    plt.close()

    # Create histograms for numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.savefig(f'{col}_distribution.png', format='png')
        plt.close()

def create_markdown_report(analysis, filename):
    """
    Create a README.md report with the analysis results, visualizations, and insights.
    """
    with open('README.md', 'w') as f:
        # Add title and description of the analysis
        f.write(f"# Automated Analysis of {filename}\n\n")
        f.write(f"## Summary Statistics\n")
        f.write(f"{analysis['summary']}\n\n")

        # Add missing values
        f.write(f"## Missing Values\n")
        f.write(f"{analysis['missing_values']}\n\n")

        # Add outlier summary
        f.write(f"## Outliers\n")
        f.write(f"{analysis['outliers']}\n\n")

        # Add visualizations
        f.write(f"## Visualizations\n")
        f.write(f"![Correlation Matrix](correlation_matrix.png)\n")
        for col in analysis['summary'].columns:
            if col in analysis['outliers'].index:
                f.write(f"![{col} Distribution]({col}_distribution.png)\n")

def get_llm_summary(df, analysis):
    """
    Use OpenAI API to generate a narrative summary of the analysis results.
    """
    # Prepare the prompt for the LLM
    prompt = f"""
    The dataset contains the following columns: {', '.join(df.columns)}
    The summary statistics are as follows:
    {analysis['summary']}

    There are missing values in the following columns:
    {analysis['missing_values']}

    The correlation matrix is:
    {analysis['correlation_matrix']}

    The outliers detected in the dataset are:
    {analysis['outliers']}

    Please summarize the key insights from this analysis and suggest potential actions or conclusions.
    """

    # Request the LLM to generate the summary
    response = openai.Completion.create(
        model="gpt-4o-mini",  # Use the correct model for this task
        prompt=prompt,
        max_tokens=500
    )

    return response.choices[0].text.strip()

def main(filename):
    """
    Main function to load the dataset, perform analysis, generate visualizations,
    create a Markdown report, and ask LLM to write a story about the data.
    """
    # Load the dataset
    df = pd.read_csv(filename)

    # Analyze the data
    analysis = analyze_data(df)

    # Generate visualizations
    generate_visualizations(df, analysis)

    # Create Markdown report
    create_markdown_report(analysis, filename)

    # Get the story from LLM
    story = get_llm_summary(df, analysis)

    # Append the LLM-generated story to the README.md
    with open('README.md', 'a') as f:
        f.write(f"\n## Insights and Implications\n")
        f.write(story)

    print("Analysis complete! Check the generated README.md and PNG files.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Please provide a CSV file to analyze.")
    else:
        filename = sys.argv[1]
        main(filename)
