import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import os
from datetime import datetime
from dashboard_v2 import (
    load_data, 
    create_plotly_progress_chart,
    create_project_summary,
    create_province_summary,
    create_gap_oa_analysis,
    create_cluster_milestone_analysis,
    plot_map
)

def get_safe_column_value(df, column_name, operation='count'):
    """Safely get column value, return N/A if column doesn't exist"""
    try:
        if column_name not in df.columns:
            return "N/A"
        if operation == 'count':
            return len(df[column_name].unique())
        elif operation == 'mean':
            return round(df[column_name].mean() * 100, 1)
        return "N/A"
    except:
        return "N/A"

def calculate_progress_rate(df):
    """Calculate overall progress rate"""
    return get_safe_column_value(df, 'Progress', 'mean')

def calculate_completion_status(df):
    """Calculate project completion status"""
    try:
        if 'Status' in df.columns:
            total = len(df)
            completed = len(df[df['Status'] == 'Completed'])
            return f"{completed}/{total} ({round(completed/total * 100, 1)}%)"
    except:
        pass
    return "N/A"

def generate_chart_content(chart):
    """Convert chart to appropriate format for presentation"""
    try:
        if isinstance(chart, go.Figure):
            # If it's a Plotly figure, convert to image
            img_bytes = chart.to_image(format="png", width=800, height=400)
            encoding = base64.b64encode(img_bytes).decode()
            return f'<img src="data:image/png;base64,{encoding}" alt="Chart">'
        elif isinstance(chart, str):
            # If it's HTML string, return as is
            return chart
        elif chart is None:
            return '<p>Chart not available</p>'
    except Exception as e:
        print(f"Error generating chart content: {str(e)}")
        return '<p>Error generating chart</p>'
    return '<p>Chart not available</p>'

def generate_chart_safely(func, *args, **kwargs):
    """Safely generate a chart, return None if there's an error"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"Error generating chart {func.__name__}: {str(e)}")
        return None

def main():
    # Get the script directory and file path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "Cellcard Project_Overall_Implementation_Plan(Sitewise)_V2.2_20250409.xlsx")
    
    if not os.path.exists(file_path):
        print(f"Error: Implementation plan file not found at {file_path}")
        return
    
    # Load available sheets
    try:
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
    except Exception as e:
        print(f"Error reading Excel file: {str(e)}")
        return
    
    if not sheet_names:
        print("Error: No sheets found in the Excel file")
        return
        
    # Use the first sheet by default
    selected_sheet = sheet_names[0]
    print(f"Using sheet: {selected_sheet}")
    
    # Load and process data
    try:
        df_forecast = load_data(file_path, selected_sheet)
        df_actual = load_data(file_path, selected_sheet)
        
        if df_forecast is None or df_actual is None:
            print("Error: Failed to load data from Excel file")
            return
        
        # Process forecast and actual data
        df_forecast = df_forecast.dropna(subset=["forecast oa date"])
        df_actual = df_actual.dropna(subset=["oa actual"])
        
        print("Generating charts...")
        
        # Generate charts safely
        progress_content = generate_chart_content(
            generate_chart_safely(create_plotly_progress_chart, df_forecast, df_actual, "Project Progress Overview")
        )
        
        summary_content = generate_chart_content(
            generate_chart_safely(create_project_summary, df_forecast, df_actual)
        )
        
        province_content = generate_chart_content(
            generate_chart_safely(create_province_summary, df_actual)
        )
        
        gap_content = generate_chart_content(
            generate_chart_safely(create_gap_oa_analysis, df_actual)
        )
        
        milestone_content = generate_chart_content(
            generate_chart_safely(create_cluster_milestone_analysis, df_actual)
        )
        
        print("Generating presentation...")
        
        # Get metrics safely
        total_provinces = get_safe_column_value(df_actual, 'Province')
        progress_rate = calculate_progress_rate(df_actual)
        completion_status = calculate_completion_status(df_actual)
        
        # Create presentation HTML with actual data
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Project Dashboard Analysis</title>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/reveal.min.css">
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/theme/white.min.css">
            <style>
                .reveal section img {{
                    border: none;
                    box-shadow: none;
                    max-height: 500px;
                }}
                .metric {{
                    font-size: 2em;
                    color: #2a76dd;
                }}
                .chart-container {{
                    margin: 20px 0;
                }}
                .na-text {{
                    color: #999;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="reveal">
                <div class="slides">
                    <!-- Title -->
                    <section>
                        <h2>Project Progress Dashboard</h2>
                        <h3>Key Metrics & Analysis</h3>
                        <p><small>Generated {datetime.now().strftime('%Y-%m-%d')}</small></p>
                        <p><small>Sheet: {selected_sheet}</small></p>
                    </section>

                    <!-- Progress Overview -->
                    <section>
                        <h2>Project Progress</h2>
                        <div class="chart-container">
                            {progress_content}
                        </div>
                        <p><small>Forecast vs Actual Progress Comparison</small></p>
                    </section>

                    <!-- Project Summary -->
                    <section>
                        <h2>Project Summary</h2>
                        <div class="chart-container">
                            {summary_content}
                        </div>
                    </section>

                    <!-- Provincial Analysis -->
                    <section>
                        <h2>Provincial Performance</h2>
                        <div class="chart-container">
                            {province_content}
                        </div>
                    </section>

                    <!-- Gap Analysis -->
                    <section>
                        <h2>Gap Analysis</h2>
                        <div class="chart-container">
                            {gap_content}
                        </div>
                    </section>

                    <!-- Milestone Analysis -->
                    <section>
                        <h2>Milestone Progress</h2>
                        <div class="chart-container">
                            {milestone_content}
                        </div>
                    </section>

                    <!-- Key Findings -->
                    <section>
                        <h2>Key Findings</h2>
                        <ul>
                            <li>Progress Rate: <span class="{'' if progress_rate != 'N/A' else 'na-text'}">{progress_rate}{'%' if progress_rate != 'N/A' else ''}</span></li>
                            <li>Total Provinces: <span class="{'' if total_provinces != 'N/A' else 'na-text'}">{total_provinces}</span></li>
                            <li>Completion Status: <span class="{'' if completion_status != 'N/A' else 'na-text'}">{completion_status}</span></li>
                        </ul>
                    </section>
                </div>
            </div>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/reveal.js/4.5.0/reveal.js"></script>
            <script>
                Reveal.initialize({{
                    controls: true,
                    progress: true,
                    center: true,
                    hash: true,
                    transition: 'slide',
                    slideNumber: true
                }});
            </script>
        </body>
        </html>
        """
        
        # Save the presentation
        output_file = "presentation_with_data.html"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        print(f"\nPresentation generated successfully!")
        print(f"Sheet used: {selected_sheet}")
        print(f"Output file: {os.path.abspath(output_file)}")
        print("\nTo view the presentation:")
        print("1. Open the file in a web browser")
        print("2. Use arrow keys to navigate")
        print("3. Press 'ESC' for overview")
        print("4. Press 'F' for fullscreen")
        
    except Exception as e:
        print(f"Error generating presentation: {str(e)}")

if __name__ == "__main__":
    main() 