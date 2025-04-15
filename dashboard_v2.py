import streamlit as st
import pandas as pd
import os
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import folium
import streamlit_folium as st_folium
import geopandas as gpd
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union
import alphashape
import base64
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
import tempfile
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pytz

# Set page configuration
st.set_page_config(
    page_title="Implementation Progress Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS to enhance dashboard appearance
st.markdown("""
    <style>
    .main {
        padding: 1rem 2rem;
        background-color: #f8f9fa;
    }
    .status-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
        border-left: 5px solid #0a62a9;
        transition: all 0.3s ease;
    }
    .status-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    .status-card h3 {
        color: #2E5077;
        margin-bottom: 1rem;
        font-size: 1.2rem;
        font-weight: 600;
    }
    .status-card hr {
        margin: 0.5rem 0 1rem 0;
        border: none;
        border-top: 2px solid #f0f0f0;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #2E5077;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0.5rem;
    }
    .stMetric {
        background-color: rgba(255,255,255,0.7);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: bold;
        color: #2E5077;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 1rem;
        color: #666;
    }
    .summary-section {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 5px solid #2E5077;
    }
    .summary-header {
        color: #2E5077;
        font-size: 1.4rem;
        font-weight: 600;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #f0f0f0;
    }
    .summary-content {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 1.5rem;
        margin-top: 1rem;
    }
    .summary-item {
        padding: 1rem;
        background-color: #f8f9fa;
        border-radius: 8px;
        border-left: 3px solid #2E5077;
    }
    .summary-item-header {
        font-weight: 600;
        color: #2E5077;
        margin-bottom: 0.5rem;
    }
    .chart-container {
        background-color: white;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def load_data(file_path, sheet_name):
    """Load and process data from Excel file"""
    try:
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        df.columns = df.columns.str.strip().str.lower()
        
        # Convert dates
        df["forecast oa date"] = pd.to_datetime(df["forecast oa date"], errors="coerce")
        df["oa actual"] = pd.to_datetime(df["oa actual"], errors="coerce")
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def create_plotly_progress_chart(df_forecast, df_actual, chart_title):
    """Create a professional and clean progress chart with value labels"""
    
    try:
        # Get the exact start and end dates from forecast
        start_date = df_forecast["forecast oa date"].min()
        end_date = df_forecast["forecast oa date"].max()
        
        # Process forecast data
        forecast_counts = df_forecast["forecast oa date"].value_counts().sort_index()
        forecast_cumulative = forecast_counts.cumsum()
        
        # Process actual data
        if not df_actual.empty:
            actual_counts = df_actual["oa actual"].value_counts().sort_index()
            actual_cumulative = actual_counts.cumsum()
        else:
            actual_cumulative = pd.Series()

        # Create the figure with increased size
        fig = go.Figure()
        
        # Professional color scheme
        colors = {
            'target': '#1f77b4',      # Professional blue
            'actual': '#2ca02c',      # Forest green
            'ahead': '#00CC96',       # Bright green for positive gap
            'behind': '#EF553B',      # Bright red for negative gap
            'grid': '#E9ECEF',        # Light gray
            'text': '#2F2F2F',        # Dark gray
            'background': '#FFFFFF',   # White
            'reference': '#7f7f7f'    # Medium gray
        }

        # Add forecast area with value labels
        fig.add_trace(
            go.Scatter(
                x=forecast_cumulative.index,
                y=forecast_cumulative.values,
                name=f"Target ({len(df_forecast):,} sites)",
                line=dict(
                    color=colors['target'],
                    width=2,
                    shape='spline',
                    smoothing=0.3
                ),
                mode="lines+markers+text",
                text=forecast_cumulative.values.astype(int),
                textposition="top center",
                textfont=dict(size=10, color=colors['target']),
                fill='tozeroy',
                fillcolor=f"rgba(31, 119, 180, 0.1)",
                hovertemplate=(
                    "<b>Target Progress</b><br>" +
                    "Date: %{x|%d %b %Y}<br>" +
                    "Sites: %{y:,.0f}<br>" +
                    "<extra></extra>"
                )
            )
        )
        
        # Add actual progress area with value labels
        if not df_actual.empty:
            completion_rate = (len(df_actual) / len(df_forecast) * 100)
            
            fig.add_trace(
                go.Scatter(
                    x=actual_cumulative.index,
                    y=actual_cumulative.values,
                    name=f"Completed ({len(df_actual):,} sites, {completion_rate:.1f}%)",
                    line=dict(
                        color=colors['actual'],
                        width=2,
                        shape='spline',
                        smoothing=0.3
                    ),
                    mode="lines+markers+text",
                    text=actual_cumulative.values.astype(int),
                    textposition="bottom center",
                    textfont=dict(size=10, color=colors['actual']),
                    fill='tozeroy',
                    fillcolor=f"rgba(44, 160, 44, 0.1)",
                    hovertemplate=(
                        "<b>Actual Progress</b><br>" +
                        "Date: %{x|%d %b %Y}<br>" +
                        "Sites: %{y:,.0f}<br>" +
                        "<extra></extra>"
                    )
                )
            )

            # Calculate gaps for each actual data point
            actual_dates = actual_cumulative.index
            gaps = []
            
            # Get all dates up to each actual date
            for date in actual_dates:
                # Get forecast target for this date
                forecast_mask = forecast_cumulative.index <= date
                if forecast_mask.any():
                    forecast_value = forecast_cumulative[forecast_mask].iloc[-1]
                else:
                    forecast_value = 0
                
                # Get actual completions up to this date
                actual_value = actual_cumulative[date]
                
                # Calculate gap
                gap = actual_value - forecast_value
                gaps.append(gap)
            
            # Create gap series
            gap_series = pd.Series(gaps, index=actual_dates)
            
            # Split gaps into positive and negative
            positive_gaps = gap_series.copy()
            negative_gaps = gap_series.copy()
            positive_gaps[positive_gaps <= 0] = None
            negative_gaps[negative_gaps > 0] = None
            
            # Add positive gap bars
            if not positive_gaps.isna().all():
                fig.add_trace(
                    go.Bar(
                        x=positive_gaps.index,
                        y=positive_gaps.values,
                        name="Ahead of Target",
                        text=[f"+{int(gap):,d}" if not pd.isna(gap) else "" for gap in positive_gaps],
                        textposition="outside",
                        marker=dict(
                            color=colors['ahead'],
                            opacity=0.7,
                            line=dict(color='rgba(0,0,0,0.1)', width=1)
                        ),
                        hovertemplate=(
                            "<b>Implementation Gap</b><br>" +
                            "Date: %{x|%d %b %Y}<br>" +
                            "Gap: %{y:+,.0f} sites<br>" +
                            "<extra></extra>"
                        ),
                        yaxis='y2'
                    )
                )
            
            # Add negative gap bars
            if not negative_gaps.isna().all():
                fig.add_trace(
                    go.Bar(
                        x=negative_gaps.index,
                        y=negative_gaps.values,
                        name="Behind Target",
                        text=[f"{int(gap):,d}" if not pd.isna(gap) else "" for gap in negative_gaps],
                        textposition="outside",
                        marker=dict(
                            color=colors['behind'],
                            opacity=0.7,
                            line=dict(color='rgba(0,0,0,0.1)', width=1)
                        ),
                        hovertemplate=(
                            "<b>Implementation Gap</b><br>" +
                            "Date: %{x|%d %b %Y}<br>" +
                            "Gap: %{y:+,.0f} sites<br>" +
                            "<extra></extra>"
                        ),
                        yaxis='y2'
                    )
                )

        # Update layout with professional styling and increased size
        fig.update_layout(
            title=dict(
                text=chart_title,
                font=dict(size=24, color=colors['text'], family="Arial"),
                x=0.5,
                xanchor="center",
                y=0.95,
                yanchor="top",
                pad=dict(b=20)
            ),
            template="none",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255, 255, 255, 0.95)",
                bordercolor=colors['grid'],
                borderwidth=1,
                font=dict(size=12, color=colors['text'])
            ),
            height=800,
            margin=dict(
                t=100,
                b=150,
                r=100,
                l=50,
                pad=10
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                tickformat="%d %b %Y",
                tickangle=45,
                tickfont=dict(size=12, color=colors['text']),
                title=None,
                zeroline=False,
                range=[start_date, end_date]
            ),
            yaxis=dict(
                title=dict(
                    text="Number of Sites",
                    font=dict(size=14, color=colors['text'])
                ),
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                tickfont=dict(size=12, color=colors['text']),
                zeroline=True,
                zerolinecolor=colors['grid'],
                zerolinewidth=2,
                tickformat=",d",
                rangemode="nonnegative"
            ),
            yaxis2=dict(
                title=dict(
                    text="Gap (Sites)",
                    font=dict(size=14, color=colors['text'])
                ),
                overlaying='y',
                side='right',
                showgrid=False,
                zeroline=True,
                zerolinecolor=colors['grid'],
                zerolinewidth=2,
                tickformat=",d",
                tickfont=dict(size=12, color=colors['text'])
            )
        )

        fig.update_layout(xaxis_rangeslider_visible=False)
        return fig
    except Exception as e:
        st.error(f"Error creating chart: {str(e)}")
        return None

def create_weekly_progress_chart(df_forecast, df_actual, chart_title):
    """Create a weekly progress chart with value labels"""
    try:
        # Get the exact start and end dates from forecast
        start_date = df_forecast["forecast oa date"].min()
        end_date = df_forecast["forecast oa date"].max()
        
        # Process forecast data by week
        df_forecast_copy = df_forecast.copy()
        df_forecast_copy['year_week'] = df_forecast_copy['forecast oa date'].dt.strftime('%Y-W%V')
        df_forecast_copy['week_num'] = df_forecast_copy['forecast oa date'].dt.isocalendar().week
        df_forecast_copy['year'] = df_forecast_copy['forecast oa date'].dt.isocalendar().year
        
        # Sort forecast data by date to ensure proper cumulative calculation
        df_forecast_copy = df_forecast_copy.sort_values('forecast oa date')
        
        # Create week labels with date ranges
        week_labels = {}
        week_dates = {}  # Store actual dates for each week
        for year_week, group in df_forecast_copy.groupby('year_week'):
            week_start = group['forecast oa date'].min().normalize()
            week_start = week_start - pd.Timedelta(days=week_start.weekday())
            week_end = week_start + pd.Timedelta(days=6)
            week_num = group['week_num'].iloc[0]
            week_labels[year_week] = f"WK{week_num} ({week_start.strftime('%d %b')}-{week_end.strftime('%d %b')})"
            week_dates[year_week] = week_start
        
        # Calculate cumulative forecast by week
        weekly_forecast = df_forecast_copy.groupby('year_week').size()
        weekly_forecast_cumsum = weekly_forecast.cumsum()
        
        # Initialize variables
        weekly_actual_cumsum = pd.Series()
        actual_indices = []
        actual_values = []
        actual_weeks = []
        gaps = []
        positive_gaps = []
        negative_gaps = []
        
        # Process actual data by week
        if not df_actual.empty:
            df_actual_copy = df_actual.copy()
            df_actual_copy['year_week'] = df_actual_copy['oa actual'].dt.strftime('%Y-W%V')
            df_actual_copy = df_actual_copy.sort_values('oa actual')  # Sort by actual date
            weekly_actual = df_actual_copy.groupby('year_week').size()
            weekly_actual_cumsum = weekly_actual.cumsum()
            
            # Map actual weeks to forecast week indices
            week_to_index = {week: idx for idx, week in enumerate(weekly_forecast_cumsum.index)}
            
            for week in weekly_actual_cumsum.index:
                if week in week_to_index:
                    actual_indices.append(week_to_index[week])
                    actual_values.append(weekly_actual_cumsum[week])
                    actual_weeks.append(week)
            
            # Calculate weekly gaps
            for week in actual_weeks:
                week_idx = week_to_index[week]
                # Get all weeks up to current week
                forecast_weeks = [w for w in weekly_forecast_cumsum.index if week_dates[w] <= week_dates[week]]
                if forecast_weeks:
                    # Get forecast target for this week
                    forecast_value = weekly_forecast_cumsum[forecast_weeks[-1]]
                    # Get actual completions up to this week
                    actual_value = weekly_actual_cumsum[week]
                    # Calculate gap
                    gap = actual_value - forecast_value
                    gaps.append((week_idx, gap, week))
            
            # Split gaps into positive and negative
            positive_gaps = [(idx, gap, week) for idx, gap, week in gaps if gap > 0]
            negative_gaps = [(idx, gap, week) for idx, gap, week in gaps if gap < 0]

        # Calculate y-axis ranges for better scaling
        y_max_main = max(
            max(weekly_forecast_cumsum.values) if not weekly_forecast_cumsum.empty else 0,
            max(weekly_actual_cumsum.values) if not weekly_actual_cumsum.empty else 0
        )

        # Calculate gap range
        if gaps:
            gap_values = [gap for _, gap, _ in gaps]
            max_gap = max(gap_values)
            min_gap = min(gap_values)
            gap_abs_max = max(abs(min_gap), abs(max_gap))
            gap_range = [-gap_abs_max * 1.2, gap_abs_max * 1.2]
        else:
            gap_abs_max = y_max_main * 0.2
            gap_range = [-gap_abs_max, gap_abs_max]

        # Create the figure
        fig = go.Figure()
        
        # Professional color scheme
        colors = {
            'target': '#1f77b4',
            'actual': '#2ca02c',
            'ahead': '#00CC96',
            'behind': '#EF553B',
            'grid': '#E9ECEF',
            'text': '#2F2F2F',
            'background': '#FFFFFF'
        }

        # Add forecast line
        x_forecast = list(range(len(weekly_forecast_cumsum)))
        fig.add_trace(
            go.Scatter(
                x=x_forecast,
                y=weekly_forecast_cumsum.values,
                name=f"Target ({len(df_forecast):,} sites)",
                line=dict(color=colors['target'], width=2.5),
                mode="lines+markers+text",
                marker=dict(size=8),
                text=[f"{int(val):,}" for val in weekly_forecast_cumsum.values],
                textposition="top center",
                textfont=dict(size=10, color=colors['target']),
                fill='tozeroy',
                fillcolor=f"rgba(31, 119, 180, 0.1)",
                hovertemplate="%{customdata}<br>Target: %{y:,.0f} sites<extra></extra>",
                customdata=[week_labels[w] for w in weekly_forecast_cumsum.index]
            )
        )
        
        if not df_actual.empty:
            # Map actual weeks to forecast week indices
            week_to_index = {week: idx for idx, week in enumerate(weekly_forecast_cumsum.index)}
            actual_indices = []
            actual_values = []
            actual_weeks = []
            
            for week in weekly_actual_cumsum.index:
                if week in week_to_index:
                    actual_indices.append(week_to_index[week])
                    actual_values.append(weekly_actual_cumsum[week])
                    actual_weeks.append(week)
            
            # Add actual line
            fig.add_trace(
                go.Scatter(
                    x=actual_indices,
                    y=actual_values,
                    name=f"Completed ({len(df_actual):,} sites)",
                    line=dict(color=colors['actual'], width=2.5),
                    mode="lines+markers+text",
                    marker=dict(size=8),
                    text=[f"{int(val):,}" for val in actual_values],
                    textposition="bottom center",
                    textfont=dict(size=10, color=colors['actual']),
                    fill='tozeroy',
                    fillcolor=f"rgba(44, 160, 44, 0.1)",
                    hovertemplate="%{customdata}<br>Completed: %{y:,.0f} sites<extra></extra>",
                    customdata=[week_labels[w] for w in actual_weeks]
                )
            )

            # Add gap bars with improved text display
            if positive_gaps:
                x_vals, y_vals, weeks = zip(*positive_gaps)
                fig.add_trace(
                    go.Bar(
                        x=list(x_vals),
                        y=list(y_vals),
                        name="Ahead of Target",
                        text=[f"+{int(gap):,}" for gap in y_vals],
                        textposition="outside",
                        textfont=dict(size=11, color=colors['ahead']),
                        marker=dict(
                            color=colors['ahead'],
                            opacity=0.7,
                            line=dict(color='rgba(0,0,0,0.1)', width=1)
                        ),
                        hovertemplate="%{customdata}<br>Ahead by: %{y:+,.0f} sites<extra></extra>",
                        customdata=[week_labels[w] for w in weeks],
                        yaxis='y2',
                        width=0.5,
                        showlegend=True
                    )
                )
            
            if negative_gaps:
                x_vals, y_vals, weeks = zip(*negative_gaps)
                fig.add_trace(
                    go.Bar(
                        x=list(x_vals),
                        y=list(y_vals),
                        name="Behind Target",
                        text=[f"{int(gap):,}" for gap in y_vals],
                        textposition="outside",
                        textfont=dict(size=11, color=colors['behind']),
                        marker=dict(
                            color=colors['behind'],
                            opacity=0.7,
                            line=dict(color='rgba(0,0,0,0.1)', width=1)
                        ),
                        hovertemplate="%{customdata}<br>Behind by: %{y:,.0f} sites<extra></extra>",
                        customdata=[week_labels[w] for w in weeks],
                        yaxis='y2',
                        width=0.5,
                        showlegend=True
                    )
                )
        
        # Update layout for better spacing and readability
        fig.update_layout(
            title=dict(
                text=chart_title,
                font=dict(size=24, color=colors['text'], family="Arial"),
                x=0.5,
                xanchor="center",
                y=0.95,
                yanchor="top"
            ),
            template="none",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            hovermode="x unified",
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor=colors['grid']
            ),
            height=800,
            margin=dict(
                t=100,
                b=150,
                r=120,
                l=80,
                pad=10
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=colors['grid'],
                ticktext=[week_labels[w] for w in weekly_forecast_cumsum.index],
                tickvals=list(range(len(weekly_forecast_cumsum))),
                tickangle=45,
                tickfont=dict(size=10),
                title="Project Timeline by Week",
                type='category',
                tickmode='array',
                nticks=20
            ),
            yaxis=dict(
                title=dict(
                    text="Number of Sites",
                    font=dict(size=12)
                ),
                showgrid=True,
                gridcolor=colors['grid'],
                tickfont=dict(size=11),
                tickformat=",d",
                rangemode="nonnegative",
                range=[0, y_max_main * 1.1],
                side='left'
            ),
            yaxis2=dict(
                title=dict(
                    text="Gap (Sites)",
                    font=dict(size=12)
                ),
                overlaying='y',
                side='right',
                showgrid=False,
                zeroline=True,
                zerolinecolor=colors['grid'],
                zerolinewidth=1,
                tickformat=",d",
                tickfont=dict(size=11),
                range=gap_range,
                dtick=max(1, int(gap_abs_max/4))
            ),
            bargap=0.4
        )
        
        # Update x-axis grid
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor=colors['grid'],
            griddash='solid'
        )
        
        return fig
    except Exception as e:
        st.error(f"Error creating weekly chart: {str(e)}")
        return None

def create_project_summary(df_forecast, df_actual):
    """Create a project summary section with unique insights"""
    if df_actual.empty:
        return "No implementation data available"

    # Calculate key metrics
    total_sites = len(df_forecast)
    completed_sites = len(df_actual)
    
    # Calculate weekly trends - Create explicit copy to avoid SettingWithCopyWarning
    df_actual_copy = df_actual.copy()
    df_actual_copy.loc[:, 'week'] = df_actual_copy["oa actual"].dt.isocalendar().week
    df_actual_copy.loc[:, 'year'] = df_actual_copy["oa actual"].dt.isocalendar().year
    weekly_completions = df_actual_copy.groupby(['year', 'week']).size()
    avg_weekly_rate = int(round(weekly_completions.mean()))
    max_weekly_rate = int(weekly_completions.max())
    min_weekly_rate = int(weekly_completions.min())
    
    # Calculate current week's progress
    current_date = datetime.now()
    current_week_start = current_date - pd.Timedelta(days=current_date.weekday())
    current_week_end = current_week_start + pd.Timedelta(days=7)
    last_week_start = current_week_start - pd.Timedelta(weeks=1)
    
    # Get weekly forecast and actual counts
    last_week_forecast = len(df_forecast[
        (df_forecast["forecast oa date"] >= last_week_start) & 
        (df_forecast["forecast oa date"] < current_week_start)
    ])
    last_week_actual = len(df_actual[
        (df_actual["oa actual"] >= last_week_start) & 
        (df_actual["oa actual"] < current_week_start)
    ])
    
    current_week_forecast = len(df_forecast[
        (df_forecast["forecast oa date"] >= current_week_start) & 
        (df_forecast["forecast oa date"] <= current_date)
    ])
    current_week_actual = len(df_actual[
        (df_actual["oa actual"] >= current_week_start) & 
        (df_actual["oa actual"] <= current_date)
    ])
    
    # Calculate weekly performance
    last_week_performance = (last_week_actual / last_week_forecast * 100) if last_week_forecast > 0 else 100
    current_week_performance = (current_week_actual / current_week_forecast * 100) if current_week_forecast > 0 else 100
    
    # Format week dates for display
    last_week_dates = f"{last_week_start.strftime('%d %b')} - {(current_week_start - pd.Timedelta(days=1)).strftime('%d %b')}"
    current_week_dates = f"{current_week_start.strftime('%d %b')} - {current_date.strftime('%d %b')}"

    summary_html = f"""
    <div class="summary-section">
        <div class="summary-header">📊 Weekly Performance Insights</div>
        <div class="summary-content">
            <div class="summary-item">
                <div class="summary-item-header">Weekly Implementation Trends</div>
                <div>Average Weekly: {avg_weekly_rate:,} sites</div>
                <div>Best Week: {max_weekly_rate:,} sites</div>
                <div>Slowest Week: {min_weekly_rate:,} sites</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-header">Last Week Performance</div>
                <div>Period: {last_week_dates}</div>
                <div>Completed: {last_week_actual:,} of {last_week_forecast:,} sites</div>
                <div style="color: {'#28A745' if last_week_performance >= 100 else '#DC3545'}">
                    Achievement: {last_week_performance:.1f}%
                </div>
            </div>
            <div class="summary-item">
                <div class="summary-item-header">Current Week Progress</div>
                <div>Period: {current_week_dates}</div>
                <div>Completed: {current_week_actual:,} of {current_week_forecast:,} sites</div>
                <div style="color: {'#28A745' if current_week_performance >= 100 else '#DC3545'}">
                    Progress: {current_week_performance:.1f}%
                </div>
            </div>
        </div>
    </div>
    """
    return summary_html

def create_province_summary(df_actual):
    """Create a summary of completed sites by province"""
    if df_actual.empty or 'province' not in df_actual.columns:
        return ""

    province_counts = df_actual['province'].value_counts()
    
    province_html = """
    <div class="summary-section" style="margin-top: 1rem;">
        <div class="summary-header">📍 Completed Sites by Province</div>
        <div class="summary-content" style="display: block;">
            <table style="width: 100%; border-collapse: collapse;">
                <tr>
                    <th style="text-align: left; padding: 8px; border-bottom: 2px solid #eee;">Province</th>
                    <th style="text-align: right; padding: 8px; border-bottom: 2px solid #eee;">Completed Sites</th>
                </tr>
    """
    
    for province, count in province_counts.items():
        province_html += f"""
                <tr>
                    <td style="text-align: left; padding: 8px; border-bottom: 1px solid #eee;">{province}</td>
                    <td style="text-align: right; padding: 8px; border-bottom: 1px solid #eee;">{count:,}</td>
                </tr>
        """
    
    province_html += """
            </table>
        </div>
    </div>
    """
    return province_html

def load_shapefile():
    """Load Cambodia province shapefile"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        province_dir = os.path.join(script_dir, 'province')
        
        if os.path.exists(province_dir):
            shapefiles = [f for f in os.listdir(province_dir) if f.endswith('.shp')]
            if shapefiles:
                shapefile_path = os.path.join(province_dir, shapefiles[0])
                cambodia_provinces_gdf = gpd.read_file(shapefile_path)
                
                if cambodia_provinces_gdf.crs is None:
                    cambodia_provinces_gdf.set_crs(epsg=4326, inplace=True)
                else:
                    cambodia_provinces_gdf = cambodia_provinces_gdf.to_crs(epsg=4326)
                
                return cambodia_provinces_gdf
        return None
    except Exception as e:
        return None

def plot_map(df, cambodia_provinces_gdf=None, sheet_name=None):
    """Create Folium map with clusters and sophisticated visualization"""
    try:
        # Make a copy and clean the data
        df = df.copy()
        
        # Convert coordinates to numeric, removing any invalid values
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        
        # Remove rows with invalid coordinates
        valid_coords = df.dropna(subset=['lat', 'lon'])
        if len(valid_coords) < len(df):
            st.warning(f"Removed {len(df) - len(valid_coords)} rows with invalid coordinates")
        df = valid_coords

        # Create base map centered on Cambodia
        map_center = [12.5657, 104.9910]
        map_clusters = folium.Map(
            location=map_center,
            zoom_start=7,
            tiles='CartoDB positron'
        )

        # Add province boundaries if available
        if cambodia_provinces_gdf is not None and not cambodia_provinces_gdf.empty:
            try:
                geojson_data = cambodia_provinces_gdf.__geo_interface__
                folium.GeoJson(
                    geojson_data,
                    name='Provinces',
                    style_function=lambda x: {
                        'fillColor': 'green',
                        'color': 'black',
                        'weight': 2,
                        'fillOpacity': 0.1
                    },
                    highlight_function=lambda x: {
                        'weight': 3,
                        'fillColor': 'yellow'
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=['HRName'],
                        aliases=['Province:'],
                        localize=True,
                        sticky=True,
                        labels=True
                    )
                ).add_to(map_clusters)
            except Exception as e:
                pass

        # Determine vendor name and map type based on sheet name
        if sheet_name:
            if "349_NOKIA_SWAP" in sheet_name:
                vendor_name = "Nokia Swap"
                use_clusters = True
            elif "185_ALU&HW_SWAP" in sheet_name:
                vendor_name = "ALU & HW Swap"
                use_clusters = True
            elif "153_ZTE_UPGRADE" in sheet_name:
                vendor_name = "ZTE Upgrade"
                use_clusters = False
            elif "20_HUAWEI_REDEPLOY" in sheet_name:
                vendor_name = "Huawei Redeploy"
                use_clusters = False
            elif "BTB-NEWSITE" in sheet_name:
                vendor_name = "BTB New Sites"
                use_clusters = False
            else:
                vendor_name = "Sites"
                use_clusters = True
        else:
            vendor_name = "Sites"
            use_clusters = True

        # Get script directory for icon path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        flag_icon_path = os.path.join(script_dir, 'ZTE Flag.png')
        
        # Convert flag image to base64 for reliable display in HTML
        flag_icon_base64 = None
        if os.path.exists(flag_icon_path):
            with open(flag_icon_path, "rb") as image_file:
                flag_icon_base64 = base64.b64encode(image_file.read()).decode('utf-8')

        # Create marker cluster groups
        completed_sites = folium.FeatureGroup(name="Completed Sites")
        pending_sites = folium.FeatureGroup(name="Pending Sites")
        if use_clusters:
            cluster_boundaries = folium.FeatureGroup(name="Cluster Boundaries")

        # Add site markers with appropriate icons
        flag_markers_added = 0
        circle_markers_added = 0

        if use_clusters:
            # Color palette for clusters
            distinct_colors = [
                "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", 
                "#ffff33", "#a65628", "#f781bf", "#999999", "#66c2a5", 
                "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"
            ]
            
            # Clean and prepare cluster data
            df['cluster_id'] = df['cluster_id'].fillna('Unassigned')
            df['swap_batch'] = df['swap_batch'].fillna('Unassigned')
            df.loc[df['cluster_id'] == 'nan', 'cluster_id'] = 'Unassigned'
            
            # Assign colors to clusters
            unique_clusters = df['cluster_id'].unique()
            cluster_colors = {cluster_id: distinct_colors[i % len(distinct_colors)] 
                            for i, cluster_id in enumerate(unique_clusters)}

            # Create cluster boundary polygons
            for cluster_id in unique_clusters:
                if cluster_id != 'Unassigned':
                    try:
                        # Get cluster data
                        cluster_df = df[df['cluster_id'] == cluster_id]
                        cluster_points = cluster_df[['lat', 'lon']].values
                        
                        if len(cluster_points) >= 3:  # Need at least 3 points for a polygon
                            # Convert points to the format required by alphashape
                            points = [(float(lon), float(lat)) for lat, lon in cluster_points]
                            
                            # Generate a concave hull (alpha shape) around the points
                            alpha_shape = alphashape.alphashape(points, 0.5)
                            
                            if alpha_shape:
                                # Add the cluster boundary
                                folium.GeoJson(
                                    alpha_shape.__geo_interface__,
                                    name=f'Cluster {cluster_id}',
                                    style_function=lambda x: {
                                        'fillColor': cluster_colors[cluster_id],
                                        'color': cluster_colors[cluster_id],
                                        'weight': 2,
                                        'fillOpacity': 0.2
                                    }
                                ).add_to(cluster_boundaries)
                                
                                # Calculate center for label
                                center_lat = cluster_df['lat'].mean()
                                center_lon = cluster_df['lon'].mean()
                                
                                # Add cluster label with improved styling for better text fitting
                                label_html = f'''
                                    <div style="
                                        display: inline-block;
                                        background-color: {cluster_colors[cluster_id]};
                                        color: white;
                                        border: 2px solid white;
                                        border-radius: 4px;
                                        padding: 3px 8px;
                                        font-size: 12px;
                                        font-weight: bold;
                                        box-shadow: 2px 2px 4px rgba(0,0,0,0.4);
                                        white-space: nowrap;
                                        text-align: center;
                                        min-width: max-content;
                                        position: absolute;
                                        left: 50%;
                                        top: 50%;
                                        transform: translate(-50%, -50%);
                                    ">{cluster_id}</div>
                                '''
                                
                                folium.Marker(
                                    [center_lat, center_lon],
                                    icon=folium.DivIcon(
                                        html=label_html,
                                        icon_size=(0, 0)
                                    )
                                ).add_to(cluster_boundaries)
                                
                    except Exception as e:
                        continue

        # Add site markers for all projects (both swap and non-swap)
        for _, row in df.iterrows():
            try:
                lat = float(row['lat'])
                lon = float(row['lon'])
                
                if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
                    continue
                
                # Prepare popup text
                popup_text = f"""
                <div style='min-width: 200px'>
                    <b>Site Details:</b><br>
                    Site Name: {row['sitename']}<br>
                """
                
                # Add cluster info only for swap projects
                if use_clusters:
                    cluster_id = str(row['cluster_id'])
                    color = cluster_colors.get(cluster_id, 'black')
                    popup_text += f"""
                    Cluster: {cluster_id}<br>
                    Swap Batch: {row['swap_batch']}<br>
                    """
                else:
                    color = '#3388ff'  # Default blue color for non-swap projects
                
                popup_text += f"""
                    {'OA Actual: ' + str(row['oa actual']) if pd.notna(row['oa actual']) else ''}
                </div>
                """
                
                if pd.notna(row['oa actual']) and flag_icon_base64:
                    flag_icon_url = f'data:image/png;base64,{flag_icon_base64}'
                    flag_icon = folium.features.CustomIcon(
                        flag_icon_url,
                        icon_size=(30, 30),
                        icon_anchor=(15, 15)
                    )
                    folium.Marker(
                        location=(lat, lon),
                        popup=folium.Popup(popup_text, max_width=300),
                        icon=flag_icon
                    ).add_to(completed_sites)
                    flag_markers_added += 1
                else:
                    folium.CircleMarker(
                        location=(lat, lon),
                        radius=5,
                        popup=folium.Popup(popup_text, max_width=300),
                        color=color,
                        fill=True,
                        fill_opacity=0.7
                    ).add_to(pending_sites)
                    circle_markers_added += 1
            except Exception as e:
                continue

        # Add marker groups to map
        if use_clusters:
            cluster_boundaries.add_to(map_clusters)
        completed_sites.add_to(map_clusters)
        pending_sites.add_to(map_clusters)

        # Create legend content
        legend_html = f'''
        <div style="
            position: fixed; 
            bottom: 50px; 
            right: 10px; 
            width: 300px;
            max-height: 500px;
            overflow-y: auto;
            background-color: white;
            border: 2px solid gray;
            z-index: 1000;
            padding: 15px;
            font-size: 14px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        ">
            <h4 style="margin-top:0; margin-bottom:10px; color:#2E5077;">{vendor_name} Project Legend</h4>
        '''

        # Add flag icon section if available
        if flag_markers_added > 0 and flag_icon_base64:
            legend_html += f'''
            <div style="margin-bottom:15px; padding-bottom:10px; border-bottom:1px solid #eee;">
                <img src="data:image/png;base64,{flag_icon_base64}" 
                     style="width:20px; height:20px; vertical-align:middle;">
                <span style="margin-left:5px; font-weight:500;">Completed Sites ({flag_markers_added})</span>
            </div>
            '''

        if use_clusters:
            # Add cluster information for swap projects
            cluster_summary = df.groupby(['swap_batch', 'cluster_id']).size().reset_index(name='Count')
            
            # Calculate total sites per batch
            batch_totals = cluster_summary.groupby('swap_batch')['Count'].sum()
            
            # Sort batches numerically (assuming batch names can be converted to numbers)
            try:
                sorted_batches = sorted(batch_totals.index, 
                                      key=lambda x: float(x) if x != 'Unassigned' else float('inf'))
            except:
                sorted_batches = sorted(batch_totals.index)

            # Add batch summary section
            legend_html += '<div style="margin-bottom:15px;"><b>Batch Summary:</b></div>'
            
            for batch in sorted_batches:
                batch_name = f"Batch {batch}" if batch != 'Unassigned' else "Unassigned"
                total_sites = batch_totals[batch]
                completed_sites = len(df[(df['swap_batch'] == batch) & (df['oa actual'].notna())])
                completion_rate = (completed_sites / total_sites * 100) if total_sites > 0 else 0
                
                legend_html += f'''
                <div style="margin-bottom:15px; padding:8px; background-color:#f8f9fa; border-radius:4px;">
                    <div style="font-weight:600; color:#2E5077; margin-bottom:5px;">
                        {batch_name} ({total_sites:,} sites)
                    </div>
                    <div style="font-size:12px; color:#666;">
                        Completed: {completed_sites:,} ({completion_rate:.1f}%)
                    </div>
                    <div style="margin-top:8px;">
                '''
                
                # Add clusters for this batch
                batch_clusters = cluster_summary[cluster_summary['swap_batch'] == batch]
                for _, row in batch_clusters.iterrows():
                    cluster_id = row['cluster_id']
                    count = row['Count']
                    color = cluster_colors.get(cluster_id, 'black')
                    cluster_completed = len(df[(df['cluster_id'] == cluster_id) & (df['oa actual'].notna())])
                    cluster_rate = (cluster_completed / count * 100) if count > 0 else 0
                    
                    legend_html += f'''
                    <div style="display:flex; align-items:center; margin:5px 0; font-size:12px;">
                        <span style="display:inline-block; width:12px; height:12px; 
                               background-color:{color}; border-radius:50%; margin-right:8px;"></span>
                        <span style="flex-grow:1;">{cluster_id}</span>
                        <span style="margin-left:5px;">
                            {count:,} sites ({cluster_rate:.1f}%)
                        </span>
                    </div>
                    '''
                
                legend_html += '</div></div>'
        else:
            # Simple legend for upgrade/redeploy/newsite projects
            legend_html += f'''
            <div style="margin:10px 0;">
                <div style="display:flex; align-items:center; margin-bottom:10px;">
                    <span style="display:inline-block; width:12px; height:12px; 
                           background-color:blue; border-radius:50%; margin-right:8px;"></span>
                    <span>Pending Sites ({circle_markers_added})</span>
                </div>
            </div>
            '''

        legend_html += f'''
            <div style="margin-top:15px; padding-top:10px; border-top:1px solid #ddd;">
                <div style="font-weight:600; color:#2E5077;">Project Summary:</div>
                <div style="margin-top:5px;">Total Sites: {len(df):,}</div>
                <div>Completed: {flag_markers_added:,} ({(flag_markers_added/len(df)*100):.1f}%)</div>
                <div>Pending: {circle_markers_added:,} ({(circle_markers_added/len(df)*100):.1f}%)</div>
            </div>
        </div>
        '''

        # Add the legend to the map
        map_clusters.get_root().html.add_child(folium.Element(legend_html))

        # Add layer control
        folium.LayerControl(
            position='topright',
            collapsed=False
        ).add_to(map_clusters)

        return map_clusters
    except Exception as e:
        st.error(f"Error in plot_map: {e}")
        return None

def register_fonts():
    """Register Calibri fonts if available, otherwise use fallback fonts"""
    try:
        # Windows system font paths
        windows_font_path = "C:/Windows/Fonts/"
        calibri_regular = os.path.join(windows_font_path, "calibri.ttf")
        calibri_bold = os.path.join(windows_font_path, "calibrib.ttf")
        
        # Register Calibri fonts if available
        if os.path.exists(calibri_regular) and os.path.exists(calibri_bold):
            pdfmetrics.registerFont(TTFont('Calibri', calibri_regular))
            pdfmetrics.registerFont(TTFont('Calibri-Bold', calibri_bold))
            return True
        return False
    except Exception as e:
        st.error(f"Error registering fonts: {str(e)}")
        return False

def capture_map_for_pdf(df, provinces_gdf, selected_sheet):
    """Generate a static map visualization for the PDF report using plotly mapbox"""
    try:
        st.write("Debug: Starting map generation...")
        
        # Verify and clean coordinates
        df = df.copy()
        df['lat'] = pd.to_numeric(df['lat'], errors='coerce')
        df['lon'] = pd.to_numeric(df['lon'], errors='coerce')
        
        # Drop rows with invalid coordinates
        valid_coords = df.dropna(subset=['lat', 'lon'])
        if len(valid_coords) == 0:
            st.error("No valid coordinates found in the data")
            return None
            
        st.write(f"Debug: Found {len(valid_coords)} sites with valid coordinates")
        
        # Create figure with mapbox
        fig = go.Figure()

        # Add province boundaries if available
        if provinces_gdf is not None and not provinces_gdf.empty:
            try:
                for idx, row in provinces_gdf.iterrows():
                    if row.geometry:
                        if row.geometry.type == 'MultiPolygon':
                            for polygon in row.geometry.geoms:
                                x, y = polygon.exterior.xy
                                # Convert array.array to list
                                x_coords = [float(coord) for coord in x]
                                y_coords = [float(coord) for coord in y]
                                fig.add_trace(go.Scattermapbox(
                                    lon=x_coords,
                                    lat=y_coords,
                                    mode='lines',
                                    name=row['HRName'] if 'HRName' in row else f'Province {idx}',
                                    line=dict(color='gray', width=1),
                                    fill='toself',
                                    fillcolor='rgba(128, 128, 128, 0.1)',
                                    hoverinfo='name',
                                    showlegend=False
                                ))
                        elif row.geometry.type == 'Polygon':
                            x, y = row.geometry.exterior.xy
                            # Convert array.array to list
                            x_coords = [float(coord) for coord in x]
                            y_coords = [float(coord) for coord in y]
                            fig.add_trace(go.Scattermapbox(
                                lon=x_coords,
                                lat=y_coords,
                                mode='lines',
                                name=row['HRName'] if 'HRName' in row else f'Province {idx}',
                                line=dict(color='gray', width=1),
                                fill='toself',
                                fillcolor='rgba(128, 128, 128, 0.1)',
                                hoverinfo='name',
                                showlegend=False
                            ))
                st.write("Debug: Province boundaries plotted successfully")
            except Exception as e:
                st.error(f"Error plotting province boundaries: {str(e)}")
                import traceback
                st.error(f"Detailed error: {traceback.format_exc()}")
        
        # Plot completed sites
        completed_mask = valid_coords['oa actual'].notna()
        if completed_mask.any():
            completed_sites = valid_coords[completed_mask]
            fig.add_trace(go.Scattermapbox(
                lon=completed_sites['lon'],
                lat=completed_sites['lat'],
                mode='markers',
                name='Completed Sites',
                marker=dict(
                    size=10,
                    color='green',
                    opacity=0.7
                ),
                text=completed_sites['sitename'],
                hovertemplate="<b>%{text}</b><br>Status: Completed<br>Lon: %{lon:.4f}<br>Lat: %{lat:.4f}<extra></extra>"
            ))
        
        # Plot pending sites
        pending_mask = ~completed_mask
        if pending_mask.any():
            pending_sites = valid_coords[pending_mask]
            fig.add_trace(go.Scattermapbox(
                lon=pending_sites['lon'],
                lat=pending_sites['lat'],
                mode='markers',
                name='Pending Sites',
                marker=dict(
                    size=10,
                    color='red',
                    opacity=0.7
                ),
                text=pending_sites['sitename'],
                hovertemplate="<b>%{text}</b><br>Status: Pending<br>Lon: %{lon:.4f}<br>Lat: %{lat:.4f}<extra></extra>"
            ))

        # Get Mapbox token from environment variable or use empty string for basic map
        mapbox_token = os.getenv('MAPBOX_TOKEN', '')
        
        # Update layout with mapbox style
        fig.update_layout(
            title=dict(
                text=f'Implementation Map - {selected_sheet}',
                x=0.5,
                xanchor='center',
                font=dict(size=16)
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255,255,255,0.8)'
            ),
            width=1200,
            height=800,
            margin=dict(l=0, r=0, t=50, b=0),
            mapbox=dict(
                style='carto-positron',  # Use Carto style which is free and doesn't require token
                center=dict(lat=12.5657, lon=104.9910),  # Center on Cambodia
                zoom=6,
                bearing=0,
                pitch=0
            ),
            paper_bgcolor='white'
        )
        
        # Convert to image
        try:
            img_bytes = fig.to_image(format="png", width=1200, height=800, scale=2)
            buf = BytesIO(img_bytes)
            st.write("Debug: Map saved to buffer successfully")
            return buf
        except Exception as e:
            st.error(f"Error converting to image: {str(e)}")
            # Try alternative method using static HTML
            try:
                st.write("Debug: Attempting alternative image conversion method...")
                import plotly.io as pio
                img_bytes = pio.to_image(fig, format="png", width=1200, height=800, scale=2)
                buf = BytesIO(img_bytes)
                st.write("Debug: Alternative conversion successful")
                return buf
            except Exception as e2:
                st.error(f"Error in alternative conversion: {str(e2)}")
                return None
            
    except Exception as e:
        st.error(f"Error in map generation: {str(e)}")
        import traceback
        st.error(f"Detailed error: {traceback.format_exc()}")
        return None

def convert_plotly_to_image(fig):
    """Convert Plotly figure to image using a more cloud-friendly approach"""
    try:
        # First try kaleido
        try:
            import kaleido
            img_bytes = fig.to_image(format="png", width=1200, height=600, scale=2)
            return BytesIO(img_bytes)
        except Exception as e:
            st.write(f"Debug: Kaleido conversion failed, trying alternative method: {str(e)}")
            
        # If kaleido fails, try alternative method using static HTML
        import selenium
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import time
        
        # Configure headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        
        # Create a temporary HTML file with the plot
        html_str = fig.to_html(include_plotlyjs=True, full_html=True)
        
        # Save to a temporary file
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as f:
            f.write(html_str)
            temp_path = f.name
        
        # Use selenium to render and capture the plot
        driver = webdriver.Chrome(options=chrome_options)
        driver.get(f"file://{temp_path}")
        time.sleep(2)  # Wait for plot to render
        
        # Capture screenshot
        img_data = driver.get_screenshot_as_png()
        driver.quit()
        
        # Clean up
        import os
        os.unlink(temp_path)
        
        return BytesIO(img_data)
        
    except Exception as e:
        st.error(f"Error converting chart to image: {str(e)}")
        return None

def generate_simple_pdf_report(df, provinces_gdf, selected_sheet):
    """Generate a comprehensive PDF report with all dashboard metrics"""
    try:
        st.write("Debug: Starting PDF generation...")
        
        # Create PDF document with wider margins
        buffer = BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        
        # Initialize story and styles
        story = []
        styles = getSampleStyleSheet()
        
        # Create custom styles with standard fonts
        styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=styles['Title'],
            fontName='Helvetica-Bold',
            fontSize=24,
            spaceAfter=30,
            alignment=1  # Center alignment
        ))
        
        styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=styles['Heading1'],
            fontName='Helvetica-Bold',
            fontSize=16,
            spaceAfter=12,
            textColor=colors.HexColor('#2E5077')
        ))
        
        styles.add(ParagraphStyle(
            name='CustomNormal',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=11,
            spaceAfter=12
        ))
        
        # Get local timezone
        local_tz = pytz.timezone('Asia/Phnom_Penh')  # Set to Cambodia timezone
        current_time = datetime.now(local_tz)
        
        # Add title and date with custom styles
        title = Paragraph(f"Implementation Progress Dashboard - {selected_sheet}", styles['CustomTitle'])
        date_str = Paragraph(f"Generated on: {current_time.strftime('%d %b %Y %H:%M')}", styles['CustomNormal'])
        story.extend([title, date_str, Spacer(1, 20)])
        
        # 1. Executive Summary Section
        story.append(Paragraph("1. Executive Summary", styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 12))
        
        # Calculate key metrics for executive summary
        total_sites = len(df)
        completed_sites = df['oa actual'].notna().sum()
        completion_rate = round((completed_sites / total_sites * 100) if total_sites > 0 else 0)
        
        if not df.empty:
            start_date = df["forecast oa date"].min()
            end_date = df["forecast oa date"].max()
            elapsed_days = (current_time.replace(tzinfo=None) - start_date).days
            remaining_days = (end_date - current_time.replace(tzinfo=None)).days
            daily_rate = round(completed_sites / max(elapsed_days, 1))
            required_rate = round((total_sites - completed_sites) / max(remaining_days, 1))
        
        # Create executive summary table
        exec_summary_data = [
            ["Project Overview", "Timeline", "Implementation Rate"],
            [
                f"Total Sites: {total_sites:,}\nCompleted: {completed_sites:,}\nProgress: {completion_rate}%",
                f"Start Date: {start_date.strftime('%d %b %Y')}\nEnd Date: {end_date.strftime('%d %b %Y')}\nElapsed: {elapsed_days} days\nRemaining: {remaining_days} days",
                f"Current Rate: {daily_rate:,} sites/day\nRequired Rate: {required_rate:,} sites/day\nCompletion Rate: {completion_rate}%"
            ]
        ]
        
        exec_table = Table(exec_summary_data, colWidths=[160, 160, 160])
        exec_table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5077')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('ROWHEIGHT', (0, 1), (-1, 1), 80),
        ]))
        
        story.append(exec_table)
        story.append(Spacer(1, 20))
        
        # 2. Implementation Progress Charts
        story.append(Paragraph("2. Implementation Progress", styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 12))
        
        try:
            # Add daily progress chart
            daily_fig = create_plotly_progress_chart(df, df[df['oa actual'].notna()], "Daily Implementation Progress")
            if daily_fig:
                img_buffer = convert_plotly_to_image(daily_fig)
                if img_buffer:
                    story.append(Image(img_buffer, width=500, height=300))
                    story.append(Spacer(1, 20))
            
            # Add weekly progress chart
            weekly_fig = create_weekly_progress_chart(df, df[df['oa actual'].notna()], "Weekly Implementation Progress")
            if weekly_fig:
                img_buffer = convert_plotly_to_image(weekly_fig)
                if img_buffer:
                    story.append(Image(img_buffer, width=500, height=300))
                    story.append(Spacer(1, 20))
        except Exception as e:
            st.error(f"Error adding progress charts: {str(e)}")
        
        # 3. Geographic Distribution
        story.append(Paragraph("3. Geographic Distribution", styles['CustomHeading1']))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
        story.append(Spacer(1, 12))
        
        try:
            map_buffer = capture_map_for_pdf(df, provinces_gdf, selected_sheet)
            if map_buffer:
                map_img = Image(map_buffer, width=500, height=400)
                story.append(map_img)
                story.append(Spacer(1, 20))
        except Exception as e:
            st.error(f"Error adding map: {str(e)}")
        
        # 4. Provincial Analysis
        if 'province' in df.columns:
            story.append(Paragraph("4. Provincial Analysis", styles['CustomHeading1']))
            story.append(HRFlowable(width="100%", thickness=1, color=colors.black))
            story.append(Spacer(1, 12))
            
            # Calculate province metrics
            province_data = []
            total_by_province = df['province'].value_counts()
            completed_by_province = df[df['oa actual'].notna()]['province'].value_counts()
            
            for province in total_by_province.index:
                total = total_by_province.get(province, 0)
                completed = completed_by_province.get(province, 0)
                completion_rate = round((completed / total * 100), 1)
                province_data.append({
                    'Province': province,
                    'Total': total,
                    'Completed': completed,
                    'Rate': completion_rate
                })
            
            # Sort by total sites descending
            province_data.sort(key=lambda x: x['Total'], reverse=True)
            
            # Create a copy of province data and reverse it for the bar chart only
            chart_data = province_data[::-1]
            
            # Create province analysis visualization
            fig_province = go.Figure()
            
            # Add bars for total and completed sites (use chart_data for reversed order)
            provinces = [p['Province'] for p in chart_data]
            totals = [p['Total'] for p in chart_data]
            completed = [p['Completed'] for p in chart_data]

            # Add the base bars first
            fig_province.add_trace(go.Bar(
                y=provinces,
                x=totals,
                name='Total Sites',
                orientation='h',
                marker=dict(
                    color='rgb(220, 220, 220)',
                    line=dict(color='black', width=1)
                ),
                showlegend=True
            ))
            
            fig_province.add_trace(go.Bar(
                y=provinces,
                x=completed,
                name='Completed Sites',
                orientation='h',
                marker=dict(
                    color='rgb(0, 100, 0)',
                    line=dict(color='black', width=1)
                ),
                showlegend=True
            ))

            # Add text layers for total sites (outside the bars)
            for i, (prov, tot) in enumerate(zip(provinces, totals)):
                # Black background text
                fig_province.add_trace(go.Scatter(
                    x=[tot],
                    y=[prov],
                    mode='text',
                    text=[f"{tot:,}"],
                    textposition='right',
                    textfont=dict(
                        size=14,
                        color='black',
                        family='Arial Black'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))
                # White text slightly offset
                fig_province.add_trace(go.Scatter(
                    x=[tot],
                    y=[prov],
                    mode='text',
                    text=[f"{tot:,}"],
                    textposition='right',
                    textfont=dict(
                        size=12,
                        color='white',
                        family='Arial Black'
                    ),
                    showlegend=False,
                    hoverinfo='none'
                ))

            # Add text layers for completed sites (inside the bars)
            for i, (prov, comp) in enumerate(zip(provinces, completed)):
                if comp > 0:  # Only add text for non-zero values
                    # Black outline text
                    fig_province.add_trace(go.Scatter(
                        x=[comp/2],  # Center of the bar
                        y=[prov],
                        mode='text',
                        text=[f"{comp:,}"],
                        textposition='middle center',
                        textfont=dict(
                            size=14,
                            color='black',
                            family='Arial Black'
                        ),
                        showlegend=False,
                        hoverinfo='none'
                    ))
                    # White text slightly offset
                    fig_province.add_trace(go.Scatter(
                        x=[comp/2],  # Center of the bar
                        y=[prov],
                        mode='text',
                        text=[f"{comp:,}"],
                        textposition='middle center',
                        textfont=dict(
                            size=12,
                            color='white',
                            family='Arial Black'
                        ),
                        showlegend=False,
                        hoverinfo='none'
                    ))
            
            # Update layout with stronger contrast
            fig_province.update_layout(
                barmode='overlay',
                height=max(400, len(province_data) * 35),  # Increased height for better spacing
                margin=dict(l=150, r=150, t=40, b=40),
                title=dict(
                    text='<b>Provincial Implementation Status</b>',
                    font=dict(
                        size=16,
                        color='black',
                        family='Arial Black'
                    ),
                    x=0.5,
                    xanchor='center'
                ),
                showlegend=True,
                bargap=0.3,  # Increased gap
                plot_bgcolor='white',
                paper_bgcolor='white',
                yaxis=dict(
                    tickfont=dict(
                        size=12,
                        color='black',
                        family='Arial Black'
                    ),
                    gridcolor='black',
                    ticklen=5,
                    tickwidth=1,
                    tickcolor='black',
                    linecolor='black',
                    linewidth=1,
                    showgrid=True,
                    gridwidth=0.5
                ),
                xaxis=dict(
                    tickfont=dict(
                        size=12,
                        color='black',
                        family='Arial Black'
                    ),
                    gridcolor='black',
                    ticklen=5,
                    tickwidth=1,
                    tickcolor='black',
                    linecolor='black',
                    linewidth=1,
                    showgrid=True,
                    gridwidth=0.5
                ),
                legend=dict(
                    font=dict(
                        size=12,
                        color='black',
                        family='Arial Black'
                    ),
                    bgcolor='white',
                    bordercolor='black',
                    borderwidth=1,
                    x=1,
                    y=1.02,
                    yanchor='bottom',
                    xanchor='right'
                )
            )

            # Add a white background with black border
            fig_province.add_shape(
                type="rect",
                xref="paper",
                yref="paper",
                x0=0,
                y0=0,
                x1=1,
                y1=1,
                line=dict(color="black", width=2),
                fillcolor="white",
                layer="below"
            )
            
            # Convert chart to image and add to PDF
            img_buffer = convert_plotly_to_image(fig_province)
            if img_buffer:
                story.append(Image(img_buffer, width=500, height=max(300, len(province_data) * 20)))
                story.append(Spacer(1, 20))
            
            # Add detailed province table (use original province_data for descending order)
            table_data = [["Province", "Total Sites", "Completed", "Completion Rate"]]
            for p in province_data:  # Use original province_data here
                table_data.append([
                    p['Province'],
                    f"{p['Total']:,}",
                    f"{p['Completed']:,}",
                    f"{p['Rate']:.1f}%"
                ])
            
            province_table = Table(table_data, colWidths=[120, 100, 100, 100])
            province_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),
                ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E5077')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), ['#F0F0F0', 'white']),
                ('TEXTCOLOR', (-1, 1), (-1, -1), lambda x: colors.green if x == 'On Track' else colors.red)
            ]))
            
            story.append(province_table)
            story.append(Spacer(1, 20))
        
        # Build PDF
        try:
            doc.build(story)
            buffer.seek(0)
            return buffer
        except Exception as e:
            st.error(f"Error building PDF: {str(e)}")
            return None
            
    except Exception as e:
        st.error(f"Error in PDF generation: {str(e)}")
        return None

def create_province_visualizations(df):
    """Create visualizations for province analysis"""
    if 'province' not in df.columns:
        return None, None
    
    # Calculate metrics by province
    total_by_province = df['province'].value_counts()
    completed_by_province = df[df['oa actual'].notna()]['province'].value_counts()
    
    # Create DataFrame for visualization
    province_data = pd.DataFrame({
        'Province': total_by_province.index,
        'Total Sites': total_by_province.values,
        'Completed Sites': [completed_by_province.get(p, 0) for p in total_by_province.index]
    })
    
    # Calculate completion rates
    province_data['Completion Rate'] = (province_data['Completed Sites'] / province_data['Total Sites'] * 100).round(1)
    province_data = province_data.sort_values('Total Sites', ascending=False)
    
    # Reverse the order to show highest values at top for horizontal bar chart
    province_data = province_data.iloc[::-1]  # Add this line to reverse the order
    
    # Create horizontal bar chart
    bar_fig = go.Figure()
    
    # Add bars for total sites
    bar_fig.add_trace(go.Bar(
        y=province_data['Province'],
        x=province_data['Total Sites'],
        name='Pending Sites',
        orientation='h',
        marker_color='rgba(169, 169, 169, 0.9)',
        text=province_data['Total Sites'].apply(lambda x: f'{x:,}'),
        textposition='auto',
    ))
    
    # Add bars for completed sites
    bar_fig.add_trace(go.Bar(
        y=province_data['Province'],
        x=province_data['Completed Sites'],
        name='Completed Sites',
        orientation='h',
        marker_color='rgba(0, 100, 0, 0.85)',
        text=province_data['Completed Sites'].apply(lambda x: f'{x:,}'),
        textposition='auto',
    ))
    
    # Update layout
    bar_fig.update_layout(
        title='Province Implementation Status',
        barmode='overlay',
        height=max(400, len(province_data) * 30),
        margin=dict(l=10, r=10, t=40, b=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    # Create treemap
    treemap_fig = go.Figure(go.Treemap(
        labels=province_data['Province'],
        parents=[''] * len(province_data),
        values=province_data['Total Sites'],
        customdata=np.column_stack((
            province_data['Completed Sites'],
            province_data['Completion Rate']
        )),
        hovertemplate='<b>%{label}</b><br>' +
                      'Total Sites: %{value}<br>' +
                      'Completed Sites: %{customdata[0]}<br>' +
                      'Completion Rate: %{customdata[1]:.1f}%<extra></extra>',
        marker=dict(
            colors=province_data['Completion Rate'],
            colorscale=[
                [0, 'rgb(165,0,38)'],
                [0.5, 'rgb(255,191,0)'],
                [1, 'rgb(0,100,0)']
            ],
            showscale=True,
            colorbar=dict(
                title='Completion Rate (%)',
                thickness=15,
                len=0.9,
                tickformat='.0f'
            )
        )
    ))
    
    treemap_fig.update_layout(
        title='Province Analysis',
        height=500,
        margin=dict(l=10, r=10, t=40, b=10)
    )
    
    return bar_fig, treemap_fig

def main():
    """Main dashboard function"""
    st.title("📊 Implementation Progress Dashboard")
    
    try:
        # File selection
        script_dir = os.path.dirname(os.path.abspath(__file__))
        default_file = "Cellcard Project_Overall_Implementation_Plan(Sitewise)_V2.2_20250409.xlsx"
        file_path = os.path.join(script_dir, default_file)
        
        if not os.path.exists(file_path):
            st.error(f"Error: Implementation plan file not found")
            return
        
        # Load available sheets
        excel_file = pd.ExcelFile(file_path)
        sheet_names = excel_file.sheet_names
        
        # Sheet selection in sidebar
        st.sidebar.title("Dashboard Controls")
        selected_sheet = st.sidebar.selectbox(
            "Select Project Sheet",
            options=sheet_names,
            format_func=lambda x: f"📑 {x}"
        )
        
        # Auto-refresh option
        auto_refresh = st.sidebar.checkbox("Enable Auto-refresh")
        if auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=30,
                max_value=300,
                value=60,
                step=30
            )
        
        # Load and process data
        df = load_data(file_path, selected_sheet)
        if df is None:
            return
            
        # Process forecast and actual data
        df_forecast = df.dropna(subset=["forecast oa date"])
        df_actual = df.dropna(subset=["oa actual"])
        
        # Calculate key metrics
        total_sites = len(df_forecast)
        completed_sites = len(df_actual)
        completion_rate = (completed_sites / total_sites * 100) if total_sites > 0 else 0
        
        # Display key metrics in cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class="status-card">
                    <h3>📈 Overall Progress</h3>
                    <hr>
            """, unsafe_allow_html=True)
            st.metric("Total Sites", f"{total_sites:,}")
            st.metric("Completed Sites", f"{completed_sites:,}", f"{completion_rate:.1f}%")
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Add province summary if available
            if 'province' in df.columns:
                st.markdown("""
                    <div class="status-card" style="margin-top: 1rem;">
                        <h3>📍 Completed Sites by Province</h3>
                        <hr>
                """, unsafe_allow_html=True)
                
                # Calculate province metrics
                total_by_province = df['province'].value_counts()
                completed_by_province = df_actual['province'].value_counts()
                
                # Display metrics only for provinces with completed sites
                for province in sorted(completed_by_province.index):
                    total = total_by_province.get(province, 0)
                    completed = completed_by_province.get(province, 0)
                    completion_rate = (completed / total * 100) if total > 0 else 0
                    
                    st.metric(
                        province,
                        f"{completed:,} of {total:,}",
                        f"{completion_rate:.1f}%",
                        help=f"Completion rate for {province}"
                    )
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
                <div class="status-card">
                    <h3>⏱️ Timeline</h3>
                    <hr>
            """, unsafe_allow_html=True)
            if not df_actual.empty and not df_forecast.empty:
                start_date = df_forecast["forecast oa date"].min()
                end_date = df_forecast["forecast oa date"].max()
                current_date = datetime.now()
                elapsed_days = (current_date - start_date).days
                remaining_days = (end_date - current_date).days
                st.metric("Days Elapsed", f"{elapsed_days:,}")
                st.metric("Days Remaining", f"{remaining_days:,}")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
                <div class="status-card">
                    <h3>🎯 Implementation Rate</h3>
                    <hr>
            """, unsafe_allow_html=True)
            if not df_actual.empty:
                daily_rate = int(round(completed_sites / max(elapsed_days, 1)))
                required_rate = int(round((total_sites - completed_sites) / max(remaining_days, 1)))
                st.metric("Current Rate", f"{daily_rate:,} sites/day")
                st.metric("Required Rate", f"{required_rate:,} sites/day")
            st.markdown("</div>", unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
                <div class="status-card">
                    <h3>📅 Last Implementation</h3>
                    <hr>
            """, unsafe_allow_html=True)
            if not df_actual.empty:
                last_completion = df_actual["oa actual"].max()
                days_since_last = (datetime.now() - last_completion).days
                st.metric(
                    "Last Completion",
                    last_completion.strftime("%d %b %Y"),
                    f"{days_since_last} days ago",
                    delta_color="inverse" if days_since_last > 7 else "normal"
                )
                sites_on_last_day = len(df_actual[df_actual["oa actual"].dt.date == last_completion.date()])
                st.metric("Sites on Last Day", f"{sites_on_last_day:,}")
            else:
                st.metric("Last Completion", "No data")
                st.metric("Sites on Last Day", "0")
            st.markdown("</div>", unsafe_allow_html=True)

        # Add project summary section after metrics and before chart
        if df is not None and not df_actual.empty:
            summary_html = create_project_summary(df_forecast, df_actual)
            st.markdown(summary_html, unsafe_allow_html=True)

        # Create and display charts in a container
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Daily Progress", "📊 Weekly Progress", "🗺️ Map View", "📍 Province Analysis"])
        
        with tab1:
            daily_fig = create_plotly_progress_chart(
                df_forecast,
                df_actual,
                "Daily Implementation Progress"
            )
            if daily_fig is not None:
                st.plotly_chart(daily_fig, use_container_width=True)
        
        with tab2:
            weekly_fig = create_weekly_progress_chart(
                df_forecast,
                df_actual,
                "Weekly Implementation Progress"
            )
            if weekly_fig is not None:
                st.plotly_chart(weekly_fig, use_container_width=True)
                
        with tab3:
            if selected_sheet in ["349_NOKIA_SWAP(NOKIA)", "185_ALU&HW_SWAP(ALU)", "153_ZTE_UPGRADE", "20_HUAWEI_REDEPLOY(ALU)", "BTB-NEWSITE(ZTE-31&NOKIA-99)"]:
                st.subheader(f"{selected_sheet} Map View")
                
                try:
                    # Verify required columns exist
                    required_columns = ['lat', 'lon', 'sitename', 'oa actual']
                    # Only check for cluster_id and swap_batch if it's a swap project
                    if "SWAP" in selected_sheet:
                        required_columns.extend(['cluster_id', 'swap_batch'])
                    
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    
                    if missing_columns:
                        st.error(f"Missing required columns: {', '.join(missing_columns)}")
                        st.write("Available columns:", df.columns.tolist())
                    else:
                        with st.spinner("Generating map visualization..."):
                            # Load province boundaries
                            cambodia_provinces_gdf = load_shapefile()
                            
                            # Create and display the map with province boundaries
                            map_obj = plot_map(df, cambodia_provinces_gdf, selected_sheet)
                            
                            if map_obj is not None:
                                # Display map with increased height for better visibility
                                st_folium.folium_static(map_obj, width=1200, height=800)
                                
                                # Add summary statistics
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    total_sites = len(df)
                                    completed_sites = df['oa actual'].notna().sum()
                                    completion_rate = (completed_sites / total_sites * 100) if total_sites > 0 else 0
                                    st.metric("Total Sites", f"{total_sites:,}")
                                    st.metric("Completed Sites", f"{completed_sites:,}", 
                                            f"{completion_rate:.1f}% Complete")
                                
                                with col2:
                                    if "SWAP" in selected_sheet:
                                        clusters = df['cluster_id'].nunique()
                                        avg_sites_per_cluster = total_sites / clusters if clusters > 0 else 0
                                        st.metric("Total Clusters", f"{clusters:,}")
                                        st.metric("Avg Sites per Cluster", f"{int(round(avg_sites_per_cluster)):,}")
                                        
                                        batches = df['swap_batch'].nunique()
                                        avg_sites_per_batch = total_sites / batches if batches > 0 else 0
                                        st.metric("Swap Batches", f"{batches:,}")
                                        st.metric("Avg Sites per Batch", f"{int(round(avg_sites_per_batch)):,}")
                                    else:
                                        st.metric("Project Type", "Upgrade/Redeploy/Newsite")
                                        st.metric("Visualization", "Simple markers without clustering")
                            else:
                                st.error("Could not create map. Please check the data format.")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
            else:
                st.info("Map view is not available for this project sheet.")
        
        with tab4:
            if 'province' in df.columns:
                bar_fig, treemap_fig = create_province_visualizations(df)
                if bar_fig is not None:
                    st.plotly_chart(bar_fig, use_container_width=True)
                if treemap_fig is not None:
                    st.plotly_chart(treemap_fig, use_container_width=True)
            else:
                st.info("Province information is not available in this dataset.")

        st.markdown('</div>', unsafe_allow_html=True)
        
        # Add Report Generation Section in sidebar
        st.sidebar.markdown("---")
        st.sidebar.subheader("📑 Report Generation")
        
        if st.sidebar.button("Generate PDF Report"):  # Updated button label
            with st.spinner("Generating report..."):
                st.write("Starting PDF generation process...")
                
                # Check if data is available
                if df_forecast is None or df_actual is None:
                    st.error("No data available for report generation")
                    return
                
                # Generate PDF with charts
                pdf_output = generate_simple_pdf_report(
                    df,
                    load_shapefile(),
                    selected_sheet
                )
                
                if pdf_output:
                    try:
                        # Get the PDF data
                        pdf_data = pdf_output.getvalue()
                        
                        # Create download button
                        st.sidebar.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_data,
                            file_name=f"implementation_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                            mime="application/pdf"
                        )
                        st.sidebar.success("Report generated successfully!")
                    except Exception as e:
                        st.error(f"Error creating download button: {str(e)}")
                else:
                    st.sidebar.error("Failed to generate PDF report. Please check the error messages above.")
        
        # Auto-refresh logic
        if auto_refresh:
            time.sleep(refresh_interval)
            st.experimental_rerun()
            
    except Exception as e:
        st.error(f"Error in main: {str(e)}")
        import traceback
        st.write("Debug: Main error:", traceback.format_exc())

if __name__ == "__main__":
    main() 