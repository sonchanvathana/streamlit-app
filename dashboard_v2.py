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

# Add this at the beginning of your script to enable better error tracking
st.set_option('deprecation.showPyplotGlobalUse', False)

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
            # Get forecast values for actual dates
            forecast_values = []
            for date in actual_dates:
                # Find the forecast value on or before this date
                mask = forecast_cumulative.index <= date
                if mask.any():
                    forecast_values.append(forecast_cumulative[mask].iloc[-1])
                else:
                    forecast_values.append(0)
            
            forecast_series = pd.Series(forecast_values, index=actual_dates)
            gaps = actual_cumulative - forecast_series
            
            # Split gaps into positive and negative
            positive_gaps = gaps.copy()
            negative_gaps = gaps.copy()
            positive_gaps[positive_gaps <= 0] = None
            negative_gaps[negative_gaps > 0] = None
            
            # Add positive gap bars
            if not positive_gaps.isna().all():
                fig.add_trace(
                    go.Bar(
                        x=actual_dates,
                        y=positive_gaps,
                        name="Ahead of Target",
                        text=[f"+{int(gap):,d}" if not pd.isna(gap) else "" for gap in positive_gaps],
                        textposition="outside",
                        marker=dict(
                            color=colors['ahead'],
                            opacity=0.7,
                            line=dict(
                                color='rgba(0,0,0,0.1)',
                                width=1
                            )
                        ),
                        hovertemplate=(
                            "<b>Implementation Gap</b><br>" +
                            "Date: %{x|%d %b %Y}<br>" +
                            "Gap: %{y:+,.0f} sites<br>" +
                            "<extra></extra>"
                        ),
                        yaxis='y2'  # Use secondary y-axis for gaps
                    )
                )
            
            # Add negative gap bars
            if not negative_gaps.isna().all():
                fig.add_trace(
                    go.Bar(
                        x=actual_dates,
                        y=negative_gaps,
                        name="Behind Target",
                        text=[f"{int(gap):,d}" if not pd.isna(gap) else "" for gap in negative_gaps],
                        textposition="outside",
                        marker=dict(
                            color=colors['behind'],
                            opacity=0.7,
                            line=dict(
                                color='rgba(0,0,0,0.1)',
                                width=1
                            )
                        ),
                        hovertemplate=(
                            "<b>Implementation Gap</b><br>" +
                            "Date: %{x|%d %b %Y}<br>" +
                            "Gap: %{y:+,.0f} sites<br>" +
                            "<extra></extra>"
                        ),
                        yaxis='y2'  # Use secondary y-axis for gaps
                    )
                )

        # Update layout with professional styling and increased size
        fig.update_layout(
            title=dict(
                text=chart_title,
                font=dict(
                    size=24,
                    color=colors['text'],
                    family="Arial"
                ),
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
                font=dict(
                    size=12,
                    color=colors['text']
                )
            ),
            # Increase height and adjust margins
            height=800,  # Increased height
            margin=dict(
                t=100,   # top margin
                b=150,   # increased bottom margin for x-axis labels
                r=100,   # increased right margin for secondary y-axis
                l=50,    # left margin
                pad=10   # padding
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                tickformat="%d %b %Y",
                tickangle=45,
                tickfont=dict(
                    size=12,
                    color=colors['text']
                ),
                title=None,
                zeroline=False,
                range=[start_date, end_date]
            ),
            yaxis=dict(
                title=dict(
                    text="Number of Sites",
                    font=dict(
                        size=14,
                        color=colors['text']
                    )
                ),
                showgrid=True,
                gridcolor=colors['grid'],
                gridwidth=1,
                tickfont=dict(
                    size=12,
                    color=colors['text']
                ),
                zeroline=True,
                zerolinecolor=colors['grid'],
                zerolinewidth=2,
                tickformat=",d",
                rangemode="nonnegative"
            ),
            # Add secondary y-axis for gaps
            yaxis2=dict(
                title=dict(
                    text="Gap (Sites)",
                    font=dict(
                        size=14,
                        color=colors['text']
                    )
                ),
                overlaying='y',
                side='right',
                showgrid=False,
                zeroline=True,
                zerolinecolor=colors['grid'],
                zerolinewidth=2,
                tickformat=",d",
                tickfont=dict(
                    size=12,
                    color=colors['text']
                )
            )
        )

        # Remove the rangeslider to fix the extra axis issue
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
        
        # Process forecast data by week - create a copy to avoid warnings
        df_forecast_copy = df_forecast.copy()
        # Group by ISO week and year
        df_forecast_copy['year_week'] = df_forecast_copy['forecast oa date'].dt.strftime('%Y-W%V')
        df_forecast_copy['week_num'] = df_forecast_copy['forecast oa date'].dt.isocalendar().week
        df_forecast_copy['year'] = df_forecast_copy['forecast oa date'].dt.isocalendar().year
        
        # Create week labels with date ranges
        week_labels = {}
        for year_week, group in df_forecast_copy.groupby('year_week'):
            week_start = group['forecast oa date'].min().normalize()
            # Adjust to Monday if not already
            week_start = week_start - pd.Timedelta(days=week_start.weekday())
            week_end = week_start + pd.Timedelta(days=6)
            week_num = group['week_num'].iloc[0]
            # Changed format to single line with smaller text
            week_labels[year_week] = f"WK{week_num} ({week_start.strftime('%d %b')}-{week_end.strftime('%d %b')})"
        
        weekly_forecast = df_forecast_copy.groupby('year_week').size()
        weekly_forecast_cumsum = weekly_forecast.cumsum()
        
        # Process actual data by week
        if not df_actual.empty:
            df_actual_copy = df_actual.copy()
            df_actual_copy['year_week'] = df_actual_copy['oa actual'].dt.strftime('%Y-W%V')
            weekly_actual = df_actual_copy.groupby('year_week').size()
            weekly_actual_cumsum = weekly_actual.cumsum()
        else:
            weekly_actual_cumsum = pd.Series()

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

        # Add forecast line with value labels
        fig.add_trace(
            go.Scatter(
                x=list(range(len(weekly_forecast_cumsum))),
                y=weekly_forecast_cumsum.values,
                name=f"Target ({len(df_forecast):,} sites)",
                line=dict(color=colors['target'], width=2),
                mode="lines+markers+text",
                text=weekly_forecast_cumsum.values.astype(int),
                textposition="top center",
                textfont=dict(size=10, color=colors['target']),
                fill='tozeroy',
                fillcolor=f"rgba(31, 119, 180, 0.1)",
                hovertemplate="%{customdata}<br>Target: %{y:,.0f} sites<extra></extra>",
                customdata=[week_labels[w] for w in weekly_forecast_cumsum.index]
            )
        )
        
        if not weekly_actual_cumsum.empty:
            # Create index mapping for actual data
            actual_indices = [list(weekly_forecast_cumsum.index).index(w) if w in weekly_forecast_cumsum.index else None 
                            for w in weekly_actual_cumsum.index]
            actual_indices = [i for i in actual_indices if i is not None]
            
            # Add actual line with value labels
            fig.add_trace(
                go.Scatter(
                    x=actual_indices,
                    y=weekly_actual_cumsum.values,
                    name=f"Completed ({len(df_actual):,} sites)",
                    line=dict(color=colors['actual'], width=2),
                    mode="lines+markers+text",
                    text=weekly_actual_cumsum.values.astype(int),
                    textposition="bottom center",
                    textfont=dict(size=10, color=colors['actual']),
                    fill='tozeroy',
                    fillcolor=f"rgba(44, 160, 44, 0.1)",
                    hovertemplate="%{customdata}<br>Completed: %{y:,.0f} sites<extra></extra>",
                    customdata=[week_labels[w] for w in weekly_actual_cumsum.index]
                )
            )

            # Calculate gaps
            forecast_values = weekly_forecast_cumsum.reindex(weekly_actual_cumsum.index).fillna(method='ffill')
            gaps = weekly_actual_cumsum - forecast_values

            # Split gaps into positive and negative
            positive_gaps = gaps.copy()
            negative_gaps = gaps.copy()
            positive_gaps[positive_gaps <= 0] = None
            negative_gaps[negative_gaps > 0] = None
            
            # Add positive gap bars
            if not positive_gaps.isna().all():
                fig.add_trace(
                    go.Bar(
                        x=actual_indices,
                        y=positive_gaps.values,
                        name="Ahead of Target",
                        text=[f"+{int(gap):,d}" if not pd.isna(gap) else "" for gap in positive_gaps],
                        textposition="outside",
                        marker=dict(
                            color=colors['ahead'],
                            opacity=0.7,
                            line=dict(color='rgba(0,0,0,0.1)', width=1)
                        ),
                        hovertemplate="%{customdata}<br>Gap: %{y:+,.0f} sites<extra></extra>",
                        customdata=[week_labels[w] for w in positive_gaps.index],
                        yaxis='y2'
                    )
                )
            
            # Add negative gap bars
            if not negative_gaps.isna().all():
                fig.add_trace(
                    go.Bar(
                        x=actual_indices,
                        y=negative_gaps.values,
                        name="Behind Target",
                        text=[f"{int(gap):,d}" if not pd.isna(gap) else "" for gap in negative_gaps],
                        textposition="outside",
                        marker=dict(
                            color=colors['behind'],
                            opacity=0.7,
                            line=dict(color='rgba(0,0,0,0.1)', width=1)
                        ),
                        hovertemplate="%{customdata}<br>Gap: %{y:+,.0f} sites<extra></extra>",
                        customdata=[week_labels[w] for w in negative_gaps.index],
                        yaxis='y2'
                    )
                )
    
        # Update layout with rotated labels
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
                x=1
            ),
            height=800,
            margin=dict(
                t=100,   # top margin
                b=120,   # bottom margin for rotated labels
                r=100,   # right margin for secondary y-axis
                l=50,    # left margin
                pad=10   # padding
            ),
            xaxis=dict(
                showgrid=True,
                gridcolor=colors['grid'],
                ticktext=[week_labels[w] for w in weekly_forecast_cumsum.index],
                tickvals=list(range(len(weekly_forecast_cumsum))),
                tickangle=45,  # Rotate labels 45 degrees
                tickfont=dict(size=11),  # Slightly smaller font
                title="Project Timeline by Week",
                type='category',
                tickmode='array',
                nticks=len(weekly_forecast_cumsum)  # Show all ticks
            ),
            yaxis=dict(
                title="Number of Sites",
                showgrid=True,
                gridcolor=colors['grid'],
                tickfont=dict(size=12),
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
        
        # Add vertical grid lines at week boundaries
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
    """Create a project summary section"""
    if df_actual.empty:
        return "No implementation data available"

    # Calculate key metrics
    total_sites = len(df_forecast)
    completed_sites = len(df_actual)
    completion_rate = (completed_sites / total_sites * 100) if total_sites > 0 else 0
    
    start_date = df_forecast["forecast oa date"].min()
    end_date = df_forecast["forecast oa date"].max()
    current_date = datetime.now()
    
    elapsed_days = (current_date - start_date).days
    remaining_days = (end_date - current_date).days
    total_duration = (end_date - start_date).days
    
    # Calculate daily expected vs actual
    expected_sites = len(df_forecast[df_forecast["forecast oa date"] <= current_date])
    expected_completion = (expected_sites / total_sites * 100) if total_sites > 0 else 0
    daily_difference = completion_rate - expected_completion
    
    # Calculate weekly expected vs actual
    current_week_start = current_date - pd.Timedelta(days=current_date.weekday())
    current_week_end = current_week_start + pd.Timedelta(days=7)
    last_week_start = current_week_start - pd.Timedelta(weeks=1)
    
    # Get weekly forecast and actual counts for last complete week
    last_week_forecast = len(df_forecast[
        (df_forecast["forecast oa date"] >= last_week_start) & 
        (df_forecast["forecast oa date"] < current_week_start)
    ])
    last_week_actual = len(df_actual[
        (df_actual["oa actual"] >= last_week_start) & 
        (df_actual["oa actual"] < current_week_start)
    ])
    
    # Get current week's progress
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
    
    # Calculate rates without decimal points
    current_rate = int(round(completed_sites / max(elapsed_days, 1)))
    required_rate = int(round((total_sites - completed_sites) / max(remaining_days, 1)))
    
    last_completion = df_actual["oa actual"].max()
    days_since_last = (current_date - last_completion).days
    sites_on_last_day = len(df_actual[df_actual["oa actual"].dt.date == last_completion.date()])
    
    # Calculate weekly trends without decimal points
    weekly_completions = df_actual.groupby(df_actual["oa actual"].dt.isocalendar().week).size()
    avg_weekly_rate = int(round(weekly_completions.mean()))
    
    # Format week dates for display
    last_week_dates = f"{last_week_start.strftime('%d %b')} - {(current_week_start - pd.Timedelta(days=1)).strftime('%d %b')}"
    current_week_dates = f"{current_week_start.strftime('%d %b')} - {current_date.strftime('%d %b')}"
    
    # Determine status considering both daily and weekly progress
    if daily_difference >= 10 or (daily_difference >= -5 and (last_week_performance >= 100 or current_week_performance >= 100)):
        status = "AHEAD OF SCHEDULE"
        status_color = "#28A745"
        status_detail = (
            f"Daily: {daily_difference:+.1f}% vs forecast | "
            f"Last Week ({last_week_dates}): {last_week_performance:.1f}% | "
            f"This Week ({current_week_dates}): {current_week_performance:.1f}%"
        )
    elif daily_difference >= -10 or last_week_performance >= 90 or current_week_performance >= 90:
        status = "ON TRACK"
        status_color = "#2E5077"
        status_detail = (
            f"Daily: {daily_difference:+.1f}% vs forecast | "
            f"Last Week ({last_week_dates}): {last_week_performance:.1f}% | "
            f"This Week ({current_week_dates}): {current_week_performance:.1f}%"
        )
    else:
        status = "BEHIND SCHEDULE"
        status_color = "#DC3545"
        status_detail = (
            f"Daily: {daily_difference:.1f}% vs forecast | "
            f"Last Week ({last_week_dates}): {last_week_performance:.1f}% | "
            f"This Week ({current_week_dates}): {current_week_performance:.1f}%"
        )

    summary_html = f"""
    <div class="summary-section">
        <div class="summary-header">📊 Project Implementation Summary</div>
        <div class="summary-content">
            <div class="summary-item">
                <div class="summary-item-header">Overall Status</div>
                <div style="color: {status_color}; font-size: 1.2rem; font-weight: 600;">{status}</div>
                <div style="color: {status_color}; font-size: 0.9rem;">{status_detail}</div>
                <div>Overall Completion: {completion_rate:.1f}% ({completed_sites:,} sites)</div>
                <div>Expected to Date: {expected_completion:.1f}% ({expected_sites:,} sites)</div>
                <div>Last Week ({last_week_dates}): {last_week_actual:,} of {last_week_forecast:,} sites ({last_week_performance:.1f}%)</div>
                <div>This Week ({current_week_dates}): {current_week_actual:,} of {current_week_forecast:,} sites ({current_week_performance:.1f}%)</div>
                <div>Total Plan: {total_sites:,} sites</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-header">Timeline</div>
                <div>Start: {start_date.strftime('%d %b %Y')}</div>
                <div>Target End: {end_date.strftime('%d %b %Y')}</div>
                <div>Days Remaining: {remaining_days:,}</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-header">Implementation Rates</div>
                <div>Current: {current_rate:,} sites/day</div>
                <div>Required: {required_rate:,} sites/day</div>
                <div>Weekly Average: {avg_weekly_rate:,} sites</div>
            </div>
            <div class="summary-item">
                <div class="summary-item-header">Recent Activity</div>
                <div>Last Implementation: {last_completion.strftime('%d %b %Y')}</div>
                <div>Days Since Last: {days_since_last}</div>
                <div>Sites on Last Day: {sites_on_last_day:,}</div>
            </div>
        </div>
    </div>
    """
    return summary_html

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
                # Show sites completed on the last day
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
        
        # Create tabs for daily and weekly views
        tab1, tab2, tab3 = st.tabs(["📈 Daily Progress", "📊 Weekly Progress", "🗺️ Map View"])
        
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
                                        st.metric("Avg Sites per Cluster", f"{avg_sites_per_cluster:.1f}")
                                        
                                        batches = df['swap_batch'].nunique()
                                        avg_sites_per_batch = total_sites / batches if batches > 0 else 0
                                        st.metric("Swap Batches", f"{batches:,}")
                                        st.metric("Avg Sites per Batch", f"{avg_sites_per_batch:.1f}")
                                    else:
                                        st.metric("Project Type", "Upgrade/Redeploy/Newsite")
                                        st.metric("Visualization", "Simple markers without clustering")
                            else:
                                st.error("Could not create map. Please check the data format.")
                except Exception as e:
                    st.error(f"Error processing data: {str(e)}")
            else:
                st.info("Map view is not available for this project sheet.")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
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