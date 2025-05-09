import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime
import matplotlib as mpl
import re
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Ensure Calibri is available and set as default
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Calibri', 'Arial']  # Fallback to Arial if Calibri not available

# Set up Seaborn style with customizations
sns.set_theme(style="whitegrid")
sns.set_palette("deep")

# Comprehensive style settings
plt.rcParams.update({
    # Font settings
    'font.size': 10,
    'axes.titlesize': 14,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    
    # Grid settings
    'grid.alpha': 0.3,
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    
    # Axis settings
    'axes.spines.top': False,
    'axes.spines.right': True,  # Keep for secondary axis
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    
    # Figure settings
    'figure.facecolor': 'white',
    'figure.figsize': (14, 8),
    'figure.dpi': 100,
    
    # Export settings
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'savefig.facecolor': 'white'
})

# Custom color palette optimized for PowerPoint
colors = {
    'forecast': '#2C3E50',    # Dark blue-gray
    'actual': '#27AE60',      # Forest green
    'positive': '#2ECC71',    # Emerald
    'negative': '#E74C3C',    # Soft red
    'grid': '#CCCCCC',        # Medium gray for better visibility
    'text': '#2C3E50'         # Dark blue-gray
}

# Update Seaborn palette
sns.set_palette([colors['forecast'], colors['actual'], colors['positive'], colors['negative']])

# Function to configure date axis with user-selected or smart formatting
def configure_date_axis(ax, date_range, chart_width, user_interval=None):
    """Configure the date axis with smart formatting based on the date range or user selection"""
    if len(date_range) == 0:
        return
        
    days_span = (date_range[-1] - date_range[0]).days + 1
    
    if user_interval is not None:
        interval = user_interval
        date_format = '%d-%b' if interval < 30 else '%b-%y'
    else:
        estimated_label_width = 40
        optimal_label_count = max(chart_width // estimated_label_width, 4)
        base_interval = max(days_span // optimal_label_count, 1)
        
        if days_span <= 14:
            interval = 2
            date_format = '%d-%b'
        elif days_span <= 30:
            interval = 3
            date_format = '%d-%b'
        elif days_span <= 90:
            interval = 7
            date_format = '%d-%b'
        elif days_span <= 180:
            interval = 14
            date_format = '%d-%b'
        elif days_span <= 365:
            interval = 30
            date_format = '%b-%y'
        else:
            interval = 60
            date_format = '%b-%y'
    
    if days_span > 365 and user_interval is None:
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    else:
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=interval))
    
    ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    
    # Enhanced grid
    ax.grid(axis='x', which='major', linestyle='--', alpha=0.3, color=colors['grid'], linewidth=0.8)
    ax.grid(axis='y', which='major', linestyle='--', alpha=0.3, color=colors['grid'], linewidth=0.8)
    
    # Optimize label rotation
    if days_span <= 90:
        ax.tick_params(axis='x', rotation=45, labelsize=9, pad=5)
    else:
        ax.tick_params(axis='x', rotation=30, labelsize=9, pad=3)

    # Ensure all text uses Calibri
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontname('Calibri')

# Function to create daily progress chart
def create_daily_progress_chart(df_forecast, df_actual, selected_sheet, script_dir, date_interval=None):
    # Count occurrences by day for each dataset separately
    forecast_counts = df_forecast["forecast oa date"].value_counts().sort_index()
    actual_counts = df_actual["oa actual"].value_counts().sort_index() if not df_actual.empty else pd.Series(dtype='float64')

    # Create separate DataFrames for forecast and actual with a complete date range
    all_dates = pd.date_range(
        start=min(forecast_counts.index.min(),
                actual_counts.index.min() if not actual_counts.empty else forecast_counts.index.min()),
        end=max(forecast_counts.index.max(),
                actual_counts.index.max() if not actual_counts.empty else forecast_counts.index.max()),
        freq='D'
    )

    # Create forecast DataFrame with complete date range
    forecast_df = pd.DataFrame(index=all_dates)
    forecast_df["Forecast Count"] = forecast_counts.reindex(all_dates, fill_value=0)
    forecast_df["Cumulative Forecast"] = forecast_df["Forecast Count"].cumsum()
    
    # For dates beyond the last forecast date, use the final cumulative forecast value
    last_forecast_date = forecast_counts.index.max()
    if last_forecast_date is not None:
        last_cumulative_forecast = forecast_df.loc[last_forecast_date, "Cumulative Forecast"]
        forecast_df.loc[forecast_df.index > last_forecast_date, "Cumulative Forecast"] = last_cumulative_forecast

    # Create actual DataFrame only up to the last actual date
    last_actual_date = actual_counts.index.max() if not actual_counts.empty else None
    if last_actual_date is not None:
        actual_dates = pd.date_range(start=all_dates.min(), end=last_actual_date, freq='D')
        actual_df = pd.DataFrame(index=actual_dates)
        actual_df["Actual Count"] = actual_counts.reindex(actual_dates, fill_value=0)
        actual_df["Cumulative Actual"] = actual_df["Actual Count"].cumsum()
    else:
        actual_df = pd.DataFrame(index=all_dates)
        actual_df["Actual Count"] = 0
        actual_df["Cumulative Actual"] = 0

    # Calculate completion stats
    total_forecasted = len(df_forecast)
    total_completed = len(df_actual)
    completion_percentage = (total_completed / total_forecasted * 100) if total_forecasted > 0 else 0

    # Create a full DataFrame for plotting
    comparison_df = pd.DataFrame(index=all_dates)
    comparison_df = comparison_df.join(forecast_df, how='left')
    comparison_df = comparison_df.join(actual_df, how='left')
    comparison_df = comparison_df.fillna(0)

    # Calculate the gap between actual and forecast counts
    comparison_df["Daily Gap"] = comparison_df["Actual Count"] - comparison_df["Forecast Count"]
    comparison_df["Cumulative Gap"] = comparison_df["Cumulative Actual"] - comparison_df["Cumulative Forecast"]

    # Filter for dates that have actual data
    gap_df = comparison_df[comparison_df['Actual Count'] > 0]

    # Determine project status using latest actual date
    if not gap_df.empty:
        latest_date = gap_df.index.max()
        overall_gap = comparison_df.loc[latest_date, "Cumulative Gap"]
        if overall_gap > 0:
            status = "EXCEEDING TARGET"
            status_color = "#2c6b2c"
        elif overall_gap < 0:
            status = "PROGRESSING"
            status_color = "#1f77b4"
        else:
            status = "ON TARGET"
            status_color = "#2c6b2c"
    else:
        overall_gap = 0
        status = "INITIALIZED"
        status_color = "#1f77b4"

    # Set up the figure with an elegant, minimalist style for executive presentation
    fig, ax1 = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')

    # Use refined, professional colors suitable for executive presentations
    forecast_color = '#0a62a9'  # Deep corporate blue
    actual_color = '#0a8045'    # Deep corporate green
    positive_gap_color = '#379b52'  # More subtle green
    negative_gap_color = '#b13b3b'  # More subtle red

    # Plot forecast line with minimal markers
    line1, = ax1.plot(forecast_df.index, forecast_df["Cumulative Forecast"], 
            label=f"Forecast OA ({total_forecasted:,} sites)", 
            marker="o", markersize=4, 
            linewidth=2.0, color=forecast_color, zorder=3)

    # Add today's date line with professional styling
    today = pd.Timestamp.now().normalize()
    if today >= forecast_df.index.min() and today <= forecast_df.index.max():
        # Add vertical line for today with deeper color
        ax1.axvline(x=today, color='#2c6b9c', linestyle='--', linewidth=1.2, alpha=0.6, zorder=2)
        
        # Calculate y position for "Today" label - place it at the top of chart
        y_range = ax1.get_ylim()
        y_pos = y_range[1] * 0.95  # Position at 95% of chart height
        
        # Add "Today" label with enhanced visibility
        ax1.text(today, y_pos, ' Today ', 
                rotation=90, 
                color='white',  # White text for contrast
                fontsize=9,
                fontweight='bold',  # Made bold for emphasis
                ha='center',  # Center align
                va='bottom',
                bbox=dict(facecolor='#2c6b9c',  # Deeper blue background
                         edgecolor='#1a4971',  # Darker border
                         alpha=0.9,  # More opaque
                         pad=3,
                         boxstyle='round,pad=0.5'),
                zorder=4)

    # Calculate dynamic offsets based on data density
    def calculate_dynamic_offset(index, value, prev_value=None, prev_date=None):
        # Start with base offset
        offset = 15
        
        # If we have previous values, check density
        if prev_value is not None and prev_date is not None:
            days_diff = (index - prev_date).days
            value_diff = abs(value - prev_value)
            value_pct_diff = value_diff / max(value, prev_value)
            
            # Adjust offset based on temporal and value proximity
            if days_diff <= 3:  # Close in time
                if value_pct_diff < 0.05:  # Very close values
                    offset = 35
                elif value_pct_diff < 0.1:  # Moderately close values
                    offset = 25
                else:
                    offset = 20
            elif days_diff <= 7:  # Moderately close in time
                if value_pct_diff < 0.05:
                    offset = 25
                elif value_pct_diff < 0.1:
                    offset = 20
        
        return offset

    # Add labels to forecast points with dynamic positioning
    prev_forecast_date = None
    prev_forecast_value = None
    
    for i, (date, row) in enumerate(forecast_df.iterrows()):
        if row['Cumulative Forecast'] > 0:
            # Calculate dynamic offset
            offset = calculate_dynamic_offset(date, row['Cumulative Forecast'], 
                                           prev_forecast_value, prev_forecast_date)
            
            # Alternate between top and bottom for dense regions
            if prev_forecast_date is not None:
                days_diff = (date - prev_forecast_date).days
                if days_diff <= 3:
                    offset = -offset if i % 2 == 1 else offset
            
            ax1.annotate(f"{int(row['Cumulative Forecast'])}", 
                        (date, row['Cumulative Forecast']),
                        textcoords="offset points", 
                        xytext=(0, offset), 
                        ha='center',
                        fontsize=9,
                        fontweight='bold',
                        color=forecast_color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=forecast_color, alpha=0.7),
                        zorder=4)
            
            prev_forecast_date = date
            prev_forecast_value = row['Cumulative Forecast']

    # Plot actual line with minimal markers - only if there is actual data
    if not df_actual.empty:
        line2, = ax1.plot(actual_df.index, actual_df["Cumulative Actual"], 
                label=f"Actual OA ({total_completed:,} sites, {completion_percentage:.1f}%)", 
                marker="s", markersize=4, 
                linewidth=2.0, color=actual_color, zorder=3)
        
        # For actual data - show points with dynamic positioning
        prev_actual_date = None
        prev_actual_value = None
        
        actual_dates = actual_df[actual_df["Actual Count"] > 0].index
        for i, date in enumerate(actual_dates):
            value = actual_df.loc[date, "Cumulative Actual"]
            
            # Calculate dynamic offset for actual values
            offset = calculate_dynamic_offset(date, value, prev_actual_value, prev_actual_date)
            
            # Check proximity to forecast value if it exists
            if date in forecast_df.index:
                forecast_value = forecast_df.loc[date, "Cumulative Forecast"]
                value_diff = abs(value - forecast_value)
                if value_diff / max(value, forecast_value) < 0.1:  # If within 10%
                    offset = -offset  # Place on opposite side
            
            # Alternate between top and bottom for dense regions
            if prev_actual_date is not None:
                days_diff = (date - prev_actual_date).days
                if days_diff <= 3:
                    offset = -offset if i % 2 == 0 else offset
            
            ax1.annotate(f"{int(value)}", 
                        (date, value),
                        textcoords="offset points", 
                        xytext=(0, offset), 
                        ha='center',
                        fontsize=9,
                        fontweight='bold',
                        color=actual_color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=actual_color, alpha=0.7),
                        zorder=4)
            
            prev_actual_date = date
            prev_actual_value = value

    # Create secondary y-axis for gaps
    ax2 = ax1.twinx()

    # Calculate appropriate scaling for dual axes
    max_cumulative = max(
        forecast_df["Cumulative Forecast"].max() if not forecast_df.empty else 0,
        actual_df["Cumulative Actual"].max() if not actual_df.empty else 0
    )
    max_gap = max(abs(gap_df["Cumulative Gap"].max() if not gap_df.empty else 0), 
                abs(gap_df["Cumulative Gap"].min() if not gap_df.empty else 0))

    # Set appropriate axis limits
    ax1.set_ylim(0, max_cumulative * 1.15)  # Main axis with headroom
    gap_scale = max(max_gap * 1.5, max_cumulative * 0.25)
    ax2.set_ylim(-gap_scale, gap_scale)  # Secondary axis with balanced range

    # Create subtle gap bars
    bars = ax2.bar(gap_df.index, gap_df["Cumulative Gap"], width=0.7, align='center', alpha=0.6, zorder=2)

    # Color the bars based on positive/negative values
    for i, bar in enumerate(bars):
        if gap_df["Cumulative Gap"].iloc[i] >= 0:
            bar.set_color(positive_gap_color)
        else:
            bar.set_color(negative_gap_color)

    # Add labels to ALL gap bars
    for i, (date, row) in enumerate(gap_df.iterrows()):
        bar_height = row['Cumulative Gap']
        
        ax2.annotate(f"{int(row['Cumulative Gap'])}", 
                    xy=(date, bar_height),
                    xytext=(0, 5 if bar_height >= 0 else -15),
                    textcoords="offset points",
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaaaaa", alpha=0.7),
                    zorder=4)

    # Add professional title and labels
    report_date = datetime.now().strftime("%b %d, %Y")
    ax1.set_title(f"Daily Implementation Progress: {selected_sheet}", 
            fontsize=16, fontweight="bold", pad=20)
    ax1.set_ylabel("Cumulative Site Count", fontsize=11, fontweight="regular", labelpad=10, color="#333333")
    ax2.set_ylabel("Cumulative Gap", fontsize=11, fontweight="regular", labelpad=10, color="#666666")

    # Format the x-axis with smart date formatting - UPDATED
    configure_date_axis(ax1, all_dates, fig.get_figwidth() * fig.dpi, user_interval=date_interval)

    # Add subtle grid lines only where needed
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#dddddd')
    ax1.grid(axis='x', linestyle='--', alpha=0.2, color='#dddddd')

    # Optimize tick styling - moved to configure_date_axis function
    # ax1.tick_params(axis='x', rotation=90, labelsize=9, color='#333333', pad=5)
    ax1.tick_params(axis='y', labelsize=9, color='#333333')
    ax2.tick_params(axis='y', labelsize=9, color='#666666')

    # Add reference line for gaps
    ax2.axhline(y=0, color='#666666', linestyle='-', alpha=0.3, linewidth=1, zorder=1)

    # Add subtle background to differentiate dual axis areas
    ax2.fill_between(all_dates, ax2.get_ylim()[0], ax2.get_ylim()[1], 
                    color='#f5f5f5', alpha=0.2, zorder=0)

    # Create a clean, executive-friendly legend with proper representation of gap types
    from matplotlib.patches import Patch

    # Start with the line elements - only include actual line if there is data
    legend_elements = [line1]
    if not df_actual.empty:
        legend_elements.append(line2)

    # Add separate legend items for positive and negative gaps if they exist
    has_positive_gaps = (gap_df["Cumulative Gap"] > 0).any() if not gap_df.empty else False
    has_negative_gaps = (gap_df["Cumulative Gap"] < 0).any() if not gap_df.empty else False

    if has_positive_gaps:
        legend_elements.append(
            Patch(facecolor=positive_gap_color, edgecolor=positive_gap_color, alpha=0.6, 
                label="Cumulative Gap: Ahead of Forecast")
        )
        
    if has_negative_gaps:
        legend_elements.append(
            Patch(facecolor=negative_gap_color, edgecolor=negative_gap_color, alpha=0.6, 
                label="Cumulative Gap: Behind Forecast")
        )

    # Create the legend with improved styling
    leg = ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, 
            frameon=True, facecolor='white', edgecolor='#dddddd', framealpha=0.9)
    leg.get_frame().set_linewidth(0.5)

    # Add executive summary annotation in top-right showing key metrics
    if completion_percentage > 0:
        # Calculate daily completion rate using actual data points
        actual_dates = actual_counts.index.sort_values()
        first_actual_date = actual_dates.min()
        last_actual_date = actual_dates.max()
        days_elapsed = (last_actual_date - first_actual_date).days + 1
        
        # Calculate implementation rate (total completed / actual working days)
        daily_completion_rate = total_completed / max(days_elapsed, 1)

        # Get target completion date from forecast
        target_completion_date = forecast_counts.index.max()
        
        # Calculate if project is on track
        current_date = datetime.now()
        days_to_deadline = (target_completion_date - current_date).days
        
        # Calculate required daily rate to meet target
        remaining_sites = total_forecasted - total_completed
        if days_to_deadline > 0:
            required_daily_rate = remaining_sites / days_to_deadline
            if daily_completion_rate >= required_daily_rate:
                timeline_status = "ON SCHEDULE"
            else:
                timeline_status = "ADDITIONAL FOCUS NEEDED"
        else:
            timeline_status = "TIMELINE EXTENDED"

        exec_summary = (
            f"Project Implementation Summary (as of {datetime.now().strftime('%d %b %Y')})\n"
            f"• Status: {status}\n"
            f"• Progress: {int(completion_percentage)}% Complete ({total_completed:,}/{total_forecasted:,} sites)\n"
            f"• Implementation Rate: {int(daily_completion_rate)} sites/day\n"
            f"• Timeline: {timeline_status}\n"
            f"• Target Completion: {target_completion_date.strftime('%b %Y')}"
        )
        
        # Create text box for executive summary with enhanced styling
        props = dict(
            boxstyle='round,pad=1.0',
            facecolor='white',
            alpha=0.95,
            ec=status_color,
            lw=2
        )
        
        # Position the summary in the top right with improved visibility and left-aligned text
        ax1.text(0.98, 0.99, exec_summary,
                transform=ax1.transAxes,
                fontsize=10,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='left',
                bbox=props,
                zorder=5,
                linespacing=1.5)

    # Enhance chart aesthetics for executive presentation
    # Add chart border
    fig.patch.set_edgecolor('#666666')
    fig.patch.set_linewidth(1)
    
    # Add subtle gradient background
    ax1.set_facecolor('#f8f9fa')
    
    # Enhance grid styling
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#666666', zorder=1)
    ax1.grid(axis='x', linestyle='--', alpha=0.2, color='#666666', zorder=1)
    
    # Make title more prominent
    ax1.set_title(f"Daily Implementation Progress Report: {selected_sheet}",
                fontsize=16,
                fontweight="bold",
                pad=20,
                color='#333333')
    
    # Add subtle footer with report generation date
    fig.text(0.99, 0.01,
            f"Report generated: {datetime.now().strftime('%Y-%m-%d')}",
            fontsize=8,
            color='#666666',
            ha='right',
            va='bottom')
    
    # Enhance legend styling
    leg.set_title("Progress Indicators", prop={'size': 10, 'weight': 'bold'})
    
    # Add data source reference
    fig.text(0.01, 0.01,
            "Data Source: Project Implementation Tracking System",
            fontsize=8,
            color='#666666',
            ha='left',
            va='bottom')
            
    # Adjust layout with new elements
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08, right=0.95)  # Adjust margins for new elements

    # Save the plot with PowerPoint-optimized settings
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    safe_sheet_name = re.sub(r'[\\/*?:"<>|]', "_", selected_sheet)

    # Save PNG with high DPI for PowerPoint
    comparison_plot_path_png = os.path.join(output_dir, f"CTO_Daily_Progress_{safe_sheet_name}.png")
    plt.savefig(comparison_plot_path_png, 
                dpi=300,  # High DPI for sharp display
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1,  # Slight padding
                format='png',
                transparent=False)  # Solid white background for PowerPoint

    # Save SVG for vector graphics
    comparison_plot_path_svg = os.path.join(output_dir, f"CTO_Daily_Progress_{safe_sheet_name}.svg")
    plt.savefig(comparison_plot_path_svg,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1,
                format='svg')

    print(f"Saved PowerPoint-ready plots:")
    print(f"PNG (high-res): {comparison_plot_path_png}")
    print(f"SVG (vector): {comparison_plot_path_svg}")

    plt.close()
    
    return comparison_df  # Return the data for potential further analysis

# Function to create weekly progress chart
def create_weekly_progress_chart(df_forecast, df_actual, selected_sheet, script_dir):
    # Calculate week information - ISO week (starts on Monday)
    df_forecast['iso_year'] = df_forecast["forecast oa date"].dt.isocalendar().year
    df_forecast['iso_week'] = df_forecast["forecast oa date"].dt.isocalendar().week
    df_forecast['year_week'] = df_forecast['iso_year'].astype(str) + '-W' + df_forecast['iso_week'].astype(str).str.zfill(2)

    if not df_actual.empty:
        df_actual['iso_year'] = df_actual["oa actual"].dt.isocalendar().year
        df_actual['iso_week'] = df_actual["oa actual"].dt.isocalendar().week
        df_actual['year_week'] = df_actual['iso_year'].astype(str) + '-W' + df_actual['iso_week'].astype(str).str.zfill(2)

    # Group by ISO week and count
    forecast_weekly = df_forecast.groupby('year_week').size().reset_index(name='Forecast Count')
    forecast_weekly = forecast_weekly.sort_values('year_week')
    forecast_weekly["Cumulative Forecast"] = forecast_weekly["Forecast Count"].cumsum()

    if not df_actual.empty:
        actual_weekly = df_actual.groupby('year_week').size().reset_index(name='Actual Count')
        actual_weekly = actual_weekly.sort_values('year_week')
        actual_weekly["Cumulative Actual"] = actual_weekly["Actual Count"].cumsum()
    else:
        actual_weekly = pd.DataFrame(columns=['year_week', 'Actual Count', 'Cumulative Actual'])

    # Create a merged dataframe for comparison with all weeks
    all_weeks = pd.DataFrame({'year_week': pd.unique(pd.concat([forecast_weekly['year_week'], actual_weekly['year_week']]))})
    all_weeks = all_weeks.sort_values('year_week')

    # Merge weekly data
    comparison_weekly = all_weeks.merge(forecast_weekly, on='year_week', how='left').merge(
        actual_weekly, on='year_week', how='left')
    comparison_weekly = comparison_weekly.fillna(0)

    # For weeks beyond the last forecast week, use the final cumulative forecast value
    last_forecast_week = forecast_weekly['year_week'].iloc[-1] if not forecast_weekly.empty else None
    if last_forecast_week is not None:
        last_cumulative_forecast = forecast_weekly.loc[forecast_weekly['year_week'] == last_forecast_week, 'Cumulative Forecast'].iloc[0]
        comparison_weekly.loc[comparison_weekly['year_week'] > last_forecast_week, 'Cumulative Forecast'] = last_cumulative_forecast

    # Create date representations of the weeks for plotting
    comparison_weekly['week_date'] = comparison_weekly['year_week'].apply(
        lambda x: datetime.strptime(f"{x}-1", '%Y-W%W-%w')
    )

    # Calculate weekly gaps
    comparison_weekly["Weekly Gap"] = comparison_weekly["Actual Count"] - comparison_weekly["Forecast Count"]
    comparison_weekly["Cumulative Gap"] = comparison_weekly["Cumulative Actual"] - comparison_weekly["Cumulative Forecast"]

    # Filter for weeks that have actual data
    gap_weekly = comparison_weekly[comparison_weekly['Actual Count'] > 0]

    # Determine if we're ahead or behind overall using the latest cumulative gap
    if not gap_weekly.empty:
        latest_week = gap_weekly.index.max()
        overall_gap = comparison_weekly.loc[latest_week, "Cumulative Gap"]
        if overall_gap > 0:
            status = "EXCEEDING TARGET"
            status_color = "#2c6b2c"
        elif overall_gap < 0:
            status = "PROGRESSING"
            status_color = "#1f77b4"
        else:
            status = "ON TARGET"
            status_color = "#2c6b2c"
    else:
        overall_gap = 0
        status = "INITIALIZED"
        status_color = "#1f77b4"

    # Set up the figure with an elegant, minimalist style for executive presentation
    fig, ax1 = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')

    # Use refined, professional colors suitable for executive presentations
    forecast_color = '#0a62a9'  # Deep corporate blue
    actual_color = '#0a8045'    # Deep corporate green
    positive_gap_color = '#379b52'  # More subtle green
    negative_gap_color = '#b13b3b'  # More subtle red

    # Plot forecast line with minimal markers
    line1, = ax1.plot(comparison_weekly['week_date'], comparison_weekly["Cumulative Forecast"], 
            label=f"Forecast OA ({len(df_forecast)} sites)", 
            marker="o", markersize=6, 
            linewidth=2.0, color=forecast_color, zorder=3)

    # Plot actual line with minimal markers
    if not actual_weekly.empty:
        line2, = ax1.plot(
            comparison_weekly[comparison_weekly['Actual Count'] > 0]['week_date'], 
            comparison_weekly[comparison_weekly['Actual Count'] > 0]["Cumulative Actual"], 
            label=f"Actual OA ({len(df_actual)} sites, {len(df_actual) / len(df_forecast) * 100:.1f}%)", 
            marker="s", markersize=6, 
            linewidth=2.0, color=actual_color, zorder=3)
    else:
        line2, = ax1.plot([], [], label="Actual OA (0 sites)", 
                marker="s", markersize=6, 
                linewidth=2.0, color=actual_color, zorder=3)

    # Create secondary y-axis for gaps
    ax2 = ax1.twinx()

    # Calculate appropriate scaling for dual axes
    max_cumulative = max(
        comparison_weekly["Cumulative Forecast"].max() if not comparison_weekly.empty else 0,
        comparison_weekly["Cumulative Actual"].max() if not comparison_weekly.empty else 0
    )
    max_gap = max(abs(gap_weekly["Cumulative Gap"].max() if not gap_weekly.empty else 0), 
                abs(gap_weekly["Cumulative Gap"].min() if not gap_weekly.empty else 0))

    # Set appropriate axis limits
    ax1.set_ylim(0, max_cumulative * 1.15)  # Main axis with headroom
    gap_scale = max(max_gap * 1.5, max_cumulative * 0.25)
    ax2.set_ylim(-gap_scale, gap_scale)  # Secondary axis with balanced range

    # Create subtle gap bars - only for weeks with actual data
    if not gap_weekly.empty:
        bars = ax2.bar(gap_weekly['week_date'], gap_weekly["Cumulative Gap"], width=5, align='center', alpha=0.6, zorder=2)

        # Color the bars based on positive/negative values
        for i, bar in enumerate(bars):
            if gap_weekly["Cumulative Gap"].iloc[i] >= 0:
                bar.set_color(positive_gap_color)
            else:
                bar.set_color(negative_gap_color)

    # Add data labels for ALL forecast points
    for i, row in comparison_weekly.iterrows():
        if row["Cumulative Forecast"] > 0:
            # Alternate label positions to avoid overlap when weeks are close
            vert_offset = 10 if i % 2 == 0 else -15
            
            ax1.annotate(f"{int(row['Cumulative Forecast'])}", 
                        (row['week_date'], row['Cumulative Forecast']),
                        textcoords="offset points", 
                        xytext=(0, vert_offset), 
                        ha='center',
                        fontsize=9,
                        fontweight='bold',
                        color=forecast_color,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=forecast_color, alpha=0.7),
                        zorder=4)

    # For actual data - showing all points
    actual_weeks = comparison_weekly[comparison_weekly['Actual Count'] > 0]
    for i, row in actual_weeks.iterrows():
        ax1.annotate(f"{int(row['Cumulative Actual'])}", 
                    (row['week_date'], row['Cumulative Actual']),
                    textcoords="offset points", 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color=actual_color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=actual_color, alpha=0.7),
                    zorder=4)

    # Add labels to ALL gap bars
    for i, row in gap_weekly.iterrows():
        bar_height = row['Cumulative Gap']
        
        ax2.annotate(f"{int(row['Cumulative Gap'])}", 
                    xy=(row['week_date'], bar_height),
                    xytext=(0, 5 if bar_height >= 0 else -15),
                    textcoords="offset points",
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    color='black',
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#aaaaaa", alpha=0.7),
                    zorder=4)

    # Add professional title and labels
    report_date = datetime.now().strftime("%b %d, %Y")
    ax1.set_title(f"Weekly Implementation Progress: {selected_sheet}", 
            fontsize=16, fontweight="bold", pad=20)
    ax1.set_ylabel("Cumulative Site Count", fontsize=11, fontweight="regular", labelpad=10, color="#333333")
    ax2.set_ylabel("Cumulative Gap", fontsize=11, fontweight="regular", labelpad=10, color="#666666")

    # Format the x-axis with week numbers
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('Week %W\n%b %d'))
    ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))  # Monday

    # Add subtle grid lines only where needed
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#dddddd')
    ax1.grid(axis='x', linestyle='--', alpha=0.2, color='#dddddd')

    # Optimize tick styling
    ax1.tick_params(axis='x', rotation=0, labelsize=9, color='#333333', pad=5)
    ax1.tick_params(axis='y', labelsize=9, color='#333333')
    ax2.tick_params(axis='y', labelsize=9, color='#666666')

    # Add reference line for gaps
    ax2.axhline(y=0, color='#666666', linestyle='-', alpha=0.3, linewidth=1, zorder=1)

    # Add subtle background to differentiate dual axis areas
    min_date = comparison_weekly['week_date'].min()
    max_date = comparison_weekly['week_date'].max()
    ax2.fill_between([min_date, max_date], ax2.get_ylim()[0], ax2.get_ylim()[1], 
                    color='#f5f5f5', alpha=0.2, zorder=0)

    # Create a clean, executive-friendly legend with proper representation of gap types
    from matplotlib.patches import Patch

    # Start with the line elements
    legend_elements = [
        line1,
        line2,
    ]

    # Add separate legend items for positive and negative gaps if they exist
    has_positive_gaps = (gap_weekly["Cumulative Gap"] > 0).any() if not gap_weekly.empty else False
    has_negative_gaps = (gap_weekly["Cumulative Gap"] < 0).any() if not gap_weekly.empty else False

    if has_positive_gaps:
        legend_elements.append(
            Patch(facecolor=positive_gap_color, edgecolor=positive_gap_color, alpha=0.6, 
                label="Cumulative Gap: Ahead of Forecast")
        )
        
    if has_negative_gaps:
        legend_elements.append(
            Patch(facecolor=negative_gap_color, edgecolor=negative_gap_color, alpha=0.6, 
                label="Cumulative Gap: Behind Forecast")
        )

    # Create the legend with improved styling
    leg = ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, 
            frameon=True, facecolor='white', edgecolor='#dddddd', framealpha=0.9)
    leg.get_frame().set_linewidth(0.5)

    # Add executive summary annotation in top-right showing key metrics
    if len(df_actual) > 0:
        # Calculate weekly completion rate using actual data
        weeks_elapsed = len(actual_weekly)
        weekly_completion_rate = len(df_actual) / max(weeks_elapsed, 1)

        # Get target completion date from forecast
        target_completion_date = forecast_weekly['year_week'].iloc[-1]
        target_completion_datetime = datetime.strptime(f"{target_completion_date}-1", '%Y-W%W-%w')
        
        # Calculate if project is on track
        current_date = datetime.now()
        weeks_to_deadline = (target_completion_datetime - current_date).days / 7
        
        # Calculate required weekly rate to meet target
        remaining_sites = len(df_forecast) - len(df_actual)
        if weeks_to_deadline > 0:
            required_weekly_rate = remaining_sites / weeks_to_deadline
            if weekly_completion_rate >= required_weekly_rate:
                timeline_status = "ON SCHEDULE"
            else:
                timeline_status = "ADDITIONAL FOCUS NEEDED"
        else:
            timeline_status = "TIMELINE EXTENDED"

        completion_percentage = (len(df_actual) / len(df_forecast) * 100)

        exec_summary = (
            f"Project Implementation Summary (as of {datetime.now().strftime('%d %b %Y')})\n"
            f"• Status: {status}\n"
            f"• Progress: {int(completion_percentage)}% Complete ({len(df_actual):,}/{len(df_forecast):,} sites)\n"
            f"• Implementation Rate: {int(weekly_completion_rate)} sites/week\n"
            f"• Timeline: {timeline_status}\n"
            f"• Target Completion: {target_completion_datetime.strftime('%b %Y')}"
        )
        
        # Create text box for executive summary with enhanced styling
        props = dict(
            boxstyle='round,pad=1.0',
            facecolor='white',
            alpha=0.95,
            ec=status_color,
            lw=2
        )
        
        # Position the summary in the top right with improved visibility and left-aligned text
        ax1.text(0.98, 0.99, exec_summary,
                transform=ax1.transAxes,
                fontsize=10,
                fontweight='bold',
                verticalalignment='top',
                horizontalalignment='left',
                bbox=props,
                zorder=5,
                linespacing=1.5)

    # Enhance chart aesthetics for executive presentation
    # Add chart border
    fig.patch.set_edgecolor('#666666')
    fig.patch.set_linewidth(1)
    
    # Add subtle gradient background
    ax1.set_facecolor('#f8f9fa')
    
    # Enhance grid styling
    ax1.grid(axis='y', linestyle='--', alpha=0.3, color='#666666', zorder=1)
    ax1.grid(axis='x', linestyle='--', alpha=0.2, color='#666666', zorder=1)
    
    # Make title more prominent
    ax1.set_title(f"Implementation Progress Report: {selected_sheet}",
                fontsize=16,
                fontweight="bold",
                pad=20,
                color='#333333')
    
    # Add subtle footer with report generation date
    fig.text(0.99, 0.01,
            f"Report generated: {datetime.now().strftime('%Y-%m-%d')}",
            fontsize=8,
            color='#666666',
            ha='right',
            va='bottom')
    
    # Enhance legend styling
    leg.set_title("Progress Indicators", prop={'size': 10, 'weight': 'bold'})
    
    # Add data source reference
    fig.text(0.01, 0.01,
            "Data Source: Project Implementation Tracking System",
            fontsize=8,
            color='#666666',
            ha='left',
            va='bottom')
            
    # Adjust layout with new elements
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.08, right=0.95)  # Adjust margins for new elements

    # Save the plot with PowerPoint-optimized settings
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    safe_sheet_name = re.sub(r'[\\/*?:"<>|]', "_", selected_sheet)

    # Save PNG with high DPI for PowerPoint
    comparison_plot_path_png = os.path.join(output_dir, f"CTO_Weekly_Progress_{safe_sheet_name}.png")
    plt.savefig(comparison_plot_path_png,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1,
                format='png',
                transparent=False)

    # Save SVG for vector graphics
    comparison_plot_path_svg = os.path.join(output_dir, f"CTO_Weekly_Progress_{safe_sheet_name}.svg")
    plt.savefig(comparison_plot_path_svg,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1,
                format='svg')

    print(f"Saved PowerPoint-ready plots:")
    print(f"PNG (high-res): {comparison_plot_path_png}")
    print(f"SVG (vector): {comparison_plot_path_svg}")

    plt.close()
    
    return comparison_weekly  # Return the data for potential further analysis

# Function to create gap analysis chart
def create_gap_analysis_chart(df_forecast, df_actual, selected_sheet, script_dir):
    if len(df_forecast[df_forecast['oa actual'].isna()]) == 0:
        return

    # Create figure with clean design
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Use consistent color scheme
    colors = ['#2C3E50', '#27AE60', '#E74C3C', '#F39C12', '#8E44AD'][:5]
    
    # Calculate gap analysis
    gap_data = df_forecast[df_forecast['oa actual'].isna()]['gap oa analysis'].value_counts()
    
    # Create simplified pie chart
    wedges, texts, autotexts = ax.pie(gap_data,
                                     colors=colors,
                                     autopct='%1.1f%%',
                                     pctdistance=0.85,
                                     startangle=90)
    
    # Clean up labels
    plt.setp(autotexts, size=9, weight="bold", color="white")
    plt.setp(texts, size=9)
    
    # Add minimal title
    plt.title(f"{selected_sheet}\nImplementation Gap Analysis",
             pad=20,
             fontsize=14,
             color='#2C3E50')
    
    # Add clean legend
    legend = plt.legend(gap_data.index,
                       title="Gap Categories",
                       loc="center left",
                       bbox_to_anchor=(1, 0, 0.5, 1))
    
    # Save with PowerPoint-optimized settings
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    safe_sheet_name = re.sub(r'[\\/*?:"<>|]', "_", selected_sheet)

    # Save PNG with high DPI for PowerPoint
    gap_analysis_path_png = os.path.join(output_dir, f"CTO_Gap_Analysis_{safe_sheet_name}.png")
    plt.savefig(gap_analysis_path_png,
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1,
                format='png',
                transparent=False)

    # Save SVG for vector graphics
    gap_analysis_path_svg = os.path.join(output_dir, f"CTO_Gap_Analysis_{safe_sheet_name}.svg")
    plt.savefig(gap_analysis_path_svg,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none',
                pad_inches=0.1,
                format='svg')

    print(f"Saved PowerPoint-ready plots:")
    print(f"PNG (high-res): {gap_analysis_path_png}")
    print(f"SVG (vector): {gap_analysis_path_svg}")

    plt.close()

# Example of how the daily progress chart would look with Plotly
def create_daily_progress_chart_plotly(df_forecast, df_actual, selected_sheet, script_dir, date_interval=None):
    # Process data same as before
    forecast_counts = df_forecast["forecast oa date"].value_counts().sort_index()
    actual_counts = df_actual["oa actual"].value_counts().sort_index() if not df_actual.empty else pd.Series(dtype='float64')

    all_dates = pd.date_range(
        start=min(forecast_counts.index.min(),
                actual_counts.index.min() if not actual_counts.empty else forecast_counts.index.min()),
        end=max(forecast_counts.index.max(),
                actual_counts.index.max() if not actual_counts.empty else forecast_counts.index.max()),
        freq='D'
    )

    # Create DataFrames
    forecast_df = pd.DataFrame(index=all_dates)
    forecast_df["Forecast Count"] = forecast_counts.reindex(all_dates, fill_value=0)
    forecast_df["Cumulative Forecast"] = forecast_df["Forecast Count"].cumsum()
    
    if not df_actual.empty:
        actual_df = pd.DataFrame(index=all_dates)
        actual_df["Actual Count"] = actual_counts.reindex(all_dates, fill_value=0)
        actual_df["Cumulative Actual"] = actual_df["Actual Count"].cumsum()
    else:
        actual_df = pd.DataFrame(index=all_dates)
        actual_df["Actual Count"] = 0
        actual_df["Cumulative Actual"] = 0

    # Calculate stats
    total_forecasted = len(df_forecast)
    total_completed = len(df_actual)
    completion_percentage = (total_completed / total_forecasted * 100) if total_forecasted > 0 else 0

    # Create Plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add forecast line
    fig.add_trace(
        go.Scatter(
            x=forecast_df.index,
            y=forecast_df["Cumulative Forecast"],
            name=f"Forecast ({total_forecasted:,} sites)",
            line=dict(color='#2C3E50', width=2),
            mode='lines+markers',
            marker=dict(size=8, symbol='circle', line=dict(color='#2C3E50', width=2)),
        ),
        secondary_y=False
    )

    # Add actual line if data exists
    if not df_actual.empty:
        fig.add_trace(
            go.Scatter(
                x=actual_df.index,
                y=actual_df["Cumulative Actual"],
                name=f"Actual ({total_completed:,} sites, {completion_percentage:.1f}%)",
                line=dict(color='#27AE60', width=2),
                mode='lines+markers',
                marker=dict(size=8, symbol='square', line=dict(color='#27AE60', width=2)),
            ),
            secondary_y=False
        )

    # Add gap bars
    if not df_actual.empty:
        gap_df = actual_df.copy()
        gap_df["Cumulative Gap"] = actual_df["Cumulative Actual"] - forecast_df["Cumulative Forecast"]
        
        colors = ['#2ECC71' if x >= 0 else '#E74C3C' for x in gap_df["Cumulative Gap"]]
        
        fig.add_trace(
            go.Bar(
                x=gap_df.index,
                y=gap_df["Cumulative Gap"],
                name="Implementation Gap",
                marker_color=colors,
                opacity=0.3
            ),
            secondary_y=True
        )

    # Update layout for clean, modern look
    fig.update_layout(
        title={
            'text': f"{selected_sheet}<br>Implementation Progress",
            'y':0.95,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=24, color='#2C3E50')
        },
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family="Calibri", size=12, color='#2C3E50'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255, 255, 255, 0.9)',
            bordercolor='#E0E0E0'
        ),
        hovermode='x unified',
        margin=dict(t=100, l=50, r=50, b=50)
    )

    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#E0E0E0',
        zeroline=False,
        title_text="Date"
    )

    fig.update_yaxes(
        title_text="Cumulative Site Count",
        showgrid=True,
        gridwidth=1,
        gridcolor='#E0E0E0',
        zeroline=False,
        secondary_y=False
    )

    fig.update_yaxes(
        title_text="Implementation Gap",
        showgrid=False,
        secondary_y=True
    )

    # Add today's reference line
    today = pd.Timestamp.now().normalize()
    if today >= all_dates.min() and today <= all_dates.max():
        fig.add_vline(
            x=today,
            line_dash="dash",
            line_color="#666666",
            opacity=0.5,
            annotation_text="Today",
            annotation_position="top"
        )

    # Save both interactive and static versions
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    safe_sheet_name = re.sub(r'[\\/*?:"<>|]', "_", selected_sheet)
    
    # Save interactive HTML
    fig.write_html(os.path.join(output_dir, f"CTO_Daily_Progress_{safe_sheet_name}.html"))
    
    # Save static image
    fig.write_image(os.path.join(output_dir, f"CTO_Daily_Progress_{safe_sheet_name}.png"), 
                    width=1400, height=800, scale=2)

    return fig

# Process a single sheet
def process_sheet(file_path, sheet_name, script_dir, date_interval=None):
    print(f"\n--- Processing sheet: {sheet_name} ---")
    
    try:
        # Read the sheet
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        # Normalize column names
        df.columns = df.columns.str.strip().str.lower()
        
        # Check if required columns exist
        if "forecast oa date" not in df.columns or "oa actual" not in df.columns:
            print(f"Error: Required columns 'Forecast OA Date' and 'OA Actual' not found in the sheet '{sheet_name}'. Skipping.")
            return False
        
        # Try different approaches to convert dates
        try:
            # First attempt with default parser
            df["forecast oa date"] = pd.to_datetime(df["forecast oa date"], errors="coerce")
            df["oa actual"] = pd.to_datetime(df["oa actual"], errors="coerce")
            
            # Count valid dates in each column
            valid_forecast = df["forecast oa date"].notna().sum()
            valid_actual = df["oa actual"].notna().sum()
            
            print(f"Valid dates count: Forecast OA Date: {valid_forecast}, OA Actual: {valid_actual}")
            
            if valid_forecast == 0:
                print(f"WARNING: No valid dates found in 'Forecast OA Date' column for sheet '{sheet_name}'!")
                # Try a different format if no valid dates were found
                print("Attempting to parse dates with explicit format...")
                df["forecast oa date"] = pd.to_datetime(df["forecast oa date"].astype(str), errors="coerce", format="%Y-%m-%d")
                valid_forecast = df["forecast oa date"].notna().sum()
                print(f"After format conversion: Valid Forecast OA Date count: {valid_forecast}")
        except Exception as e:
            print(f"Error during date conversion for sheet '{sheet_name}': {e}")
            return False
        
        # Process forecast and actual data separately
        df_forecast = df.dropna(subset=["forecast oa date"])
        df_actual = df.dropna(subset=["oa actual"])
        
        print(f"After filtering: Forecast rows: {len(df_forecast)}, Actual rows: {len(df_actual)}")
        
        if len(df_forecast) == 0:
            print(f"Error: No valid forecast data for sheet '{sheet_name}'. Skipping.")
            return False
        
        # Create both daily and weekly charts
        print(f"Generating Daily Progress Chart for '{sheet_name}'...")
        create_daily_progress_chart(df_forecast, df_actual, sheet_name, script_dir, date_interval)
        
        print(f"Generating Weekly Progress Chart for '{sheet_name}'...")
        create_weekly_progress_chart(df_forecast, df_actual, sheet_name, script_dir)
        
        # Add the gap analysis chart
        create_gap_analysis_chart(df_forecast, df_actual, sheet_name, script_dir)
        
        print(f"--- Completed processing sheet: {sheet_name} ---")
        return True
        
    except Exception as e:
        print(f"Error processing sheet '{sheet_name}': {e}")
        return False

# Define the script directory
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the Excel file
file_path = os.path.join(script_dir, "Cellcard Project_Overall_Implementation_Plan(Sitewise)_V2.2_20250409.xlsx")
if not os.path.exists(file_path):
    print(f"Error: File not found at {file_path}")
    exit()

# List available sheets
excel_file = pd.ExcelFile(file_path)
print("Available sheets:")
for i, sheet_name in enumerate(excel_file.sheet_names):
    print(f"{i + 1}: {sheet_name}")

# Prompt for multi-sheet selection
while True:
    print("\nOptions:")
    print("  - Enter numbers separated by commas (e.g., '1,3,5') to select specific sheets")
    print("  - Enter 'all' to process all sheets")
    print("  - Enter 'q' to quit")
    
    selection = input("\nEnter your selection: ").strip().lower()
    
    if selection == 'q':
        print("Exiting program.")
        exit()
    
    if selection == 'all':
        selected_sheets = excel_file.sheet_names
        break
    
    try:
        # Parse the comma-separated list of sheet indices
        indices = [int(idx.strip()) - 1 for idx in selection.split(',')]
        
        # Validate all indices
        invalid_indices = [idx + 1 for idx in indices if idx < 0 or idx >= len(excel_file.sheet_names)]
        if invalid_indices:
            print(f"Invalid sheet numbers: {', '.join(map(str, invalid_indices))}. Please try again.")
            continue
        
        # Get the selected sheet names
        selected_sheets = [excel_file.sheet_names[idx] for idx in indices]
        break
    
    except ValueError:
        print("Invalid input. Please enter numbers separated by commas, 'all', or 'q'.")

# Get the desired date interval from the user
print("\nChoose the date interval for the x-axis labels:")
print("  - Enter a number for specific day intervals (e.g., '1' for daily, '2' for every 2 days)")
print("  - Common options: 1 (daily), 2, 3, 5, 7 (weekly), 14 (bi-weekly), 30 (monthly)")
print("  - Enter '0' or press Enter to use automatic smart formatting")

interval_input = input("\nEnter your desired interval: ").strip()
date_interval = None
if interval_input and interval_input != '0':
    try:
        date_interval = int(interval_input)
        if date_interval < 1:
            print("Invalid interval. Using automatic smart formatting instead.")
            date_interval = None
        else:
            print(f"Using {date_interval}-day intervals for x-axis labels.")
    except ValueError:
        print("Invalid input. Using automatic smart formatting instead.")

# Process each selected sheet
print(f"\nProcessing {len(selected_sheets)} sheet(s)...")
successful_sheets = 0

for sheet_name in selected_sheets:
    success = process_sheet(file_path, sheet_name, script_dir, date_interval)
    if success:
        successful_sheets += 1

print(f"\nCompleted processing {successful_sheets} out of {len(selected_sheets)} selected sheets.")
print(f"Output files are saved in: {os.path.join(script_dir, 'output')}")