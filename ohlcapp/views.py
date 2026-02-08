from django.shortcuts import render

# Create your views here.
# ohlcapp/views.py

from django.shortcuts import render
from django.http import HttpResponse, FileResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
import io
import os
import time
from .utils.get_and_clean_data import get_and_clean_data
from .utils.main import plot_strategy, DEFAULT_CONFIG
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def index(request):
    """Render the main page with both CSV download and visualizer options."""
    return render(request, 'ohlcapp/index.html')


@csrf_exempt
def download_csv(request):
    """
    Download CSV file with OHLC data.
    
    Expected POST parameters:
    - asset: Asset name (e.g., 'btcusd')
    - tf: Timeframe in minutes (e.g., 15)
    - output_candles: Number of candles to fetch (e.g., 85)
    - ohlc_tz: Timezone (e.g., 'utc+3:30')
    - date_range: 'true' or 'false'
    - from_date: Start date (optional, format: 'YYYY-MM-DD HH:MM:SS')
    - to_date: End date (optional, format: 'YYYY-MM-DD HH:MM:SS')
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)
    
    try:
        # Get parameters
        asset = request.POST.get('asset', 'eurusd').strip().lower()
        tf = int(request.POST.get('tf', 15))
        output_candles = int(request.POST.get('output_candles', 85))
        ohlc_tz = request.POST.get('ohlc_tz', 'utc+3:30')
        date_range = request.POST.get('date_range', 'false').lower() == 'true'
        from_date_str = request.POST.get('from_date', None)
        to_date_str = request.POST.get('to_date', None)
        
        # Fetch data
        df = get_and_clean_data(
            date_range=date_range,
            from_date_str=from_date_str,
            to_date_str=to_date_str,
            ohlc_tz_str=ohlc_tz,
            output_candles=output_candles,
            tf=tf,
            asset=asset,
        )
        
        if df is None or df.empty:
            return JsonResponse({'error': 'No data available'}, status=404)
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer)
        csv_buffer.seek(0)
        
        # Create response
        response = HttpResponse(csv_buffer.getvalue(), content_type='text/csv')
        filename = f"{asset}_tf{tf}_{int(time.time())}.csv"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)


@csrf_exempt
def visualize_chart(request):
    """
    Generate and download PDF chart.
    
    Expected POST parameters:
    - mode: 'direct' or 'csv'
    
    For 'direct' mode:
    - assets: Space-separated asset names (e.g., 'btcusd ethusd xrpusd')
    - tf: Timeframe in minutes
    - output_candles: Number of candles
    - ohlc_tz: Timezone
    - date_range: 'true' or 'false'
    - from_date: Start date (optional)
    - to_date: End date (optional)
    
    For 'csv' mode:
    - csv_file: Uploaded CSV file
    - assets: Space-separated asset names (for titles)
    - tf: Timeframe (for titles)
    - ohlc_tz: Timezone (for titles)
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)
    
    try:
        mode = request.POST.get('mode', 'direct')
        assets_str = request.POST.get('assets', 'eurusd')
        assets = [a.strip().lower() for a in assets_str.split() if a.strip()]
        tf = int(request.POST.get('tf', 15))
        ohlc_tz = request.POST.get('ohlc_tz', 'utc+3:30')
        
        # Update DEFAULT_CONFIG with timezone
        DEFAULT_CONFIG['ohlc_tz'] = ohlc_tz
        
        dataframes = []
        
        if mode == 'csv':
            # Load from uploaded CSV
            if 'csv_file' not in request.FILES:
                return JsonResponse({'error': 'No CSV file uploaded'}, status=400)
            
            csv_file = request.FILES['csv_file']
            
            # Read CSV
            df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            
            # If multiple assets specified, assume they want to split/repeat for each
            # For simplicity, we'll use the same data for each asset name
            for asset in assets:
                dataframes.append((asset, df.copy()))
        
        else:  # direct mode
            output_candles = int(request.POST.get('output_candles', 85))
            date_range = request.POST.get('date_range', 'false').lower() == 'true'
            from_date_str = request.POST.get('from_date', None)
            to_date_str = request.POST.get('to_date', None)
            
            # Fetch data for each asset
            for asset in assets:
                df = get_and_clean_data(
                    date_range=date_range,
                    from_date_str=from_date_str,
                    to_date_str=to_date_str,
                    ohlc_tz_str=ohlc_tz,
                    output_candles=output_candles,
                    tf=tf,
                    asset=asset,
                )
                
                if df is not None and not df.empty:
                    dataframes.append((asset, df))
        
        if not dataframes:
            return JsonResponse({'error': 'No data available for any asset'}, status=404)
        
        # Generate chart(s)
        if len(dataframes) == 1:
            # Single asset - use existing plot_strategy
            asset, df = dataframes[0]
            fig = plot_strategy(
                df=df,
                output_candles=len(df),
                tf=tf,
                asset=asset,
            )
        else:
            # Multiple assets - create stacked subplots
            fig = create_multi_asset_chart(dataframes, tf, ohlc_tz)
        
        # Save to PDF
        pdf_buffer = io.BytesIO()
        fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight', facecolor=DEFAULT_CONFIG['bg_color'])
        plt.close(fig)
        pdf_buffer.seek(0)
        
        # Create response
        response = HttpResponse(pdf_buffer.getvalue(), content_type='application/pdf')
        filename = f"chart_{'_'.join(assets)}_tf{tf}_{int(time.time())}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        return response
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JsonResponse({'error': str(e)}, status=500)


def create_multi_asset_chart(dataframes, tf, ohlc_tz):
    """
    Create a multi-asset chart with stacked subplots.
    Each subplot is 9 inches tall (not total figure height).
    
    Args:
        dataframes: List of (asset_name, dataframe) tuples
        tf: Timeframe
        ohlc_tz: Timezone string
    
    Returns:
        matplotlib Figure object
    """
    from .utils.plot_candlestick import plot_candlestick
    
    num_assets = len(dataframes)
    
    # Calculate individual subplot dimensions
    # Use the same width calculation as plot_strategy
    max_candles = max(len(df) for _, df in dataframes)
    width_per_candle = 0.12
    min_width = 12
    max_width = 300
    
    # Each subplot maintains the same width calculation
    subplot_width = max_candles * width_per_candle
    subplot_width = max(min_width, min(max_width, subplot_width))
    
    # IMPORTANT: Each subplot is 9 inches tall, not the total figure
    subplot_height = 9
    total_height = subplot_height * num_assets  # Total figure height
    
    figsize = (subplot_width, total_height)
    
    # Create figure with subplots
    fig, axes = plt.subplots(
        num_assets, 1,
        figsize=figsize,
        facecolor=DEFAULT_CONFIG['bg_color'],
    )
    
    # Ensure axes is always a list
    if num_assets == 1:
        axes = [axes]
    
    # Plot each asset
    for idx, (asset, df) in enumerate(dataframes):
        ax = axes[idx]
        
        # Plot candlestick
        plot_candlestick(
            ax=ax,
            df=df,
            tf=tf,
            ticker=asset,
            timezone=ohlc_tz,
            up_color=DEFAULT_CONFIG["up_color"],
            down_color=DEFAULT_CONFIG["down_color"],
            edge_color=DEFAULT_CONFIG["edge_color"],
            wick_color=DEFAULT_CONFIG["wick_color"],
            volume_color=DEFAULT_CONFIG["volume_color"],
            bg_color=DEFAULT_CONFIG["bg_color"],
            grid_color=DEFAULT_CONFIG["grid_color"],
            grid_style=DEFAULT_CONFIG["grid_style"],
            grid_alpha=DEFAULT_CONFIG["grid_alpha"],
            show_grid=DEFAULT_CONFIG["show_grid"],
            candle_width=DEFAULT_CONFIG["candle_width"],
            date_format=DEFAULT_CONFIG["date_format"],
            rotation=DEFAULT_CONFIG["rotation"],
            show_nontrading=DEFAULT_CONFIG["show_nontrading"],
            title_fontsize=DEFAULT_CONFIG["title_fontsize"],
            title_fontweight=DEFAULT_CONFIG["title_fontweight"],
            title_color=DEFAULT_CONFIG["title_color"],
            label_fontsize=DEFAULT_CONFIG["label_fontsize"],
            label_color=DEFAULT_CONFIG["label_color"],
            tick_fontsize=DEFAULT_CONFIG["tick_fontsize"],
            tick_color=DEFAULT_CONFIG["tick_color"],
            spine_color=DEFAULT_CONFIG["spine_color"],
            spine_linewidth=DEFAULT_CONFIG["spine_linewidth"],
            show_top_spine=DEFAULT_CONFIG["show_top_spine"],
            show_right_spine=DEFAULT_CONFIG["show_right_spine"],
            y_padding=DEFAULT_CONFIG["y_padding"]
        )
        
        # Add right margin
        xlim = ax.get_xlim()
        right_margin = DEFAULT_CONFIG['right_margin']
        ax.set_xlim(xlim[0], xlim[1] + right_margin)
        
        # Add overlay elements (similar to plot_strategy)
        _plot_last_price_line(ax, df, asset)
        _plot_last_datetime_info(ax, df, asset)
    
    plt.tight_layout()
    return fig


def _plot_last_price_line(ax, df, asset):
    """Draw TradingView-style last price line starting from last candle."""
    if df is None or df.empty:
        return

    last_price = float(df['close'].iloc[-1])
    last_x = len(df) - 1
    x_left, x_right = ax.get_xlim()

    ax.plot(
        [last_x, x_right],
        [last_price, last_price],
        linestyle='--',
        linewidth=1.5,
        color='#2962ff',
        alpha=0.9,
        zorder=4,
        clip_on=False
    )

    # Price formatting
    import numpy as np
    if np.abs(last_price) >= 100:
        price_txt = f"{last_price:,.2f}"
    elif np.abs(last_price) >= 1:
        price_txt = f"{last_price:,.4f}"
    else:
        price_txt = f"{last_price:,.6f}"

    ax.text(
        x_right,
        last_price,
        f" {price_txt} ",
        ha='left',
        va='center',
        fontsize=9,
        color='white',
        bbox=dict(
            boxstyle='round,pad=0.35',
            facecolor='#2962ff',
            edgecolor='none',
            alpha=0.95
        ),
        zorder=5,
        clip_on=False
    )


def _plot_last_datetime_info(ax, df, asset):
    """TradingView-style datetime box under the last candle"""
    if df is None or df.empty:
        return

    from matplotlib.transforms import blended_transform_factory
    
    last_datetime = df.index[-1]
    last_x = len(df) - 1
    formatted_date = last_datetime.strftime("%a %d %b '%y  %H:%M")

    trans = blended_transform_factory(ax.transData, ax.transAxes)

    ax.text(
        last_x,
        -0.03,
        formatted_date,
        ha='center',
        va='top',
        fontsize=10,
        color='#d1d4dc',
        bbox=dict(
            boxstyle='round,pad=0.45',
            facecolor='#2a2e39',
            edgecolor='#434651',
            linewidth=1,
            alpha=0.95
        ),
        transform=trans,
        clip_on=False,
        zorder=10
    )