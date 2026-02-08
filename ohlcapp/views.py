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
    - output_candles: Number of candles to fetch (e.g., 85) - IGNORED if date_range=true
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
        ohlc_tz = request.POST.get('ohlc_tz', 'utc+3:30')
        date_range = request.POST.get('date_range', 'false').lower() in ['true', 'on']
        from_date_str = request.POST.get('from_date', None)
        to_date_str = request.POST.get('to_date', None)
        
        # Validate date format if date_range mode is enabled
        if date_range and from_date_str and from_date_str.strip():
            from_date_str = from_date_str.strip()
            # Check for invalid dates like '2024-00-00'
            if '00-00' in from_date_str or '-00' in from_date_str:
                return JsonResponse({
                    'error': f'Invalid date format: "{from_date_str}". Month and day cannot be 00. Use format: YYYY-MM-DD (e.g., 2024-01-01)'
                }, status=400)
            
            if to_date_str and to_date_str.strip():
                to_date_str = to_date_str.strip()
                if '00-00' in to_date_str or '-00' in to_date_str:
                    return JsonResponse({
                        'error': f'Invalid date format: "{to_date_str}". Month and day cannot be 00. Use format: YYYY-MM-DD (e.g., 2024-12-31)'
                    }, status=400)
        
        # FIX: Only use output_candles when NOT in date_range mode
        if date_range:
            output_candles = 9999  # Large fallback value (will be ignored by get_and_clean_data)
        else:
            output_candles = int(request.POST.get('output_candles', 85))
        
        # Fetch data
        df = get_and_clean_data(
            date_range=date_range,
            from_date_str=from_date_str,
            to_date_str=to_date_str,
            ohlc_tz_str=ohlc_tz,
            output_candles=output_candles,  # Pass actual value in lookback mode, ignored in date_range mode
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
        import traceback
        traceback.print_exc()
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
    - output_candles: Number of candles - IGNORED if date_range=true
    - ohlc_tz: Timezone
    - date_range: 'true' or 'false'
    - from_date: Start date (optional)
    - to_date: End date (optional)
    
    For 'csv' mode:
    - csv_file: Uploaded CSV file (required)
    - chart_name: Optional name for the chart title
    """
    if request.method != 'POST':
        return JsonResponse({'error': 'POST request required'}, status=400)
    
    try:
        mode = request.POST.get('mode', 'direct')
        
        if mode == 'csv':
            # CSV mode - only needs the file and optional name
            if 'csv_file' not in request.FILES:
                return JsonResponse({'error': 'No CSV file uploaded'}, status=400)
            
            csv_file = request.FILES['csv_file']
            chart_name = request.POST.get('chart_name', 'Chart').strip() or 'Chart'
            
            # Read CSV
            try:
                df = pd.read_csv(csv_file, index_col=0, parse_dates=True)
            except Exception as e:
                return JsonResponse({'error': f'Invalid CSV file: {str(e)}'}, status=400)
            
            # Validate required columns
            required_cols = ['open', 'high', 'low', 'close']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                return JsonResponse({
                    'error': f'CSV missing required columns: {", ".join(missing_cols)}'
                }, status=400)
            
            # Generate chart from CSV
            fig = plot_strategy_from_csv(
                df=df,
                chart_name=chart_name,
            )
            
            # Prepare filename
            filename = f"{chart_name.replace(' ', '_')}_chart_{int(time.time())}.pdf"
            
        else:  # direct mode
            # FIX #2: Proper handling of direct mode parameters
            assets_str = request.POST.get('assets', 'eurusd')
            assets = [a.strip().lower() for a in assets_str.split() if a.strip()]
            tf = int(request.POST.get('tf', 15))
            ohlc_tz = request.POST.get('ohlc_tz', 'utc+3:30')
            date_range = request.POST.get('date_range', 'false').lower() in ['true', 'on']
            from_date_str = request.POST.get('from_date', None)
            to_date_str = request.POST.get('to_date', None)
            
            # Validate date format if date_range mode is enabled
            if date_range and from_date_str and from_date_str.strip():
                from_date_str = from_date_str.strip()
                # Check for invalid dates like '2024-00-00'
                if '00-00' in from_date_str or '-00' in from_date_str:
                    return JsonResponse({
                        'error': f'Invalid date format: "{from_date_str}". Month and day cannot be 00. Use format: YYYY-MM-DD (e.g., 2024-01-01)'
                    }, status=400)
                
                if to_date_str and to_date_str.strip():
                    to_date_str = to_date_str.strip()
                    if '00-00' in to_date_str or '-00' in to_date_str:
                        return JsonResponse({
                            'error': f'Invalid date format: "{to_date_str}". Month and day cannot be 00. Use format: YYYY-MM-DD (e.g., 2024-12-31)'
                        }, status=400)
            
            # FIX: Only use output_candles when NOT in date_range mode
            if date_range:
                output_candles = 9999  # Large fallback value (will be ignored by get_and_clean_data)
            else:
                output_candles = int(request.POST.get('output_candles', 85))
            
            dataframes = []
            
            # Fetch data for each asset
            for asset in assets:
                df = get_and_clean_data(
                    date_range=date_range,
                    from_date_str=from_date_str,
                    to_date_str=to_date_str,
                    ohlc_tz_str=ohlc_tz,
                    output_candles=output_candles,  # Pass actual value in lookback mode, ignored in date_range mode
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
                    ohlc_tz=ohlc_tz,  # FIX #2: Pass timezone directly
                )
            else:
                # Multiple assets - create stacked subplots
                fig = create_multi_asset_chart(dataframes, tf, ohlc_tz)
            
            # Prepare filename
            filename = f"chart_{'_'.join(assets)}_tf{tf}_{int(time.time())}.pdf"
        
        # Save to PDF
        pdf_buffer = io.BytesIO()
        fig.savefig(pdf_buffer, format='pdf', bbox_inches='tight', facecolor=DEFAULT_CONFIG['bg_color'])
        plt.close(fig)
        pdf_buffer.seek(0)
        
        # Create response
        response = HttpResponse(pdf_buffer.getvalue(), content_type='application/pdf')
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
    
    # Each subplot gets height=9, total figure height = 9 * num_assets
    subplot_height = 9
    total_height = subplot_height * num_assets
    
    # Create figure with stacked subplots
    fig, axes = plt.subplots(
        num_assets, 1,
        figsize=(subplot_width, total_height),
        facecolor=DEFAULT_CONFIG['bg_color'],
    )
    
    # Handle single subplot case (axes is not an array)
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


def plot_strategy_from_csv(df, chart_name):
    """
    Plot a single chart from uploaded CSV data.
    
    Args:
        df: DataFrame with datetime index and OHLC columns
        chart_name: Name for the chart title
    
    Returns:
        matplotlib Figure object
    """
    from .utils.plot_candlestick import plot_candlestick
    
    # Calculate figsize
    num_candles = len(df)
    width_per_candle = 0.12
    min_width = 12
    max_width = 300
    height = 9
    
    width = num_candles * width_per_candle
    width = max(min_width, min(max_width, width))
    figsize = (width, height)
    
    # Create figure
    fig, ax = plt.subplots(
        1, 1,
        figsize=figsize,
        facecolor=DEFAULT_CONFIG['bg_color'],
    )
    
    # Get timezone from the dataframe index if available
    timezone_str = ''
    if hasattr(df.index, 'tz') and df.index.tz is not None:
        timezone_str = str(df.index.tz)
    
    # Plot candlestick - use 'CSV' as placeholder for tf to avoid errors
    # The title will show the chart name
    plot_candlestick(
        ax=ax,
        df=df,
        tf='CSV',  # Placeholder since we don't know timeframe
        ticker=chart_name,
        timezone=timezone_str if timezone_str else None,
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
    
    # Override title to just show chart name without tf
    title = ax.set_title(
        chart_name.upper(),
        fontsize=DEFAULT_CONFIG["title_fontsize"],
        fontweight=DEFAULT_CONFIG["title_fontweight"],
        color=DEFAULT_CONFIG["title_color"],
        pad=10
    )
    title.set_bbox(dict(
        facecolor='black',
        alpha=0.2,
        edgecolor='none',
        boxstyle='round,pad=0.5'
    ))
    
    # Add right margin
    xlim = ax.get_xlim()
    right_margin = DEFAULT_CONFIG['right_margin']
    ax.set_xlim(xlim[0], xlim[1] + right_margin)
    
    # Add overlay elements
    _plot_last_price_line(ax, df, chart_name)
    _plot_last_datetime_info(ax, df, chart_name)
    
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