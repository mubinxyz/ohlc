# utils/plot_candlestick.py 

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates


def plot_candlestick(ax, df, tf, ticker,timezone=None, up_color='#26a69a', down_color='#ef5350',
                     edge_color=None, wick_color=None, volume_color='#7cb5ec',
                     bg_color='white', grid_color='#e0e0e0', grid_style='--',
                     grid_alpha=0.3, show_grid=False, candle_width=0.6,
                     date_format='%m/%d', rotation=45, show_nontrading=False,
                    #  plot_current_price=False,

                     # Font and text styling
                     title_fontsize=14,
                     title_fontweight='bold',
                     title_color='white',
                     label_fontsize=11,
                     label_color='#d1d4dc',
                     tick_fontsize=9,
                     tick_color='#787b86',
                    #  legend_fontsize=10,

                     # Axes styling
                     spine_color='#2a2e39',
                     spine_linewidth=1,
                     show_top_spine=False,
                     show_right_spine=False,

                     # Price info
                    #  show_price_info=True,
                    #  price_info_position='top_left',  # 'top_left', 'top_right', 'bottom_left', 'bottom_right'
                    #  price_info_fontsize=9,
                    #  price_info_bg='#1e222d',
                    #  price_info_text_color='white',

                     # Padding
                     y_padding=0.05  # Percentage padding on y-axis
                     ):
    """
    Plot candlestick chart on given axes with enhanced customization

    Parameters:
    -----------
    ax : matplotlib axis
        The axis to plot on
    df : pandas DataFrame
        OHLCV data with datetime index
    tf : int
        For title only
    ticker : str
        Stock ticker symbol
    up_color : str
        Color for bullish candles (default: '#26a69a')
    down_color : str
        Color for bearish candles (default: '#ef5350')
    edge_color : str or None
        Color for candle body edges. If None, uses body color
    wick_color : str or None
        Color for wicks. If None, uses body color
    volume_color : str
        Color for volume bars (default: '#7cb5ec')
    bg_color : str
        Background color for the plot (default: 'white')
    grid_color : str
        Color for grid lines (default: '#e0e0e0')
    grid_style : str
        Line style for grid ('-', '--', '-.', ':')
    grid_alpha : float
        Transparency of grid lines (0-1)
    show_grid : bool
        Whether to show grid lines
    candle_width : float
        Width of candle bodies as proportion
    date_format : str
        Format string for date labels
    rotation : int
        Rotation angle for x-axis date labels
    show_nontrading : bool
        Whether to show gaps for non-trading periods
    plot_current_price : bool
        Whether to plot current price line
    
    # Font parameters
    title_fontsize : int
        Font size for chart title (default: 14)
    title_fontweight : str
        Font weight for title (default: 'bold')
    title_color : str
        Color for title text (default: 'white')
    label_fontsize : int
        Font size for axis labels (default: 11)
    label_color : str
        Color for axis labels (default: '#d1d4dc')
    tick_fontsize : int
        Font size for tick labels (default: 9)
    tick_color : str
        Color for tick labels (default: '#787b86')
    legend_fontsize : int
        Font size for legend (default: 10)
    
    # Axes styling
    spine_color : str
        Color for axes spines (default: '#2a2e39')
    spine_linewidth : float
        Width of spine lines (default: 1)
    show_top_spine : bool
        Whether to show top spine (default: False)
    show_right_spine : bool
        Whether to show right spine (default: False)
    
    # Price info box
    show_price_info : bool
        Whether to show OHLC info box (default: True)
    price_info_position : str
        Position of price info box (default: 'top_left')
    price_info_fontsize : int
        Font size for price info (default: 9)
    price_info_bg : str
        Background color for price info box (default: '#1e222d')
    price_info_text_color : str
        Text color for price info (default: 'white')
    
    # Layout
    y_padding : float
        Percentage padding on y-axis (default: 0.05)
    """

    # Set background color
    ax.set_facecolor(bg_color)

    # Prepare data
    df = df.copy()

    if show_nontrading:
        # Use actual datetime for x-axis (shows gaps)
        x_values = mdates.date2num(df.index.to_pydatetime())
        ax.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    else:
        # Use integer indices (no gaps)
        x_values = range(len(df))
        # Set custom x-axis labels at intervals
        n_labels = min(10, len(df))
        label_indices = [int(i * len(df) / n_labels) for i in range(n_labels)]
        ax.set_xticks(label_indices)
        ax.set_xticklabels([df.index[i].strftime(date_format) for i in label_indices])

    # First pass: Draw all wicks
    for idx in range(len(df)):
        row = df.iloc[idx]
        x_pos = x_values[idx]
        open_price = float(row['open'])
        close_price = float(row['close'])
        high_price = float(row['high'])
        low_price = float(row['low'])

        # Determine color
        body_color = up_color if close_price >= open_price else down_color
        actual_wick_color = wick_color if wick_color else body_color

        # Draw high-low line (wick)
        ax.plot([x_pos, x_pos], [low_price, high_price],
                color=actual_wick_color, linewidth=1, solid_capstyle='round', zorder=1)

    # Second pass: Draw all bodies on top
    for idx in range(len(df)):
        row = df.iloc[idx]
        x_pos = x_values[idx]
        open_price = float(row['open'])
        close_price = float(row['close'])

        # Determine color
        body_color = up_color if close_price >= open_price else down_color
        actual_edge_color = edge_color if edge_color else body_color

        # Draw open-close rectangle (body)
        height = abs(close_price - open_price)
        bottom = min(open_price, close_price)

        half_width = candle_width / 2
        rect = mpatches.Rectangle((x_pos - half_width, bottom), candle_width, height,
                                   facecolor=body_color, edgecolor=actual_edge_color, zorder=2)
        ax.add_patch(rect)

    # Title with background
    title = ax.set_title(
        f'{ticker.upper()} - {tf}(m) - TIMEZONE:{timezone.upper() if timezone is not None else ""}', 
        fontsize=title_fontsize, 
        fontweight=title_fontweight, 
        color=title_color, 
        pad=10
    )

    # Add semi-transparent background to title
    title.set_bbox(dict(
        facecolor='black',  # Background color
        alpha=0.2,          # Transparency (0-1)
        edgecolor='none',   # No border
        boxstyle='round,pad=0.5'  # Rounded corners with padding
    ))

    # Axis labels
    ax.set_xlabel('Date', fontsize=label_fontsize, color=label_color)
    ax.set_ylabel('Price ($)', fontsize=label_fontsize, color=label_color)

    # Tick parameters
    ax.tick_params(axis='both', labelsize=tick_fontsize, colors=tick_color,
                   length=4, width=0.5)

    # Grid
    if show_grid:
        ax.grid(True, alpha=grid_alpha, linestyle=grid_style, color=grid_color, linewidth=0.5)

    # Spine styling
    for spine in ['top', 'right', 'bottom', 'left']:
        ax.spines[spine].set_color(spine_color)
        ax.spines[spine].set_linewidth(spine_linewidth)

    if not show_top_spine:
        ax.spines['top'].set_visible(False)
    if not show_right_spine:
        ax.spines['right'].set_visible(False)

    # Rotate date labels
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=rotation, ha='right')

    # Set x-axis limits with some padding
    if len(x_values) > 0:
        if show_nontrading:
            date_range = x_values[-1] - x_values[0]
            padding = date_range * 0.02
            ax.set_xlim(x_values[0] - padding, x_values[-1] + padding)
        else:
            ax.set_xlim(-0.5, len(df) - 0.5)

    # Add y-axis padding
    y_min, y_max = df[['low', 'high']].min().min(), df[['low', 'high']].max().max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - y_range * y_padding, y_max + y_range * y_padding)

    return ax