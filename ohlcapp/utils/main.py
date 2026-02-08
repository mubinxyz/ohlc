# utils/main.py

from .get_and_clean_data import get_and_clean_data
from .plot_candlestick import plot_candlestick

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
matplotlib.use('Agg')
from matplotlib.transforms import blended_transform_factory


# FIX #3: Add ohlc_tz to DEFAULT_CONFIG (will be set dynamically in views.py)
DEFAULT_CONFIG = {
    # === plot_candlestick ===
    "up_color": "#fbc02d",
    "down_color": "#9598a1",
    "edge_color": None,
    "wick_color": None,
    "volume_color": "#7cb5ec",
    "bg_color": "#131722",
    "grid_color": "#e0e0e0",
    "grid_style": "--",
    "grid_alpha": 0.3,
    "show_grid": False,
    "candle_width": 0.6,
    "date_format": "%m/%d %H:%M",
    "rotation": 45,
    "show_nontrading": False,
    "plot_current_price": False,
    # Font and text styling
    "title_fontsize": 14,
    "title_fontweight": "bold",
    "title_color": "white",
    "label_fontsize": 11,
    "label_color": "#d1d4dc",
    "tick_fontsize": 9,
    "tick_color": "#787b86",
    "legend_fontsize": 10,
    # Axes styling
    "spine_color": "#2a2e39",
    "spine_linewidth": 1,
    "show_top_spine": False,
    "show_right_spine": False,
    # Layout
    "y_padding": 0.05,
    "show_legend": False,
    "right_margin": 8,
    # FIX #3: Add timezone to config (will be set dynamically)
    "ohlc_tz": "utc+3:30",  # Default value
}


def download_csv():
    # code yourself
    pass


def plot_strategy(
    auto_figsize: bool=True,
    figsize: tuple=(16, 9),
    width_per_candle: float=0.12,
    min_width: float=12,
    max_width: float=300,
    height: float=9,
    df=None,
    output_candles=None,
    tf=None,
    asset=None,
    ohlc_tz=None,  # FIX #3: Add timezone parameter
):
    def _calculate_figsize():
        """
        Calculate optimal figsize for output based on number of candles.
        Width scales with candle count, height stays reasonable.
        
        Returns:
            tuple: (width, height) in inches
        """
        if not auto_figsize:
            # Use manual figsize from config
            return figsize

        # Calculate width based on actual data length
        num_candles = len(df) if df is not None else output_candles
        width = num_candles * width_per_candle
        width = max(min_width, min(max_width, width))

        return (width, height)

    # plotting functions
    def _plot_last_price_line(ax, df, asset):
        """Draw TradingView-style last price line starting from last candle."""
        if df is None or df.empty:
            return

        last_price = float(df['close'].iloc[-1])

        # IMPORTANT: x-axis is integer-based
        last_x = len(df) - 1

        # Axis limits (right margin already added by caller)
        x_left, x_right = ax.get_xlim()

        # Extend line to the right edge
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
        if np.abs(last_price) >= 100:
            price_txt = f"{last_price:,.2f}"
        elif np.abs(last_price) >= 1:
            price_txt = f"{last_price:,.4f}"
        else:
            price_txt = f"{last_price:,.6f}"

        # Label at the right edge
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

        # Datetime is the INDEX
        last_datetime = df.index[-1]

        # INTEGER x-axis
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

    # === MAIN PLOTTING LOGIC ===

    # Calculate optimal figsize
    figsize = _calculate_figsize()

    rows = 1
    cols = 1

    fig, axes = plt.subplots(
        rows, cols,
        figsize=figsize,  # Use calculated figsize
        facecolor=DEFAULT_CONFIG['bg_color'],
    )

    # FIX #3: Use passed timezone or fallback to config
    timezone_to_use = ohlc_tz if ohlc_tz is not None else DEFAULT_CONFIG["ohlc_tz"]

    plot_candlestick(
        ax=axes,  # FIX: Single axis, not axes[0]
        df=df,
        tf=tf,
        ticker=asset,
        timezone=timezone_to_use,
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

    xlim = axes.get_xlim()
    right_margin = DEFAULT_CONFIG['right_margin']
    axes.set_xlim(xlim[0], xlim[1] + right_margin)

    # Add all overlay elements
    _plot_last_price_line(axes, df, asset)
    _plot_last_datetime_info(axes, df, asset)

    plt.tight_layout()
    return fig