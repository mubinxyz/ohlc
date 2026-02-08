# utils/get_and_clean_data.py

import re
from datetime import timezone, timedelta, datetime
import pandas as pd
import time
from .get_data import get_ohlc


def get_and_clean_data(
        date_range: bool=False,
        from_date_str: str=None,
        to_date_str: str=None,
        ohlc_tz_str: str='utc+3:30',
        output_candles: int=85,
        tf: int=15,
        asset: str='eurusd',
):
    """Fetch and clean OHLC data from configured source."""

    def _parse_date_to_unix(date_str: str, timezone_str: str = "utc") -> int:
        """
        Parse flexible date string formats to Unix timestamp.
        
        Supported formats:
        - "2024-03-06 01:34:00" (full datetime)
        - "2024-03-06 01:34" (no seconds, defaults to :00)
        - "2024-03-06" (date only, defaults to 00:00:00)
        - "2024/03/06 01:34:00" (alternative separator)
        """
        if not date_str:
            return None

        # Parse timezone offset
        def parse_tz_offset(tz_str: str) -> timezone:
            match = re.match(r'utc([+-])(\d+)(?::(\d+))?', tz_str.lower())
            if not match:
                return timezone.utc

            sign = 1 if match.group(1) == '+' else -1
            hours = int(match.group(2))
            minutes = int(match.group(3)) if match.group(3) else 0

            return timezone(timedelta(hours=sign * hours, minutes=sign * minutes))

        tz = parse_tz_offset(timezone_str)

        # Try different datetime formats
        formats = [
            "%Y-%m-%d %H:%M:%S",  # 2024-03-06 01:34:00
            "%Y-%m-%d %H:%M",     # 2024-03-06 01:34
            "%Y-%m-%d",           # 2024-03-06
            "%Y/%m/%d %H:%M:%S",  # 2024/03/06 01:34:00
            "%Y/%m/%d %H:%M",     # 2024/03/06 01:34
            "%Y/%m/%d",           # 2024/03/06
        ]

        parsed_dt = None
        for fmt in formats:
            try:
                parsed_dt = datetime.strptime(date_str.strip(), fmt)
                break
            except ValueError:
                continue

        if parsed_dt is None:
            raise ValueError(
                f"Unable to parse date string: '{date_str}'. "
                f"Supported formats: YYYY-MM-DD [HH:MM[:SS]]"
            )

        # Localize to specified timezone and convert to Unix timestamp
        localized_dt = parsed_dt.replace(tzinfo=tz)
        unix_timestamp = int(localized_dt.timestamp())

        return unix_timestamp

    def _parse_timezone_offset(tz_str: str) -> tuple:
        """Parse timezone string like 'utc+3:30' or 'utc-6' into (hours, minutes)."""
        match = re.match(r'utc([+-])(\d+)(?::(\d+))?', tz_str.lower())
        if not match:
            raise ValueError(f"Invalid timezone format: {tz_str}. Expected format: 'utc+H:MM' or 'utc-H:MM'")

        sign = 1 if match.group(1) == '+' else -1
        hours = int(match.group(2))
        minutes = int(match.group(3)) if match.group(3) else 0

        return sign * hours, sign * minutes

    # Determine if we're using date range mode or lookback mode
    use_date_range = date_range and from_date_str and from_date_str.strip()
    
    if use_date_range:
        # DATE RANGE MODE: use specified dates, ignore output_candles completely
        from_date = _parse_date_to_unix(
            date_str=from_date_str, 
            timezone_str=ohlc_tz_str,
        )

        if to_date_str and to_date_str.strip():
            to_date = _parse_date_to_unix(
                date_str=to_date_str,
                timezone_str=ohlc_tz_str,
            )
        else:
            # Default to current time if to_date not specified
            to_date = int(time.time())

        duration_minutes = (to_date - from_date) / 60
        estimated_candles = int(duration_minutes / tf)
        
        print(f"[get_and_clean_data] ✓ DATE RANGE MODE ACTIVE")
        print(f"  From: {from_date_str} → Unix: {from_date}")
        print(f"  To: {to_date_str if to_date_str else 'now'} → Unix: {to_date}")
        print(f"  Duration: {duration_minutes:.0f} minutes ({duration_minutes/60:.1f} hours)")
        print(f"  Estimated candles @ {tf}min: {estimated_candles}")

    else:
        # LOOKBACK MODE: use output_candles to calculate time range
        # This happens when:
        # - date_range=False, OR
        # - date_range=True but from_date_str is empty/None
        lookback_seconds = output_candles * tf * 60
        from_date = int(time.time()) - int(lookback_seconds)
        to_date = int(time.time())

        print(f"[get_and_clean_data] ✓ LOOKBACK MODE ACTIVE")
        print(f"  Requested: {output_candles} candles")
        print(f"  Timeframe: {tf} minutes")
        print(f"  Lookback: {lookback_seconds/60:.0f} minutes ({lookback_seconds/3600:.1f} hours)")

    resample_map = {
        "1": "1min",
        "2": "2min",
        "3": "3min",
        "5": "5min",
        "10": "10min",
        "15": "15min",
        "20": "20min",
        "30": "30min",
        "60": "1h",
        "120": "2h",
        "240": "4h",
        "360": "6h",
        "1440": 'D',
    }

    tf_str = str(tf)

    if tf_str in ['2', '3']:
        fetch_tf = 1  
    elif tf_str in ['10', '20']:
        fetch_tf = 5  
    elif tf_str == '30':
        fetch_tf = 15  
    elif tf_str in ['120', '240', '360']:
        fetch_tf = 60  
    else:
        fetch_tf = tf  

    # FIX: get_ohlc already handles chunking automatically for >10k candles
    print(f"[get_and_clean_data] Fetching data: from={from_date}, to={to_date}, tf={fetch_tf}")
    
    asset_df_raw = get_ohlc(
        symbol=asset,
        timeframe=fetch_tf,
        from_date=from_date,
        to_date=to_date,
    )

    if asset_df_raw.empty:
        print("[get_and_clean_data] WARNING: No data returned from API")
        return pd.DataFrame()

    asset_df_raw = asset_df_raw.dropna()

    # Apply timezone conversion
    tz_hours, tz_minutes = _parse_timezone_offset(ohlc_tz_str)
    target_tz = timezone(timedelta(hours=tz_hours, minutes=tz_minutes))

    # Check if datetime is in index or columns
    if 'datetime' in asset_df_raw.columns:
        datetime_series = asset_df_raw['datetime']
        has_tz = datetime_series.dt.tz is not None

        if has_tz:
            datetime_series = datetime_series.dt.tz_localize(None)

        datetime_series = datetime_series.dt.tz_localize('UTC').dt.tz_convert(target_tz)

        asset_df_raw['datetime'] = datetime_series
        asset_df_raw = asset_df_raw.set_index('datetime')
    else:
        datetime_index = asset_df_raw.index
        has_tz = datetime_index.tz is not None

        if has_tz:
            datetime_index = datetime_index.tz_localize(None)

        datetime_index = datetime_index.tz_localize('UTC').tz_convert(target_tz)

        asset_df_raw.index = datetime_index

    # Resample to entry_tf if needed
    if fetch_tf != tf:
        asset_df = asset_df_raw.resample(resample_map[tf_str]).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()
    else:
        asset_df = asset_df_raw

    df = asset_df

    print(f"[get_and_clean_data] Loaded {len(df)} candles")
    if not df.empty:
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
    
    return df