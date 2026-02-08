# utils/get_data.py

import requests
import pandas as pd
import time
from .normalize_data import normalize_symbol, normalize_timeframe, normalize_ohlc, to_unix_timestamp
import json

def get_ohlc(symbol: str, timeframe: int = 15, from_date: int = None, to_date: int = None) -> pd.DataFrame:
    """
    Get OHLC candles for a symbol.
    Automatically handles requests exceeding 10,000 candles by making multiple API calls.
    
    Args:
        symbol: Trading symbol (e.g., 'EURUSD')
        timeframe: Timeframe in minutes (e.g., 1, 5, 15, 60)
        from_date: Start timestamp (Unix seconds)
        to_date: End timestamp (Unix seconds)
    
    Returns:
        DataFrame with OHLC data
    """
    norm_symbol = normalize_symbol(symbol)
    norm_timeframe = normalize_timeframe(timeframe)
    to_date = int(to_date) if to_date is not None else int(time.time())
    from_date = int(from_date) if from_date is not None else 0

    # Calculate total candles requested
    timeframe_seconds = timeframe * 60
    total_seconds = to_date - from_date
    estimated_candles = total_seconds // timeframe_seconds

    # API limit
    MAX_CANDLES_PER_REQUEST = 10000

    # If estimated candles <= 10000, make single request
    if estimated_candles <= MAX_CANDLES_PER_REQUEST:
        return _fetch_single_ohlc(norm_symbol, norm_timeframe, from_date, to_date)

    # Otherwise, split into multiple requests
    print(f"[get_ohlc] Estimated {estimated_candles} candles needed. Splitting into multiple requests...")

    # Calculate number of chunks needed
    num_chunks = (estimated_candles + MAX_CANDLES_PER_REQUEST - 1) // MAX_CANDLES_PER_REQUEST
    chunk_duration = MAX_CANDLES_PER_REQUEST * timeframe_seconds

    all_dfs = []
    current_from = from_date

    for chunk_idx in range(num_chunks):
        # Calculate end time for this chunk
        current_to = min(current_from + chunk_duration, to_date)

        print(f"[get_ohlc] Fetching chunk {chunk_idx + 1}/{num_chunks} "
              f"(from {current_from} to {current_to})...")

        # Fetch chunk
        chunk_df = _fetch_single_ohlc(norm_symbol, norm_timeframe, current_from, current_to)

        if not chunk_df.empty:
            all_dfs.append(chunk_df)
            print(f"[get_ohlc] Chunk {chunk_idx + 1} added: {len(chunk_df)} candles")

        # Move to next chunk (add 1 second to avoid overlap)
        current_from = current_to + 1

        # Break if we've reached the end
        if current_to >= to_date:
            break

        # Small delay to avoid rate limiting
        if chunk_idx < num_chunks - 1:
            time.sleep(0.2)

    # Merge all chunks
    if not all_dfs:
        print("[get_ohlc] No data fetched from any chunk.")
        return pd.DataFrame()

    print(f"[get_ohlc] Merging {len(all_dfs)} chunks...")
    print(f"[get_ohlc] Chunks sizes: {[len(df) for df in all_dfs]}")

    # Concatenate all dataframes
    merged_df = pd.concat(all_dfs, ignore_index=False)
    print(f"[get_ohlc] After concat: {len(merged_df)} candles")

    # Check if index is datetime
    if not isinstance(merged_df.index, pd.DatetimeIndex):
        print("[get_ohlc] Warning: Index is not DatetimeIndex, attempting to fix...")
        if 'datetime' in merged_df.columns:
            merged_df = merged_df.set_index('datetime')

    # Remove duplicates based on datetime index
    initial_len = len(merged_df)
    merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    duplicates_removed = initial_len - len(merged_df)

    if duplicates_removed > 0:
        print(f"[get_ohlc] Removed {duplicates_removed} duplicate timestamps")

    # Sort by datetime
    merged_df = merged_df.sort_index()

    print(f"[get_ohlc] Successfully fetched {len(merged_df)} unique candles total.")

    return merged_df


def _fetch_single_ohlc(norm_symbol: str, norm_timeframe: int, from_date: int, to_date: int) -> pd.DataFrame:
    """
    Internal function to fetch a single chunk of OHLC data (max 10,000 candles).
    
    Args:
        norm_symbol: Normalized symbol
        norm_timeframe: Normalized timeframe
        from_date: Start timestamp
        to_date: End timestamp
    
    Returns:
        DataFrame with OHLC data
    """
    try:
        lite_finance_url = (
            "https://lfdata.pmobint.workers.dev/"
            f"?symbol={norm_symbol}&tf={norm_timeframe}&from={from_date}&to={to_date}"
        )   

        resp = requests.get(lite_finance_url, timeout=300)
        print(f"[_fetch_single_ohlc] URL: {lite_finance_url}")
        print(f"[_fetch_single_ohlc] Status: {resp.status_code}")

        resp.raise_for_status()
        data = resp.json()
        ohlc_data = data.get("data", {})

        if ohlc_data:
            df = normalize_ohlc(ohlc_data)
            print(f"[_fetch_single_ohlc] Fetched {len(df)} candles from API")
            return df
        else:
            print("[_fetch_single_ohlc] No data returned from API.")
            return pd.DataFrame()

    except Exception as e:
        print(f"[_fetch_single_ohlc] OHLC error for {norm_symbol}: {e}")
        return pd.DataFrame()


def get_price(symbol: str) -> dict | None:
    """
    Get the latest price of a symbol via Cloudflare Worker.
    """
    norm_symbol = normalize_symbol(symbol)
    try:
        # use Worker endpoint
        worker_url = f"https://lfdata.pmobint.workers.dev/?symbol={norm_symbol}&tf=1&from=0&to={int(time.time())}"
        resp = requests.get(worker_url, timeout=300)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        if not data:
            return None

        # Use last candle close as latest price
        price = data["c"][-1] if "c" in data and data["c"] else None
        if price is not None:
            return {
                "source": "litefinance worker",
                "symbol": norm_symbol,
                "price": float(price),
                "bid": float(price),
                "ask": float(price)
            }
    except Exception as e:
        print(f"[LiteFinance Worker] Error fetching price: {e}")
    return None