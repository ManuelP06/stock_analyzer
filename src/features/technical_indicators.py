import polars as pl
import numpy as np
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Comprehensive technical indicators calculator using Polars for efficient computation.
    Optimized for deep learning stock analysis with proper handling of missing values.
    """
    
    @staticmethod
    def sma(df: pl.DataFrame, column: str = "close", window: int = 20, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Simple Moving Average"""
        if group_by and group_by in df.columns:
            return df.with_columns(
                pl.col(column).rolling_mean(window).over(group_by).alias(f"sma_{window}")
            )
        else:
            return df.with_columns(
                pl.col(column).rolling_mean(window).alias(f"sma_{window}")
            )
    
    @staticmethod
    def ema(df: pl.DataFrame, column: str = "close", window: int = 20, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Exponential Moving Average"""
        alpha = 2.0 / (window + 1.0)
        
        if group_by and group_by in df.columns:
            return df.with_columns(
                pl.col(column).ewm_mean(alpha=alpha).over(group_by).alias(f"ema_{window}")
            )
        else:
            return df.with_columns(
                pl.col(column).ewm_mean(alpha=alpha).alias(f"ema_{window}")
            )
    
    @staticmethod
    def rsi(df: pl.DataFrame, column: str = "close", window: int = 14, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Relative Strength Index using Wilder's smoothing"""
        alpha = 1.0 / window  # Wilder's smoothing factor
        
        if group_by and group_by in df.columns:
            return df.with_columns([
                # Calculate price changes
                pl.col(column).diff().over(group_by).alias("price_change"),
            ]).with_columns([
                # Separate gains and losses
                pl.when(pl.col("price_change") > 0).then(pl.col("price_change")).otherwise(0).alias("gains"),
                pl.when(pl.col("price_change") < 0).then(-pl.col("price_change")).otherwise(0).alias("losses"),
            ]).with_columns([
                # Calculate exponentially smoothed average gains and losses (Wilder's method)
                pl.col("gains").ewm_mean(alpha=alpha).over(group_by).alias("avg_gains"),
                pl.col("losses").ewm_mean(alpha=alpha).over(group_by).alias("avg_losses"),
            ]).with_columns([
                # Calculate RSI with division by zero protection
                pl.when(pl.col("avg_losses") == 0)
                .then(100)
                .otherwise(100 - (100 / (1 + (pl.col("avg_gains") / pl.col("avg_losses")))))
                .alias(f"rsi_{window}")
            ]).drop(["price_change", "gains", "losses", "avg_gains", "avg_losses"])
        else:
            return df.with_columns([
                pl.col(column).diff().alias("price_change"),
            ]).with_columns([
                pl.when(pl.col("price_change") > 0).then(pl.col("price_change")).otherwise(0).alias("gains"),
                pl.when(pl.col("price_change") < 0).then(-pl.col("price_change")).otherwise(0).alias("losses"),
            ]).with_columns([
                pl.col("gains").ewm_mean(alpha=alpha).alias("avg_gains"),
                pl.col("losses").ewm_mean(alpha=alpha).alias("avg_losses"),
            ]).with_columns([
                pl.when(pl.col("avg_losses") == 0)
                .then(100)
                .otherwise(100 - (100 / (1 + (pl.col("avg_gains") / pl.col("avg_losses")))))
                .alias(f"rsi_{window}")
            ]).drop(["price_change", "gains", "losses", "avg_gains", "avg_losses"])
    
    @staticmethod
    def macd(df: pl.DataFrame, column: str = "close", fast: int = 12, slow: int = 26, signal: int = 9, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """MACD (Moving Average Convergence Divergence)"""
        fast_alpha = 2.0 / (fast + 1.0)
        slow_alpha = 2.0 / (slow + 1.0)
        signal_alpha = 2.0 / (signal + 1.0)
        
        if group_by and group_by in df.columns:
            return df.with_columns([
                pl.col(column).ewm_mean(alpha=fast_alpha).over(group_by).alias("ema_fast"),
                pl.col(column).ewm_mean(alpha=slow_alpha).over(group_by).alias("ema_slow"),
            ]).with_columns([
                (pl.col("ema_fast") - pl.col("ema_slow")).alias("macd_line"),
            ]).with_columns([
                pl.col("macd_line").ewm_mean(alpha=signal_alpha).over(group_by).alias("macd_signal"),
            ]).with_columns([
                (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram")
            ]).drop(["ema_fast", "ema_slow"])
        else:
            return df.with_columns([
                pl.col(column).ewm_mean(alpha=fast_alpha).alias("ema_fast"),
                pl.col(column).ewm_mean(alpha=slow_alpha).alias("ema_slow"),
            ]).with_columns([
                (pl.col("ema_fast") - pl.col("ema_slow")).alias("macd_line"),
            ]).with_columns([
                pl.col("macd_line").ewm_mean(alpha=signal_alpha).alias("macd_signal"),
            ]).with_columns([
                (pl.col("macd_line") - pl.col("macd_signal")).alias("macd_histogram")
            ]).drop(["ema_fast", "ema_slow"])
    
    @staticmethod
    def bollinger_bands(df: pl.DataFrame, column: str = "close", window: int = 20, std_dev: float = 2.0, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Bollinger Bands"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                pl.col(column).rolling_mean(window).over(group_by).alias("bb_middle"),
                pl.col(column).rolling_std(window).over(group_by).alias("bb_std"),
            ]).with_columns([
                (pl.col("bb_middle") + (pl.col("bb_std") * std_dev)).alias("bb_upper"),
                (pl.col("bb_middle") - (pl.col("bb_std") * std_dev)).alias("bb_lower"),
                ((pl.col(column) - pl.col("bb_middle")) / (pl.col("bb_std") * std_dev)).alias("bb_percent"),
            ]).drop("bb_std")
        else:
            return df.with_columns([
                pl.col(column).rolling_mean(window).alias("bb_middle"),
                pl.col(column).rolling_std(window).alias("bb_std"),
            ]).with_columns([
                (pl.col("bb_middle") + (pl.col("bb_std") * std_dev)).alias("bb_upper"),
                (pl.col("bb_middle") - (pl.col("bb_std") * std_dev)).alias("bb_lower"),
                ((pl.col(column) - pl.col("bb_middle")) / (pl.col("bb_std") * std_dev)).alias("bb_percent"),
            ]).drop("bb_std")
    
    @staticmethod
    def atr(df: pl.DataFrame, window: int = 14, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Average True Range"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                # Calculate True Range components
                (pl.col("high") - pl.col("low")).alias("hl"),
                (pl.col("high") - pl.col("close").shift(1).over(group_by)).abs().alias("hc"),
                (pl.col("low") - pl.col("close").shift(1).over(group_by)).abs().alias("lc"),
            ]).with_columns([
                # True Range is the maximum of the three
                pl.max_horizontal(["hl", "hc", "lc"]).alias("true_range")
            ]).with_columns([
                pl.col("true_range").rolling_mean(window).over(group_by).alias(f"atr_{window}")
            ]).drop(["hl", "hc", "lc", "true_range"])
        else:
            return df.with_columns([
                (pl.col("high") - pl.col("low")).alias("hl"),
                (pl.col("high") - pl.col("close").shift(1)).abs().alias("hc"),
                (pl.col("low") - pl.col("close").shift(1)).abs().alias("lc"),
            ]).with_columns([
                pl.max_horizontal(["hl", "hc", "lc"]).alias("true_range")
            ]).with_columns([
                pl.col("true_range").rolling_mean(window).alias(f"atr_{window}")
            ]).drop(["hl", "hc", "lc", "true_range"])
    
    @staticmethod
    def stochastic(df: pl.DataFrame, k_window: int = 14, d_window: int = 3, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Stochastic Oscillator"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                pl.col("low").rolling_min(k_window).over(group_by).alias("lowest_low"),
                pl.col("high").rolling_max(k_window).over(group_by).alias("highest_high"),
            ]).with_columns([
                (100 * (pl.col("close") - pl.col("lowest_low")) / 
                 (pl.col("highest_high") - pl.col("lowest_low"))).alias("stoch_k")
            ]).with_columns([
                pl.col("stoch_k").rolling_mean(d_window).over(group_by).alias("stoch_d")
            ]).drop(["lowest_low", "highest_high"])
        else:
            return df.with_columns([
                pl.col("low").rolling_min(k_window).alias("lowest_low"),
                pl.col("high").rolling_max(k_window).alias("highest_high"),
            ]).with_columns([
                (100 * (pl.col("close") - pl.col("lowest_low")) / 
                 (pl.col("highest_high") - pl.col("lowest_low"))).alias("stoch_k")
            ]).with_columns([
                pl.col("stoch_k").rolling_mean(d_window).alias("stoch_d")
            ]).drop(["lowest_low", "highest_high"])
    
    @staticmethod
    def williams_r(df: pl.DataFrame, window: int = 14, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Williams %R"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                pl.col("high").rolling_max(window).over(group_by).alias("highest_high"),
                pl.col("low").rolling_min(window).over(group_by).alias("lowest_low"),
            ]).with_columns([
                (-100 * (pl.col("highest_high") - pl.col("close")) / 
                 (pl.col("highest_high") - pl.col("lowest_low"))).alias(f"williams_r_{window}")
            ]).drop(["highest_high", "lowest_low"])
        else:
            return df.with_columns([
                pl.col("high").rolling_max(window).alias("highest_high"),
                pl.col("low").rolling_min(window).alias("lowest_low"),
            ]).with_columns([
                (-100 * (pl.col("highest_high") - pl.col("close")) / 
                 (pl.col("highest_high") - pl.col("lowest_low"))).alias(f"williams_r_{window}")
            ]).drop(["highest_high", "lowest_low"])
    
    @staticmethod
    def cci(df: pl.DataFrame, window: int = 20, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Commodity Channel Index using Mean Absolute Deviation"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price")
            ]).with_columns([
                pl.col("typical_price").rolling_mean(window).over(group_by).alias("sma_tp"),
            ]).with_columns([
                # Calculate Mean Absolute Deviation instead of standard deviation
                (pl.col("typical_price") - pl.col("sma_tp")).abs().rolling_mean(window).over(group_by).alias("mad_tp"),
            ]).with_columns([
                # CCI with division by zero protection
                pl.when(pl.col("mad_tp") == 0)
                .then(0)
                .otherwise((pl.col("typical_price") - pl.col("sma_tp")) / (0.015 * pl.col("mad_tp")))
                .alias(f"cci_{window}")
            ]).drop(["typical_price", "sma_tp", "mad_tp"])
        else:
            return df.with_columns([
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price")
            ]).with_columns([
                pl.col("typical_price").rolling_mean(window).alias("sma_tp"),
            ]).with_columns([
                (pl.col("typical_price") - pl.col("sma_tp")).abs().rolling_mean(window).alias("mad_tp"),
            ]).with_columns([
                pl.when(pl.col("mad_tp") == 0)
                .then(0)
                .otherwise((pl.col("typical_price") - pl.col("sma_tp")) / (0.015 * pl.col("mad_tp")))
                .alias(f"cci_{window}")
            ]).drop(["typical_price", "sma_tp", "mad_tp"])
    
    @staticmethod
    def obv(df: pl.DataFrame, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """On-Balance Volume"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                pl.when(pl.col("close") > pl.col("close").shift(1).over(group_by))
                .then(pl.col("volume"))
                .when(pl.col("close") < pl.col("close").shift(1).over(group_by))
                .then(-pl.col("volume"))
                .otherwise(0).alias("obv_change")
            ]).with_columns([
                pl.col("obv_change").cum_sum().over(group_by).alias("obv")
            ]).drop("obv_change")
        else:
            return df.with_columns([
                pl.when(pl.col("close") > pl.col("close").shift(1))
                .then(pl.col("volume"))
                .when(pl.col("close") < pl.col("close").shift(1))
                .then(-pl.col("volume"))
                .otherwise(0).alias("obv_change")
            ]).with_columns([
                pl.col("obv_change").cum_sum().alias("obv")
            ]).drop("obv_change")
    
    @staticmethod
    def mfi(df: pl.DataFrame, window: int = 14, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Money Flow Index"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price")
            ]).with_columns([
                (pl.col("typical_price") * pl.col("volume")).alias("money_flow"),
                pl.col("typical_price").diff().over(group_by).alias("price_change")
            ]).with_columns([
                pl.when(pl.col("price_change") > 0).then(pl.col("money_flow")).otherwise(0).alias("positive_mf"),
                pl.when(pl.col("price_change") < 0).then(pl.col("money_flow")).otherwise(0).alias("negative_mf")
            ]).with_columns([
                pl.col("positive_mf").rolling_sum(window).over(group_by).alias("positive_mf_sum"),
                pl.col("negative_mf").rolling_sum(window).over(group_by).alias("negative_mf_sum")
            ]).with_columns([
                # ENHANCED: Handle both zero division and NaN values
                pl.when(pl.col("negative_mf_sum") <= 1e-8)
                .then(50.0)  # Neutral MFI when no negative money flow
                .otherwise(
                    100 - (100 / (1 + (pl.col("positive_mf_sum") / pl.col("negative_mf_sum"))))
                )
                .clip(0, 100)  # Ensure MFI stays within valid range
                .alias(f"mfi_{window}")
            ]).drop(["typical_price", "money_flow", "price_change", "positive_mf", "negative_mf", "positive_mf_sum", "negative_mf_sum"])
        else:
            return df.with_columns([
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price")
            ]).with_columns([
                (pl.col("typical_price") * pl.col("volume")).alias("money_flow"),
                pl.col("typical_price").diff().alias("price_change")
            ]).with_columns([
                pl.when(pl.col("price_change") > 0).then(pl.col("money_flow")).otherwise(0).alias("positive_mf"),
                pl.when(pl.col("price_change") < 0).then(pl.col("money_flow")).otherwise(0).alias("negative_mf")
            ]).with_columns([
                pl.col("positive_mf").rolling_sum(window).alias("positive_mf_sum"),
                pl.col("negative_mf").rolling_sum(window).alias("negative_mf_sum")
            ]).with_columns([
                # ENHANCED: Handle both zero division and NaN values
                pl.when(pl.col("negative_mf_sum") <= 1e-8)
                .then(50.0)  # Neutral MFI when no negative money flow
                .otherwise(
                    100 - (100 / (1 + (pl.col("positive_mf_sum") / pl.col("negative_mf_sum"))))
                )
                .clip(0, 100)  # Ensure MFI stays within valid range
                .alias(f"mfi_{window}")
            ]).drop(["typical_price", "money_flow", "price_change", "positive_mf", "negative_mf", "positive_mf_sum", "negative_mf_sum"])
    
    @staticmethod
    def vwap(df: pl.DataFrame, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Volume Weighted Average Price"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price")
            ]).with_columns([
                (pl.col("typical_price") * pl.col("volume")).alias("tp_volume")
            ]).with_columns([
                (pl.col("tp_volume").cum_sum().over(group_by) / pl.col("volume").cum_sum().over(group_by)).alias("vwap")
            ]).drop(["typical_price", "tp_volume"])
        else:
            return df.with_columns([
                ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price")
            ]).with_columns([
                (pl.col("typical_price") * pl.col("volume")).alias("tp_volume")
            ]).with_columns([
                (pl.col("tp_volume").cum_sum() / pl.col("volume").cum_sum()).alias("vwap")
            ]).drop(["typical_price", "tp_volume"])
    
    @staticmethod
    def roc(df: pl.DataFrame, column: str = "close", window: int = 10, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Rate of Change with division by zero protection"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                pl.when(pl.col(column).shift(window).over(group_by) == 0)
                .then(0)
                .otherwise(100 * (pl.col(column) - pl.col(column).shift(window).over(group_by)) / 
                          pl.col(column).shift(window).over(group_by))
                .alias(f"roc_{window}")
            ])
        else:
            return df.with_columns([
                pl.when(pl.col(column).shift(window) == 0)
                .then(0)
                .otherwise(100 * (pl.col(column) - pl.col(column).shift(window)) / 
                          pl.col(column).shift(window))
                .alias(f"roc_{window}")
            ])
    
    @staticmethod
    def adx(df: pl.DataFrame, window: int = 14, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Average Directional Index (ADX) for trend strength"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                # Calculate True Range components
                (pl.col("high") - pl.col("low")).alias("hl"),
                (pl.col("high") - pl.col("close").shift(1).over(group_by)).abs().alias("hc"),
                (pl.col("low") - pl.col("close").shift(1).over(group_by)).abs().alias("lc"),
                # Calculate directional moves
                (pl.col("high") - pl.col("high").shift(1).over(group_by)).alias("up_move"),
                (pl.col("low").shift(1).over(group_by) - pl.col("low")).alias("down_move"),
            ]).with_columns([
                # True Range
                pl.max_horizontal(["hl", "hc", "lc"]).alias("tr"),
                # Positive and Negative Directional Indicators
                pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0))
                .then(pl.col("up_move")).otherwise(0).alias("plus_dm"),
                pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0))
                .then(pl.col("down_move")).otherwise(0).alias("minus_dm"),
            ]).with_columns([
                # Smooth the TR, +DM, and -DM using Wilder's smoothing
                pl.col("tr").ewm_mean(alpha=1.0/window).over(group_by).alias("atr_smooth"),
                pl.col("plus_dm").ewm_mean(alpha=1.0/window).over(group_by).alias("plus_di_smooth"),
                pl.col("minus_dm").ewm_mean(alpha=1.0/window).over(group_by).alias("minus_di_smooth"),
            ]).with_columns([
                # Calculate +DI and -DI
                pl.when(pl.col("atr_smooth") == 0)
                .then(0)
                .otherwise(100 * pl.col("plus_di_smooth") / pl.col("atr_smooth"))
                .alias("plus_di"),
                pl.when(pl.col("atr_smooth") == 0)
                .then(0)
                .otherwise(100 * pl.col("minus_di_smooth") / pl.col("atr_smooth"))
                .alias("minus_di"),
            ]).with_columns([
                # Calculate DX
                pl.when((pl.col("plus_di") + pl.col("minus_di")) == 0)
                .then(0)
                .otherwise(100 * (pl.col("plus_di") - pl.col("minus_di")).abs() / 
                          (pl.col("plus_di") + pl.col("minus_di")))
                .alias("dx")
            ]).with_columns([
                # Calculate ADX using Wilder's smoothing on DX
                pl.col("dx").ewm_mean(alpha=1.0/window).over(group_by).alias(f"adx_{window}")
            ]).drop(["hl", "hc", "lc", "up_move", "down_move", "tr", "plus_dm", "minus_dm", 
                    "atr_smooth", "plus_di_smooth", "minus_di_smooth", "plus_di", "minus_di", "dx"])
        else:
            return df.with_columns([
                (pl.col("high") - pl.col("low")).alias("hl"),
                (pl.col("high") - pl.col("close").shift(1)).abs().alias("hc"),
                (pl.col("low") - pl.col("close").shift(1)).abs().alias("lc"),
                (pl.col("high") - pl.col("high").shift(1)).alias("up_move"),
                (pl.col("low").shift(1) - pl.col("low")).alias("down_move"),
            ]).with_columns([
                pl.max_horizontal(["hl", "hc", "lc"]).alias("tr"),
                pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0))
                .then(pl.col("up_move")).otherwise(0).alias("plus_dm"),
                pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0))
                .then(pl.col("down_move")).otherwise(0).alias("minus_dm"),
            ]).with_columns([
                pl.col("tr").ewm_mean(alpha=1.0/window).alias("atr_smooth"),
                pl.col("plus_dm").ewm_mean(alpha=1.0/window).alias("plus_di_smooth"),
                pl.col("minus_dm").ewm_mean(alpha=1.0/window).alias("minus_di_smooth"),
            ]).with_columns([
                pl.when(pl.col("atr_smooth") == 0)
                .then(0)
                .otherwise(100 * pl.col("plus_di_smooth") / pl.col("atr_smooth"))
                .alias("plus_di"),
                pl.when(pl.col("atr_smooth") == 0)
                .then(0)
                .otherwise(100 * pl.col("minus_di_smooth") / pl.col("atr_smooth"))
                .alias("minus_di"),
            ]).with_columns([
                pl.when((pl.col("plus_di") + pl.col("minus_di")) == 0)
                .then(0)
                .otherwise(100 * (pl.col("plus_di") - pl.col("minus_di")).abs() / 
                          (pl.col("plus_di") + pl.col("minus_di")))
                .alias("dx")
            ]).with_columns([
                pl.col("dx").ewm_mean(alpha=1.0/window).alias(f"adx_{window}")
            ]).drop(["hl", "hc", "lc", "up_move", "down_move", "tr", "plus_dm", "minus_dm", 
                    "atr_smooth", "plus_di_smooth", "minus_di_smooth", "plus_di", "minus_di", "dx"])
    
    @staticmethod
    def donchian_channels(df: pl.DataFrame, window: int = 20, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """Donchian Channels for support/resistance levels"""
        if group_by and group_by in df.columns:
            return df.with_columns([
                pl.col("high").rolling_max(window).over(group_by).alias(f"donchian_upper_{window}"),
                pl.col("low").rolling_min(window).over(group_by).alias(f"donchian_lower_{window}"),
            ]).with_columns([
                ((pl.col(f"donchian_upper_{window}") + pl.col(f"donchian_lower_{window}")) / 2)
                .alias(f"donchian_middle_{window}")
            ])
        else:
            return df.with_columns([
                pl.col("high").rolling_max(window).alias(f"donchian_upper_{window}"),
                pl.col("low").rolling_min(window).alias(f"donchian_lower_{window}"),
            ]).with_columns([
                ((pl.col(f"donchian_upper_{window}") + pl.col(f"donchian_lower_{window}")) / 2)
                .alias(f"donchian_middle_{window}")
            ])
    
    @staticmethod
    def add_all_indicators(df: pl.DataFrame, group_by: Optional[str] = "symbol") -> pl.DataFrame:
        """
        Add all technical indicators to the DataFrame.
        Optimized for deep learning with multiple timeframes.
        """
        logger.info("Adding comprehensive technical indicators")
        
        # Start with the original dataframe
        result_df = df.clone()
        
        # Moving Averages (multiple timeframes)
        for window in [5, 10, 20, 50, 200]:
            result_df = TechnicalIndicators.sma(result_df, window=window, group_by=group_by)
            result_df = TechnicalIndicators.ema(result_df, window=window, group_by=group_by)
        
        # RSI (multiple timeframes)
        for window in [14, 21]:
            result_df = TechnicalIndicators.rsi(result_df, window=window, group_by=group_by)
        
        # MACD
        result_df = TechnicalIndicators.macd(result_df, group_by=group_by)
        
        # Bollinger Bands
        result_df = TechnicalIndicators.bollinger_bands(result_df, group_by=group_by)
        
        # Volatility indicators
        for window in [14, 21]:
            result_df = TechnicalIndicators.atr(result_df, window=window, group_by=group_by)
        
        # Trend strength
        result_df = TechnicalIndicators.adx(result_df, group_by=group_by)
        
        # Support/Resistance
        for window in [20, 50]:
            result_df = TechnicalIndicators.donchian_channels(result_df, window=window, group_by=group_by)
        
        # Oscillators
        result_df = TechnicalIndicators.stochastic(result_df, group_by=group_by)
        result_df = TechnicalIndicators.williams_r(result_df, group_by=group_by)
        result_df = TechnicalIndicators.cci(result_df, group_by=group_by)
        
        # Volume indicators (if volume data exists)
        if "volume" in df.columns:
            result_df = TechnicalIndicators.obv(result_df, group_by=group_by)
            result_df = TechnicalIndicators.mfi(result_df, group_by=group_by)
            result_df = TechnicalIndicators.vwap(result_df, group_by=group_by)
        
        # Rate of Change (multiple timeframes)
        for window in [5, 10, 20]:
            result_df = TechnicalIndicators.roc(result_df, window=window, group_by=group_by)
        
        # Add price ratios and relationships
        result_df = result_df.with_columns([
            # Price position within high-low range
            ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias("price_position"),
            
            # High-Low spread
            ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("hl_spread_pct"),
            
            # Open-Close spread  
            ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("oc_spread_pct"),
        ])
        
        # Add lagged features for time series patterns
        if group_by and group_by in df.columns:
            for lag in [1, 2, 3, 5]:
                lag_columns = [pl.col("close").shift(lag).over(group_by).alias(f"close_lag_{lag}")]
                if "volume" in df.columns:
                    lag_columns.append(pl.col("volume").shift(lag).over(group_by).alias(f"volume_lag_{lag}"))
                result_df = result_df.with_columns(lag_columns)
        else:
            for lag in [1, 2, 3, 5]:
                lag_columns = [pl.col("close").shift(lag).alias(f"close_lag_{lag}")]
                if "volume" in df.columns:
                    lag_columns.append(pl.col("volume").shift(lag).alias(f"volume_lag_{lag}"))
                result_df = result_df.with_columns(lag_columns)
        
        logger.info(f"Added {len(result_df.columns) - len(df.columns)} technical indicators")
        return result_df


def main():
    import argparse
    import sys
    from pathlib import Path
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Technical Indicators Calculator')
    parser.add_argument('--input', '-i', type=str, required=True, help='Input data file path')
    parser.add_argument('--output', '-o', type=str, help='Output file path (optional)')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to process')
    parser.add_argument('--indicators', nargs='+', 
                       choices=['sma', 'ema', 'rsi', 'macd', 'bollinger', 'atr', 'stochastic', 
                               'williams', 'cci', 'obv', 'mfi', 'vwap', 'roc', 'adx', 'donchian', 'all'],
                       default=['all'], help='Indicators to calculate')
    parser.add_argument('--no-group', action='store_true', help='Do not group by symbol')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level)
    
    try:
        # Load data
        input_path = Path(args.input)
        if not input_path.exists():
            logger.error(f"Input file not found: {args.input}")
            sys.exit(1)
        
        if input_path.suffix.lower() == '.parquet':
            df = pl.read_parquet(input_path)
        elif input_path.suffix.lower() == '.csv':
            df = pl.read_csv(input_path)
        else:
            logger.error(f"Unsupported file format: {input_path.suffix}")
            sys.exit(1)
        
        logger.info(f"Loaded data: {df.shape}")
        
        # Filter symbols if specified
        if args.symbols and "symbol" in df.columns:
            df = df.filter(pl.col("symbol").is_in(args.symbols))
            logger.info(f"Filtered to symbols: {args.symbols}")
        
        # Determine grouping
        group_by = None if args.no_group else "symbol" if "symbol" in df.columns else None
        
        # Calculate indicators
        if 'all' in args.indicators:
            df_with_indicators = TechnicalIndicators.add_all_indicators(df, group_by=group_by)
        else:
            df_with_indicators = df.clone()
            for indicator in args.indicators:
                if indicator == 'sma':
                    for window in [10, 20, 50]:
                        df_with_indicators = TechnicalIndicators.sma(df_with_indicators, window=window, group_by=group_by)
                elif indicator == 'ema':
                    for window in [12, 26, 50]:
                        df_with_indicators = TechnicalIndicators.ema(df_with_indicators, window=window, group_by=group_by)
                elif indicator == 'rsi':
                    df_with_indicators = TechnicalIndicators.rsi(df_with_indicators, group_by=group_by)
                elif indicator == 'macd':
                    df_with_indicators = TechnicalIndicators.macd(df_with_indicators, group_by=group_by)
                elif indicator == 'bollinger':
                    df_with_indicators = TechnicalIndicators.bollinger_bands(df_with_indicators, group_by=group_by)
                elif indicator == 'atr':
                    df_with_indicators = TechnicalIndicators.atr(df_with_indicators, group_by=group_by)
                elif indicator == 'stochastic':
                    df_with_indicators = TechnicalIndicators.stochastic(df_with_indicators, group_by=group_by)
                elif indicator == 'williams':
                    df_with_indicators = TechnicalIndicators.williams_r(df_with_indicators, group_by=group_by)
                elif indicator == 'cci':
                    df_with_indicators = TechnicalIndicators.cci(df_with_indicators, group_by=group_by)
                elif indicator == 'obv' and "volume" in df.columns:
                    df_with_indicators = TechnicalIndicators.obv(df_with_indicators, group_by=group_by)
                elif indicator == 'mfi' and "volume" in df.columns:
                    df_with_indicators = TechnicalIndicators.mfi(df_with_indicators, group_by=group_by)
                elif indicator == 'vwap' and "volume" in df.columns:
                    df_with_indicators = TechnicalIndicators.vwap(df_with_indicators, group_by=group_by)
                elif indicator == 'roc':
                    for window in [5, 10, 20]:
                        df_with_indicators = TechnicalIndicators.roc(df_with_indicators, window=window, group_by=group_by)
                elif indicator == 'adx':
                    df_with_indicators = TechnicalIndicators.adx(df_with_indicators, group_by=group_by)
                elif indicator == 'donchian':
                    for window in [20, 50]:
                        df_with_indicators = TechnicalIndicators.donchian_channels(df_with_indicators, window=window, group_by=group_by)
        
        # Save output
        if args.output:
            output_path = Path(args.output)
            if output_path.suffix.lower() == '.parquet':
                df_with_indicators.write_parquet(output_path)
            elif output_path.suffix.lower() == '.csv':
                df_with_indicators.write_csv(output_path)
            else:
                # Default to parquet
                df_with_indicators.write_parquet(output_path.with_suffix('.parquet'))
            logger.info(f"Saved indicators to {output_path}")
        else:
            # Display results
            print("\nDataFrame with Technical Indicators:")
            print(df_with_indicators)
            print(f"\nShape: {df_with_indicators.shape}")
            print(f"New columns added: {len(df_with_indicators.columns) - len(df.columns)}")
        
    except Exception as e:
        logger.error(f"Calculation failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Example use

Calculate all indicators
python src/features/technical_indicators.py -i data/processed/stocks.parquet -o data/features/with_indicators.parquet

Calculate specific indicators
python src/features/technical_indicators.py -i data.csv --indicators rsi macd bollinger --verbose

Process specific symbols
python src/features/technical_indicators.py -i data.parquet --symbols AAPL MSFT --indicators all"""