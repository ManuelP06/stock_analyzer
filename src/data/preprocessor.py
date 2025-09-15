import polars as pl
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Union
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import warnings
import argparse
import sys
from pathlib import Path
import pandas as pd

logger = logging.getLogger(__name__)


class StockDataPreprocessor:
    """
    Preprocesses raw stock data.
    Handles missing values, outliers, scaling, and creates ML-ready datasets.
    """
    
    def __init__(self, processed_data_dir: str = "../../data/processed"):
        """
        Initialize the preprocessor.
        
        Args:
            processed_data_dir: Directory to save processed data
        """
        self.processed_data_dir = Path(processed_data_dir)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.scalers = {}  # Store fitted scalers for inverse transforms
        
    def clean_basic_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Basic data cleaning: remove duplicates, handle missing values.
        
        Args:
            df: Raw stock data DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        logger.info(f"Starting basic cleaning for {len(df)} records")
        original_len = len(df)
        
        # Remove duplicates based on datetime and symbol
        if "datetime" in df.columns and "symbol" in df.columns:
            df = df.unique(subset=["datetime", "symbol"])
        elif "datetime" in df.columns:
            df = df.unique(subset=["datetime"])
            
        # Sort by datetime
        if "datetime" in df.columns:
            df = df.sort("datetime")
        
        # Check for missing values in critical columns
        price_columns = ["open", "high", "low", "close"]
        existing_price_cols = [col for col in price_columns if col in df.columns]
        
        if existing_price_cols:
            # Count missing values
            missing_counts = df.select([
                pl.col(col).is_null().sum().alias(f"{col}_missing") 
                for col in existing_price_cols
            ]).to_dict(as_series=False)
            
            for col, count in missing_counts.items():
                if count[0] > 0:
                    logger.warning(f"Found {count[0]} missing values in {col.replace('_missing', '')}")
            
            # Forward fill missing values 
            df = df.with_columns([
                pl.col(col).forward_fill().alias(col) 
                for col in existing_price_cols
            ])
            
            # If still missing values at the beginning, backward fill
            df = df.with_columns([
                pl.col(col).backward_fill().alias(col) 
                for col in existing_price_cols
            ])
            
            # Drop rows where all price columns are still null
            df = df.filter(
                ~pl.all_horizontal([pl.col(col).is_null() for col in existing_price_cols])
            )
        
        # Handle volume - set missing volumes to 0
        if "volume" in df.columns:
            df = df.with_columns(pl.col("volume").fill_null(0))
        
        logger.info(f"Basic cleaning complete: {len(df)} records (removed {original_len - len(df)})")
        return df
    
    def detect_and_handle_outliers(
        self, 
        df: pl.DataFrame, 
        method: str = "iqr",
        multiplier: float = 3.0,
        columns: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Detect and handle outliers in price data.
        
        Args:
            df: DataFrame to process
            method: Method to detect outliers ('iqr', 'zscore', 'winsorize')
            multiplier: Multiplier for outlier detection threshold
            columns: Specific columns to check, defaults to price columns
            
        Returns:
            DataFrame with outliers handled
        """
        if columns is None:
            columns = ["open", "high", "low", "close"]
            columns = [col for col in columns if col in df.columns]
        
        logger.info(f"Detecting outliers using {method} method")
        
        for col in columns:
            if method == "iqr":
                # Interquartile Range method
                q1 = df.select(pl.col(col).quantile(0.25)).item()
                q3 = df.select(pl.col(col).quantile(0.75)).item()
                iqr = q3 - q1
                lower_bound = q1 - multiplier * iqr
                upper_bound = q3 + multiplier * iqr
                
                # Count outliers
                outliers = df.filter(
                    (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
                ).height
                
                if outliers > 0:
                    logger.warning(f"Found {outliers} outliers in {col}")
                    # Cap outliers instead of removing (preserve time series continuity)
                    df = df.with_columns(
                        pl.col(col).clip(lower_bound, upper_bound).alias(col)
                    )
            
            elif method == "zscore":
                # Z-score method
                mean_val = df.select(pl.col(col).mean()).item()
                std_val = df.select(pl.col(col).std()).item()
                
                if std_val > 0:
                    outliers = df.filter(
                        (pl.col(col) - mean_val).abs() > multiplier * std_val
                    ).height
                    
                    if outliers > 0:
                        logger.warning(f"Found {outliers} outliers in {col}")
                        # Cap outliers
                        lower_bound = mean_val - multiplier * std_val
                        upper_bound = mean_val + multiplier * std_val
                        df = df.with_columns(
                            pl.col(col).clip(lower_bound, upper_bound).alias(col)
                        )
            
            elif method == "winsorize":
                # Winsorize at specified percentiles
                lower_percentile = (100 - 99.0) / 2  # 0.5th percentile
                upper_percentile = 100 - lower_percentile  # 99.5th percentile
                
                lower_bound = df.select(pl.col(col).quantile(lower_percentile/100)).item()
                upper_bound = df.select(pl.col(col).quantile(upper_percentile/100)).item()
                
                df = df.with_columns(
                    pl.col(col).clip(lower_bound, upper_bound).alias(col)
                )
        
        return df
    
    def add_returns_and_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add basic financial features like returns, log returns, etc.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Adding basic financial features")
        
        # Sort by datetime and symbol if multiple symbols
        if "symbol" in df.columns:
            df = df.sort(["symbol", "datetime"])
        else:
            df = df.sort("datetime")
        
        # Add returns (grouped by symbol if multiple stocks)
        if "symbol" in df.columns:
            df = df.with_columns([
                # Simple returns
                (pl.col("close").pct_change().over("symbol")).alias("returns"),
                # Log returns (better for modeling)
                (pl.col("close").log().diff().over("symbol")).alias("log_returns"),
                # High-Low spread
                ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("hl_spread"),
                # Open-Close spread  
                ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("oc_spread"),
            ])
        else:
            df = df.with_columns([
                pl.col("close").pct_change().alias("returns"),
                pl.col("close").log().diff().alias("log_returns"),
                ((pl.col("high") - pl.col("low")) / pl.col("close")).alias("hl_spread"),
                ((pl.col("close") - pl.col("open")) / pl.col("open")).alias("oc_spread"),
            ])
        
        # Add volume features if available
        if "volume" in df.columns:
            if "symbol" in df.columns:
                df = df.with_columns([
                    # Volume change
                    pl.col("volume").pct_change().over("symbol").alias("volume_change"),
                    # Price-Volume relationship
                    (pl.col("volume") * pl.col("close")).alias("dollar_volume"),
                ])
            else:
                df = df.with_columns([
                    pl.col("volume").pct_change().alias("volume_change"),
                    (pl.col("volume") * pl.col("close")).alias("dollar_volume"),
                ])
        
        # Add volatility features (rolling)
        window_sizes = [5, 10, 20]  # 5, 10, 20 day windows
        
        for window in window_sizes:
            if "symbol" in df.columns:
                df = df.with_columns([
                    pl.col("returns").rolling_std(window).over("symbol").alias(f"volatility_{window}d"),
                    pl.col("close").rolling_mean(window).over("symbol").alias(f"sma_{window}d"),
                ])
            else:
                df = df.with_columns([
                    pl.col("returns").rolling_std(window).alias(f"volatility_{window}d"),
                    pl.col("close").rolling_mean(window).alias(f"sma_{window}d"),
                ])
        
        return df
    
    def scale_features(
        self, 
        df: pl.DataFrame, 
        method: str = "standard",
        feature_columns: Optional[List[str]] = None,
        fit_scaler: bool = True
    ) -> pl.DataFrame:
        """
        Scale features for deep learning models.
        
        Args:
            df: DataFrame to scale
            method: Scaling method ('standard', 'minmax', 'robust')
            feature_columns: Columns to scale, if None scales all numeric columns
            fit_scaler: Whether to fit new scaler or use existing
            
        Returns:
            DataFrame with scaled features
        """
        logger.info(f"Scaling features using {method} scaling")
        
        if feature_columns is None:
            # Auto-detect numeric columns to scale
            numeric_columns = df.select(pl.col(pl.Float64, pl.Float32, pl.Int64, pl.Int32)).columns
            # Exclude datetime and symbol columns
            feature_columns = [
                col for col in numeric_columns 
                if col not in ["datetime", "symbol"]
            ]
        
        # Filter to existing columns
        feature_columns = [col for col in feature_columns if col in df.columns]
        
        if not feature_columns:
            logger.warning("No numeric columns found to scale")
            return df
        
        # Choose scaler
        if method == "standard":
            scaler_class = StandardScaler
        elif method == "minmax":
            scaler_class = MinMaxScaler
        elif method == "robust":
            scaler_class = RobustScaler
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        # Convert to pandas for sklearn (temporarily)
        df_pandas = df.to_pandas()
        
        # Handle multiple symbols separately
        if "symbol" in df.columns:
            scaled_dfs = []
            for symbol in df_pandas["symbol"].unique():
                symbol_df = df_pandas[df_pandas["symbol"] == symbol].copy()
                
                # Fit or use existing scaler
                scaler_key = f"{method}_{symbol}"
                if fit_scaler or scaler_key not in self.scalers:
                    self.scalers[scaler_key] = scaler_class()
                    # Fit on non-null values
                    mask = symbol_df[feature_columns].notna().all(axis=1)
                    if mask.sum() > 0:
                        self.scalers[scaler_key].fit(symbol_df.loc[mask, feature_columns])
                
                # Transform
                if scaler_key in self.scalers:
                    symbol_df[feature_columns] = self.scalers[scaler_key].transform(
                        symbol_df[feature_columns]
                    )
                
                scaled_dfs.append(symbol_df)
            
            # Properly convert and concatenate DataFrames
            try:
                polars_dfs = []
                for sdf in scaled_dfs:
                    if isinstance(sdf, pd.DataFrame):
                        polars_dfs.append(pl.from_pandas(sdf))
                    else:
                        polars_dfs.append(sdf)
                df_pandas = pl.concat(polars_dfs)
            except Exception as e:
                logger.error(f"DataFrame concatenation error: {e}")
                # Fallback: Convert entire dataframe
                df_pandas = pl.from_pandas(pd.concat(scaled_dfs, ignore_index=True))
        else:
            # Single symbol or combined data
            scaler_key = f"{method}_single"
            if fit_scaler or scaler_key not in self.scalers:
                self.scalers[scaler_key] = scaler_class()
                # Fit on non-null values
                mask = df_pandas[feature_columns].notna().all(axis=1)
                if mask.sum() > 0:
                    self.scalers[scaler_key].fit(df_pandas.loc[mask, feature_columns])
            
            # Transform
            if scaler_key in self.scalers:
                df_pandas[feature_columns] = self.scalers[scaler_key].transform(
                    df_pandas[feature_columns]
                )
            
            df_pandas = pl.from_pandas(df_pandas)
        
        logger.info(f"Scaled {len(feature_columns)} features")
        return df_pandas
    
    def create_sequences(
        self,
        df: pl.DataFrame,
        sequence_length: int = 60,
        target_column: str = "close",
        prediction_horizon: int = 1,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series deep learning.
        
        Args:
            df: Preprocessed DataFrame
            sequence_length: Length of input sequences
            target_column: Column to predict
            prediction_horizon: Steps ahead to predict
            feature_columns: Features to include in sequences
            
        Returns:
            Tuple of (X, y) arrays for training
        """
        logger.info(f"Creating sequences: length={sequence_length}, horizon={prediction_horizon}")
        
        if feature_columns is None:
            # Use all numeric columns except target
            numeric_columns = df.select(pl.col(pl.Float64, pl.Float32, pl.Int64, pl.Int32)).columns
            feature_columns = [
                col for col in numeric_columns 
                if col not in ["datetime", "symbol", target_column]
            ]
        
        # Add target column to features for sequence creation
        all_columns = feature_columns + [target_column]
        all_columns = [col for col in all_columns if col in df.columns]
        
        # Convert to numpy for sequence creation
        if "symbol" in df.columns:
            # Handle multiple symbols
            X_list, y_list = [], []
            
            for symbol in df["symbol"].unique():
                symbol_df = df.filter(pl.col("symbol") == symbol)
                symbol_data = symbol_df.select(all_columns).to_numpy()
                
                if len(symbol_data) < sequence_length + prediction_horizon:
                    logger.warning(f"Not enough data for {symbol}: {len(symbol_data)} records")
                    continue
                
                # Create sequences for this symbol
                X_symbol, y_symbol = self._create_sequences_single(
                    symbol_data, sequence_length, prediction_horizon, len(feature_columns)
                )
                X_list.append(X_symbol)
                y_list.append(y_symbol)
            
            if X_list:
                X = np.concatenate(X_list, axis=0)
                y = np.concatenate(y_list, axis=0)
            else:
                raise ValueError("No valid sequences created")
        else:
            # Single symbol
            data = df.select(all_columns).to_numpy()
            X, y = self._create_sequences_single(
                data, sequence_length, prediction_horizon, len(feature_columns)
            )
        
        logger.info(f"Created {len(X)} sequences with shape {X.shape}")
        return X, y
    
    def _create_sequences_single(
        self, 
        data: np.ndarray, 
        sequence_length: int, 
        prediction_horizon: int,
        n_features: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences from single time series."""
        X, y = [], []
        
        for i in range(sequence_length, len(data) - prediction_horizon + 1):
            # Features: all columns except target (last column)
            X.append(data[i-sequence_length:i, :n_features])
            # Target: last column (target_column) shifted by prediction_horizon
            y.append(data[i + prediction_horizon - 1, -1])
        
        return np.array(X), np.array(y)
    
    def save_processed_data(
        self, 
        df: pl.DataFrame, 
        filename: str,
        save_scalers: bool = True
    ) -> None:
        """
        Save processed data and scalers.
        
        Args:
            df: Processed DataFrame
            filename: Name for the saved file
            save_scalers: Whether to save fitted scalers
        """
        filepath = self.processed_data_dir / f"{filename}.parquet"
        df.write_parquet(filepath)
        logger.info(f"Saved processed data to {filepath}")
        
        if save_scalers and self.scalers:
            import joblib
            scaler_path = self.processed_data_dir / f"{filename}_scalers.joblib"
            joblib.dump(self.scalers, scaler_path)
            logger.info(f"Saved scalers to {scaler_path}")
    
    def load_processed_data(self, filename: str, load_scalers: bool = True) -> pl.DataFrame:
        """
        Load processed data and scalers.
        
        Args:
            filename: Name of the file to load
            load_scalers: Whether to load scalers
            
        Returns:
            Processed DataFrame
        """
        filepath = self.processed_data_dir / f"{filename}.parquet"
        df = pl.read_parquet(filepath)
        logger.info(f"Loaded processed data from {filepath}")
        
        if load_scalers:
            import joblib
            scaler_path = self.processed_data_dir / f"{filename}_scalers.joblib"
            if scaler_path.exists():
                self.scalers = joblib.load(scaler_path)
                logger.info(f"Loaded scalers from {scaler_path}")
        
        return df
    
    def _handle_nan_values(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Handle NaN values in the dataset after feature engineering.
        
        Strategy:
        - Drop rows where ALL technical indicators are NaN (beginning of time series)
        - For remaining NaN values, use forward fill then backward fill
        - For columns that still have NaN, drop those rows
        
        Args:
            df: DataFrame with potential NaN values
            
        Returns:
            DataFrame with NaN values handled
        """
        logger.info(f"Handling NaN values in dataset with {len(df)} records")
        
        # Get technical indicator columns (likely to have NaN)
        tech_columns = [col for col in df.columns if col not in ['symbol', 'datetime', 'date', 'open', 'high', 'low', 'close', 'volume']]
        
        initial_count = len(df)
        
        # Strategy 1: Remove rows where ALL technical indicators are NaN (start of series)
        if tech_columns:
            # Count non-null technical indicators per row
            non_null_count = df.select([
                pl.sum_horizontal([pl.col(col).is_not_null().cast(pl.Int32) for col in tech_columns]).alias("non_null_tech_count")
            ])["non_null_tech_count"]
            
            # Keep rows with at least some technical indicators
            mask = non_null_count > len(tech_columns) * 0.3  # Keep rows with >30% non-null tech indicators
            df = df.filter(mask)
            
            logger.info(f"Removed {initial_count - len(df)} rows with too many NaN technical indicators")
        
        # Strategy 2: Forward fill then backward fill remaining NaN values
        if "symbol" in df.columns:
            # Handle multi-symbol data
            df = df.with_columns([
                pl.col(col).forward_fill().backward_fill().over("symbol") if df[col].dtype.is_numeric() else pl.col(col)
                for col in df.columns
            ])
        else:
            # Handle single symbol data
            df = df.with_columns([
                pl.col(col).forward_fill().backward_fill() if df[col].dtype.is_numeric() else pl.col(col)
                for col in df.columns
            ])
        
        # Strategy 3: Drop any remaining rows with NaN values
        initial_len = len(df)
        df = df.drop_nulls()
        final_len = len(df)
        
        if initial_len != final_len:
            logger.info(f"Dropped {initial_len - final_len} rows with remaining NaN values")
        
        logger.info(f"NaN handling complete: {final_len} records remaining")
        return df
    
    def process_pipeline(
        self,
        df: pl.DataFrame,
        clean_data: bool = True,
        handle_outliers: bool = True,
        add_features: bool = True,
        scale_features: bool = True,
        scaling_method: str = "standard",
        save_result: bool = True,
        filename: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Run the complete preprocessing pipeline.
        
        Args:
            df: Raw DataFrame
            clean_data: Whether to clean basic data issues
            handle_outliers: Whether to handle outliers
            add_features: Whether to add derived features
            scale_features: Whether to scale features
            scaling_method: Method for scaling
            save_result: Whether to save the result
            filename: Filename for saving
            
        Returns:
            Fully processed DataFrame
        """
        logger.info("Starting complete preprocessing pipeline")
        
        processed_df = df.clone()
        
        if clean_data:
            processed_df = self.clean_basic_data(processed_df)
        
        if handle_outliers:
            processed_df = self.detect_and_handle_outliers(processed_df)
        
        if add_features:
            processed_df = self.add_returns_and_features(processed_df)
        
        # Handle NaN values after feature creation
        processed_df = self._handle_nan_values(processed_df)
        
        if scale_features:
            processed_df = self.scale_features(processed_df, method=scaling_method)
        
        if save_result:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_data_{timestamp}"
            self.save_processed_data(processed_df, filename)
        
        logger.info("Preprocessing pipeline complete")
        return processed_df


def main():    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Stock Data Preprocessor')
    parser.add_argument('--input', '-i', type=str, help='Input data file path')
    parser.add_argument('--output', '-o', type=str, help='Output filename (without extension)')
    parser.add_argument('--config', '-c', type=str, help='Configuration file path')
    parser.add_argument('--symbols', nargs='+', help='Stock symbols to process')
    parser.add_argument('--sequence-length', type=int, default=60, help='Sequence length for ML')
    parser.add_argument('--prediction-horizon', type=int, default=1, help='Prediction horizon')
    parser.add_argument('--scaling-method', choices=['standard', 'minmax', 'robust'], 
                       default='standard', help='Scaling method')
    parser.add_argument('--skip-outliers', action='store_true', help='Skip outlier handling')
    parser.add_argument('--skip-features', action='store_true', help='Skip feature engineering')
    parser.add_argument('--create-sequences', action='store_true', help='Create ML sequences')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('preprocessing.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger.info("Starting stock data preprocessing pipeline")
    
    try:
        # Initialize preprocessor
        preprocessor = StockDataPreprocessor()
        
        # Load data
        if args.input:
            input_path = Path(args.input)
            if not input_path.exists():
                logger.error(f"Input file not found: {args.input}")
                sys.exit(1)
            
            # Determine file format and load
            if input_path.suffix.lower() == '.parquet':
                df = pl.read_parquet(input_path)
            elif input_path.suffix.lower() == '.csv':
                df = pl.read_csv(input_path)
            else:
                logger.error(f"Unsupported file format: {input_path.suffix}")
                sys.exit(1)
            
            logger.info(f"Loaded data from {input_path}: {len(df)} records")
        else:
            # Try to load from downloader if no input specified
            try:
                from downloader import StockDataDownloader
                downloader = StockDataDownloader()
                
                if args.symbols:
                    # Download specific symbols
                    dfs = []
                    for symbol in args.symbols:
                        logger.info(f"Downloading data for {symbol}")
                        symbol_df = downloader.download_stock_data(symbol, period="2y")
                        dfs.append(symbol_df)
                    df = pl.concat(dfs) if len(dfs) > 1 else dfs[0]
                else:
                    # Default to Apple data
                    logger.info("No input specified, downloading AAPL data")
                    df = downloader.download_stock_data("AAPL", period="2y")
                
            except ImportError:
                logger.error("No input file specified and downloader not available")
                sys.exit(1)
        
        # Filter symbols if specified and data has symbol column
        if args.symbols and "symbol" in df.columns:
            df = df.filter(pl.col("symbol").is_in(args.symbols))
            logger.info(f"Filtered to symbols: {args.symbols}")
        
        logger.info(f"Processing {len(df)} records")
        
        # Run preprocessing pipeline
        processed_df = preprocessor.process_pipeline(
            df,
            clean_data=True,
            handle_outliers=not args.skip_outliers,
            add_features=not args.skip_features,
            scale_features=True,
            scaling_method=args.scaling_method,
            save_result=True,
            filename=args.output
        )
        
        # Create sequences if requested
        if args.create_sequences:
            logger.info("Creating ML sequences")
            X, y = preprocessor.create_sequences(
                processed_df,
                sequence_length=args.sequence_length,
                prediction_horizon=args.prediction_horizon
            )
            
            # Save sequences
            sequences_dir = Path("data/sequences")
            sequences_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            seq_filename = args.output or f"sequences_{timestamp}"
            
            np.save(sequences_dir / f"{seq_filename}_X.npy", X)
            np.save(sequences_dir / f"{seq_filename}_y.npy", y)
            logger.info(f"Saved sequences to data/sequences/{seq_filename}_X.npy and _y.npy")
        
        # Print summary statistics
        logger.info("Preprocessing completed successfully")
        logger.info(f"Final dataset shape: {processed_df.shape}")
        
        if "symbol" in processed_df.columns:
            symbol_counts = processed_df.group_by("symbol").len().sort("len", descending=True)
            logger.info("Records per symbol:")
            for row in symbol_counts.iter_rows():
                logger.info(f"  {row[0]}: {row[1]} records")
        
        # Show feature columns
        numeric_cols = processed_df.select(pl.col(pl.Float64, pl.Float32, pl.Int64, pl.Int32)).columns
        feature_cols = [col for col in numeric_cols if col not in ["datetime", "symbol"]]
        logger.info(f"Generated {len(feature_cols)} features: {feature_cols}")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Example use

Process from file
python preprocessor.py --input data/raw/stocks.csv --output processed_stocks

Download and process specific symbols
python preprocessor.py --symbols AAPL GOOGL MSFT --create-sequences

Custom preprocessing options
python preprocessor.py --input data.csv --scaling-method robust --skip-outliers --verbose
"""