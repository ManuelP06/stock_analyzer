import yfinance as yf
import polars as pl
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Union
import logging
import argparse
import sys
import json
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint

logger = logging.getLogger(__name__)

class StockDataDownloader:
    """Downloads and saves historical stock data using yfinance and polars."""

    def __init__(self, data_dir: str = "../../data/raw"):
        """
        Initialize the stock data downloader.

        Args:
            data_dir: Directory to save the downloaded data
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def download_stock_data(
        self,
        symbol: str,
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        save_parquet: bool = True,
        include_prepost: bool = False
    ) -> pl.DataFrame:
        """
        Download historical stock data for a given symbol.

        Args:
            symbol: Stock symbol (e.g., 'AAPL', 'MSFT')
            period: Period to download data for. Valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
            start: Start date string (YYYY-MM-DD) or datetime object
            end: End date string (YYYY-MM-DD) or datetime object  
            interval: Data interval. Valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
            save_parquet: Whether to save data as parquet file
            include_prepost: Include pre and post market data

        Returns:
            Polars DataFrame with stock data
        """
        logger.info(f"Downloading data for {symbol}")

        try:
            ticker = yf.Ticker(symbol)

            # Download data
            if period:
                data = ticker.history(
                    period=period,
                    interval=interval,
                    prepost=include_prepost,
                    auto_adjust=True,
                    back_adjust=False
                )
            else:
                data = ticker.history(
                    start=start,
                    end=end,
                    interval=interval,
                    prepost=include_prepost,
                    auto_adjust=True,
                    back_adjust=False
                )

            if data.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            # Convert to polars DataFrame
            data = data.reset_index()
            df = pl.DataFrame(data)

            # Add symbol column
            df = df.with_columns(pl.lit(symbol).alias("Symbol"))

            # Rename columns to standard format
            column_mapping = {
                "Date": "date",
                "Datetime": "datetime", 
                "Open": "open",
                "High": "high", 
                "Low": "low",
                "Close": "close",
                "Volume": "volume",
                "Symbol": "symbol"
            }

            # Only rename columns that exist
            existing_columns = df.columns
            rename_dict = {old: new for old, new in column_mapping.items() if old in existing_columns}
            df = df.rename(rename_dict)

            # Ensure datetime column exists and is properly typed
            if "datetime" in df.columns:
                df = df.with_columns(pl.col("datetime").dt.cast_time_unit("us"))
            elif "date" in df.columns:
                df = df.with_columns(
                    pl.col("date").dt.cast_time_unit("us").alias("datetime")
                )

            # Sort by datetime
            if "datetime" in df.columns:
                df = df.sort("datetime")

            logger.info(f"Downloaded {len(df)} records for {symbol}")

            # Save as parquet if requested
            if save_parquet:
                self._save_parquet(df, symbol, interval, start, end, period)

            return df

        except Exception as e:
            logger.error(f"Error downloading data for {symbol}: {str(e)}")
            raise

    def download_apple_data(
        self,
        period: str = "2y",
        interval: str = "1d", 
        save_parquet: bool = True
    ) -> pl.DataFrame:
        """
        Convenience method to download Apple (AAPL) stock data.

        Args:
            period: Period to download data for
            interval: Data interval
            save_parquet: Whether to save as parquet file

        Returns:
            Polars DataFrame with Apple stock data
        """
        return self.download_stock_data(
            symbol="AAPL",
            period=period,
            interval=interval,
            save_parquet=save_parquet
        )

    def download_multiple_stocks(
        self,
        symbols: List[str],
        period: Optional[str] = None,
        start: Optional[str] = None,
        end: Optional[str] = None,
        interval: str = "1d",
        save_parquet: bool = True
    ) -> pl.DataFrame:
        """
        Download data for multiple stock symbols and combine into single DataFrame.

        Args:
            symbols: List of stock symbols
            period: Period to download data for
            start: Start date
            end: End date
            interval: Data interval
            save_parquet: Whether to save as parquet files

        Returns:
            Combined Polars DataFrame with all stock data
        """
        all_data = []

        for symbol in symbols:
            try:
                df = self.download_stock_data(
                    symbol=symbol,
                    period=period,
                    start=start,
                    end=end,
                    interval=interval,
                    save_parquet=save_parquet
                )
                all_data.append(df)
                logger.info(f"Successfully downloaded data for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to download data for {symbol}: {str(e)}")
                continue

        if not all_data:
            raise ValueError("No data was successfully downloaded for any symbol")

        # Combine all dataframes
        combined_df = pl.concat(all_data, how="vertical_relaxed")

        if save_parquet:
            filename = f"multiplestocks{''.join(symbols)}{interval}.parquet"
            filepath = self.data_dir / filename
            combined_df.write_parquet(filepath)
            logger.info(f"Saved combined data to {filepath}")

        return combined_df

    def _save_parquet(
        self,
        df: pl.DataFrame,
        symbol: str,
        interval: str,
        start: Optional[str] = None,
        end: Optional[str] = None,
        period: Optional[str] = None
    ) -> None:
        """Save DataFrame as parquet file with descriptive filename."""

        # Create filename based on parameters
        if period:
            filename = f"{symbol}{period}{interval}.parquet"
        else:
            start_str = start.replace("-", "") if start else "unknown"
            end_str = end.replace("-", "") if end else datetime.now().strftime("%Y%m%d")
            filename = f"{symbol}{start_str}{end_str}{interval}.parquet"

        filepath = self.data_dir / filename
        df.write_parquet(filepath)
        logger.info(f"Saved {len(df)} records to {filepath}")

    def load_parquet(self, filepath: Union[str, Path]) -> pl.DataFrame:
        """
        Load stock data from parquet file.

        Args:
            filepath: Path to parquet file

        Returns:
            Polars DataFrame with stock data
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        df = pl.read_parquet(filepath)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        return df

    def get_stock_info(self, symbol: str) -> dict:
        """
        Get basic information about a stock.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with stock information
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Extract key information
            stock_info = {
                "symbol": symbol,
                "company_name": info.get("longName", "Unknown"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap", 0),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", "Unknown")
            }

            logger.info(f"Retrieved info for {symbol}: {stock_info['company_name']}")
            return stock_info

        except Exception as e:
            logger.error(f"Error getting info for {symbol}: {str(e)}")
            raise

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if a stock symbol is valid by attempting to fetch basic info.

        Args:
            symbol: Stock symbol to validate

        Returns:
            True if symbol is valid, False otherwise
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            return bool(info.get('symbol') or info.get('shortName'))
        except:
            return False

    def get_available_periods(self) -> List[str]:
        """Get list of valid period values."""
        return ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"]

    def get_available_intervals(self) -> List[str]:
        """Get list of valid interval values."""
        return ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1wk", "1mo", "3mo"]

    def list_saved_files(self) -> List[Path]:
        """List all saved parquet files in data directory."""
        return list(self.data_dir.glob("*.parquet"))

    def get_data_summary(self, df: pl.DataFrame) -> dict:
        """Get summary statistics for the downloaded data."""
        if df.is_empty():
            return {}

        summary = {
            "total_records": len(df),
            "date_range": {
                "start": df.select(pl.col("datetime").min()).item(),
                "end": df.select(pl.col("datetime").max()).item()
            },
            "symbols": df.select(pl.col("symbol").n_unique()).item() if "symbol" in df.columns else 1,
            "columns": df.columns,
            "price_stats": {}
        }

        if "close" in df.columns:
            price_stats = df.select([
                pl.col("close").min().alias("min_price"),
                pl.col("close").max().alias("max_price"),
                pl.col("close").mean().alias("avg_price"),
                pl.col("close").std().alias("price_std")
            ]).to_dict(as_series=False)

            summary["price_stats"] = {k: v[0] if v else None for k, v in price_stats.items()}

        return summary

def create_cli_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Professional Stock Data Downloader",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download single stock for 2 years
  python downloader.py --symbol AAPL --period 2y

  # Download multiple stocks with custom date range
  python downloader.py --symbols AAPL,MSFT,GOOGL --start 2022-01-01 --end 2023-12-31

  # Download with different interval
  python downloader.py --symbol TSLA --period 1y --interval 1h

  # List available periods and intervals
  python downloader.py --list-options

  # Get stock information
  python downloader.py --info AAPL
        """
    )

    parser.add_argument(
        "--symbol", "-s",
        type=str,
        help="Single stock symbol to download (e.g., AAPL)"
    )

    parser.add_argument(
        "--symbols", "-ms",
        type=str,
        help="Multiple stock symbols separated by comma (e.g., AAPL,MSFT,GOOGL)"
    )

    parser.add_argument(
        "--period", "-p",
        type=str,
        default="2y",
        help="Period to download (1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max). Default: 2y"
    )

    parser.add_argument(
        "--start",
        type=str,
        help="Start date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--end",
        type=str,
        help="End date (YYYY-MM-DD)"
    )

    parser.add_argument(
        "--interval", "-i",
        type=str,
        default="1d",
        help="Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo). Default: 1d"
    )

    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="data/raw",
        help="Output directory for saved files. Default: data/raw"
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save data to parquet file"
    )

    parser.add_argument(
        "--info",
        type=str,
        help="Get detailed information for a stock symbol"
    )

    parser.add_argument(
        "--list-options",
        action="store_true",
        help="List available periods and intervals"
    )

    parser.add_argument(
        "--list-files",
        action="store_true",
        help="List saved parquet files"
    )

    parser.add_argument(
        "--validate",
        type=str,
        help="Validate if a stock symbol exists"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet mode - minimal output"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )

    return parser


def display_stock_info(info: dict, console: Console, as_json: bool = False):
    """Display stock information in a formatted table or JSON."""
    if as_json:
        rprint(json.dumps(info, indent=2))
        return

    table = Table(title=f"Stock Information: {info.get('symbol', 'N/A')}")
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")

    for key, value in info.items():
        if key != "symbol":
            display_key = key.replace("_", " ").title()
            if isinstance(value, (int, float)) and key == "market_cap" and value > 0:
                value = f"${value:,.0f}"
            table.add_row(display_key, str(value))

    console.print(table)


def display_data_summary(summary: dict, console: Console, as_json: bool = False):
    """Display data summary in formatted table or JSON."""
    if as_json:
        rprint(json.dumps(summary, indent=2, default=str))
        return

    table = Table(title="Download Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Records", f"{summary.get('total_records', 0):,}")
    table.add_row("Symbols", str(summary.get('symbols', 1)))
    table.add_row("Columns", str(len(summary.get('columns', []))))

    if summary.get('date_range'):
        date_range = summary['date_range']
        table.add_row("Start Date", str(date_range.get('start', 'N/A')))
        table.add_row("End Date", str(date_range.get('end', 'N/A')))

    if summary.get('price_stats'):
        price_stats = summary['price_stats']
        if price_stats.get('min_price'):
            table.add_row("Min Price", f"${price_stats['min_price']:.2f}")
        if price_stats.get('max_price'):
            table.add_row("Max Price", f"${price_stats['max_price']:.2f}")
        if price_stats.get('avg_price'):
            table.add_row("Avg Price", f"${price_stats['avg_price']:.2f}")

    console.print(table)


def main():
    """Main CLI function."""
    parser = create_cli_parser()
    args = parser.parse_args()

    # Set up logging
    log_level = logging.WARNING if args.quiet else logging.INFO
    logging.basicConfig(level=log_level, format='%(levelname)s: %(message)s')

    console = Console()

    try:
        # Initialize downloader
        downloader = StockDataDownloader(data_dir=args.output_dir)

        # List options
        if args.list_options:
            if args.json:
                options = {
                    "periods": downloader.get_available_periods(),
                    "intervals": downloader.get_available_intervals()
                }
                rprint(json.dumps(options, indent=2))
            else:
                console.print("Available Periods:", style="bold")
                console.print(", ".join(downloader.get_available_periods()))
                console.print("\nAvailable Intervals:", style="bold")
                console.print(", ".join(downloader.get_available_intervals()))
            return

        # List files
        if args.list_files:
            files = downloader.list_saved_files()
            if args.json:
                rprint(json.dumps([str(f) for f in files], indent=2))
            else:
                console.print(f"Found {len(files)} saved files in {args.output_dir}:")
                for file in files:
                    console.print(f"  {file.name}")
            return

        # Validate symbol
        if args.validate:
            symbol = args.validate.upper()
            is_valid = downloader.validate_symbol(symbol)
            if args.json:
                rprint(json.dumps({"symbol": symbol, "valid": is_valid}))
            else:
                status = "[green]✓ Valid[/green]" if is_valid else "[red]✗ Invalid[/red]"
                console.print(f"Symbol {symbol}: {status}")
            return

        # Get stock info
        if args.info:
            symbol = args.info.upper()
            with console.status(f"[bold green]Fetching info for {symbol}..."):
                info = downloader.get_stock_info(symbol)
            display_stock_info(info, console, args.json)
            return

        # Download data
        symbols = []
        if args.symbol:
            symbols = [args.symbol.upper()]
        elif args.symbols:
            symbols = [s.strip().upper() for s in args.symbols.split(",")]
        else:
            console.print("[red]Error: Must specify --symbol or --symbols[/red]")
            return

        # Validate symbols
        if not args.quiet:
            with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}")) as progress:
                task = progress.add_task("Validating symbols...", total=None)
                valid_symbols = []
                for symbol in symbols:
                    if downloader.validate_symbol(symbol):
                        valid_symbols.append(symbol)
                    else:
                        console.print(f"[yellow]Warning: {symbol} may not be valid[/yellow]")

        # Download data
        save_parquet = not args.no_save

        if len(symbols) == 1:
            symbol = symbols[0]
            with console.status(f"[bold green]Downloading {symbol} data..."):
                df = downloader.download_stock_data(
                    symbol=symbol,
                    period=args.period if not (args.start and args.end) else None,
                    start=args.start,
                    end=args.end,
                    interval=args.interval,
                    save_parquet=save_parquet
                )
        else:
            with console.status(f"[bold green]Downloading {len(symbols)} stocks..."):
                df = downloader.download_multiple_stocks(
                    symbols=symbols,
                    period=args.period if not (args.start and args.end) else None,
                    start=args.start,
                    end=args.end,
                    interval=args.interval,
                    save_parquet=save_parquet
                )

        # Display results
        if not args.quiet:
            summary = downloader.get_data_summary(df)
            display_data_summary(summary, console, args.json)
            console.print(f"[green]✓ Successfully downloaded data for {len(symbols)} symbol(s)[/green]")

    except KeyboardInterrupt:
        console.print("[yellow]Download interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {str(e)}[/red]")
        sys.exit(1)

if __name__ == "__main__":
    main()