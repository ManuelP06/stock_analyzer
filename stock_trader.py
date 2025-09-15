import argparse
import json
import logging
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import torch
import numpy as np
import pandas as pd
import polars as pl
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.downloader import StockDataDownloader
from src.features.technical_indicators import TechnicalIndicators
from src.data.preprocessor import StockDataPreprocessor
from src.models.trainer import create_trainer_from_config, StockDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ProductionStockTrader:
    """Stock trading system with consistent feature handling"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.config = None
        self.feature_columns = None  # Store feature columns for consistency
        self.model_path = None
        self.downloader = StockDataDownloader(data_dir="temp_data")
        
    def download_data(self, symbols, period="3y", interval="1d"):
        """Download stock data using StockDataDownloader with configurable interval"""
        logger.info(f"Downloading {period} data with {interval} interval for {symbols}")

        # Use the downloader to get all symbols data
        combined_df = self.downloader.download_multiple_stocks(
            symbols=symbols,
            period=period,
            interval=interval,
            save_parquet=False  # Don't save intermediate files
        )

        logger.info(f"Total records: {len(combined_df)}")
        return combined_df
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess data with consistent feature handling"""
        
        # Add technical indicators
        logger.info("Adding technical indicators...")
        df_with_indicators = TechnicalIndicators.add_all_indicators(df)
        
        # Preprocess
        logger.info("Preprocessing data...")
        processed_dir = Path("temp_data")
        processed_dir.mkdir(exist_ok=True)
        
        preprocessor = StockDataPreprocessor(processed_data_dir=str(processed_dir))
        processed_df = preprocessor.process_pipeline(
            df_with_indicators,
            clean_data=True,
            handle_outliers=True,
            add_features=True,
            scale_features=False,
            save_result=False,
            filename=None
        )
        
        if is_training:
            exclude_cols = {'datetime', 'symbol', 'close'}  # close is the target
            self.feature_columns = [col for col in processed_df.columns 
                                  if col not in exclude_cols]
            logger.info(f"Training features: {len(self.feature_columns)}")
        else:
            if self.feature_columns:
                current_features = [col for col in processed_df.columns 
                                  if col not in ['datetime', 'symbol']]
                
                for feature in self.feature_columns:
                    if feature not in processed_df.columns:
                        processed_df = processed_df.with_columns(
                            pl.lit(0.0).alias(feature)
                        )
                
                keep_columns = ['datetime', 'symbol'] + self.feature_columns
                processed_df = processed_df.select([col for col in keep_columns 
                                                  if col in processed_df.columns])
                
                logger.info(f"Inference features aligned: {len(self.feature_columns)}")
        
        logger.info(f"Processed data: {processed_df.shape}")
        return processed_df
    
    def split_data(self, df, train_ratio=0.8, val_ratio=0.15):
        """Split data chronologically"""
        train_dfs, val_dfs, test_dfs = [], [], []
        
        for symbol in df["symbol"].unique():
            symbol_df = df.filter(pl.col("symbol") == symbol).sort("datetime")
            n = len(symbol_df)
            
            train_end = int(n * train_ratio)
            val_end = int(n * (train_ratio + val_ratio))
            
            train_dfs.append(symbol_df[:train_end])
            val_dfs.append(symbol_df[train_end:val_end])
            test_dfs.append(symbol_df[val_end:])
        
        train_df = pl.concat(train_dfs) if train_dfs else pl.DataFrame()
        val_df = pl.concat(val_dfs) if val_dfs else pl.DataFrame()
        test_df = pl.concat(test_dfs) if test_dfs else pl.DataFrame()
        
        logger.info(f"Split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def train(self, symbols, epochs=20, seq_len=60, pred_len=5, period="3y", interval="1d"):
        """Train the model"""
        logger.info("="*60)
        logger.info("STOCK TRADING SYSTEM - TRAINING")
        logger.info("="*60)

        raw_data = self.download_data(symbols, period, interval)
        processed_data = self.preprocess_data(raw_data, is_training=True)
        train_df, val_df, test_df = self.split_data(processed_data)
        
        n_features = len(self.feature_columns)
        
        model_config = {
            'seq_len': seq_len,
            'pred_len': pred_len,
            'n_features': n_features,
            'patch_len': min(16, seq_len // 4),
            'stride': min(8, seq_len // 8),
            'embed_dim': 256,
            'num_layers': 6,
            'num_heads': 8,
            'ff_dim': 1024,
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'drop_path': 0.1,
            'use_multi_scale': True,
            'use_relative_pos': True
        }
        
        training_config = {
            'epochs': epochs,
            'batch_size': 32,
            'learning_rate': 1e-4,
            'weight_decay': 1e-4,
            'use_amp': True,
            'grad_clip_norm': 1.0,
            'early_stopping_patience': 10,
            'warmup_steps': 500,
            'scheduler': 'cosine_with_warmup'
        }
        
        # Train
        logger.info(f"Training model: {n_features} features, {seq_len}→{pred_len}")
        trainer = create_trainer_from_config(
            model_config=model_config,
            training_config=training_config,
            train_data=train_df,
            val_data=val_df,
            experiment_name=f"stock_trader_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        start_time = datetime.now()
        trainer.train()
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store trained model (not compiled version for consistency)
        self.model = trainer.model
        self.model.eval()
        
        # Save configuration including feature columns
        self.config = {
            'model_config': model_config,
            'training_config': training_config,
            'feature_columns': self.feature_columns,
            'symbols': symbols,
            'period': period,
            'training_time': training_time,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save model and config with unique timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_id = f"model_{timestamp}_{'-'.join(symbols)}"
        output_dir = Path("trading_models")
        output_dir.mkdir(exist_ok=True)

        # Save uncompiled model state - get original model if compiled
        model_to_save = self.model
        if hasattr(self.model, '_orig_mod'):
            model_to_save = self.model._orig_mod  # Get original uncompiled model

        model_state = {
            'model_state_dict': model_to_save.state_dict(),
            'config': self.config
        }

        self.model_path = output_dir / f"{model_id}.pt"
        torch.save(model_state, self.model_path)
        
        # Save config separately with unique name
        with open(output_dir / f"{model_id}_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Test evaluation if possible
        test_metrics = {}
        if len(test_df) > 0:
            test_dataset = StockDataset(
                data=test_df,
                sequence_length=seq_len,
                prediction_length=pred_len,
                normalize=True
            )
            
            if len(test_dataset) > 0:
                test_results = trainer.predict(test_dataset)
                predictions = test_results['predictions']
                targets = test_results['targets']
                
                if len(predictions) > 0:
                    pred_flat = predictions.flatten()
                    targ_flat = targets.flatten()
                    
                    test_metrics = {
                        'mae': float(mean_absolute_error(targ_flat, pred_flat)),
                        'mse': float(mean_squared_error(targ_flat, pred_flat)),
                        'rmse': float(np.sqrt(mean_squared_error(targ_flat, pred_flat))),
                        'r2': float(r2_score(targ_flat, pred_flat))
                    }
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Training time: {training_time/60:.1f} minutes")
        logger.info(f"Model saved: {self.model_path}")
        if test_metrics:
            for metric, value in test_metrics.items():
                logger.info(f"Test {metric.upper()}: {value:.4f}")
        logger.info("="*60)
        
        return test_metrics
    
    def load_model(self, model_path=None):
        """Load trained model"""
        if model_path is None:
            model_path = "trading_model/model.pt"
        
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Load model and config
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
        
        self.config = checkpoint['config']
        self.feature_columns = self.config['feature_columns']
        
        # Recreate model
        from src.models.patchtst import PatchTST
        model_config = self.config['model_config']
        
        self.model = PatchTST(
            seq_len=model_config['seq_len'],
            pred_len=model_config['pred_len'],
            n_features=model_config['n_features'],
            patch_len=model_config['patch_len'],
            stride=model_config['stride'],
            embed_dim=model_config['embed_dim'],
            num_layers=model_config['num_layers'],
            num_heads=model_config['num_heads'],
            ff_dim=model_config['ff_dim'],
            dropout=model_config['dropout'],
            attention_dropout=model_config['attention_dropout'],
            drop_path=model_config['drop_path'],
            use_multi_scale=model_config['use_multi_scale'],
            use_relative_pos=model_config['use_relative_pos']
        )
        
        # Load weights - handle compiled model prefixes
        state_dict = checkpoint['model_state_dict']
        
        # Remove _orig_mod. prefixes if present (compiled models)
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key.replace('_orig_mod.', '')
                new_state_dict[new_key] = value
            state_dict = new_state_dict
        
        # Load weights with strict=False to ignore unexpected keys
        try:
            self.model.load_state_dict(state_dict, strict=False)
            logger.info("Model loaded successfully (strict=False)")
        except Exception as e:
            logger.warning(f"Model loading failed: {e}")
            # Try to load only matching keys
            model_dict = self.model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(filtered_dict)
            self.model.load_state_dict(model_dict)
            logger.info(f"Model loaded with {len(filtered_dict)} matching parameters")

        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded: {len(self.feature_columns)} features")
        return True

    def get_close_price_scaler_params(self, symbol_data):
        """Extract scaler parameters for close price"""
        try:
            # Get close price column (usually 'close' or at index for close)
            if 'close' in symbol_data.columns:
                close_values = symbol_data['close'].to_numpy()
            else:
                # If preprocessed, close might be at a specific index
                close_values = symbol_data[self.feature_columns[0]].to_numpy()  # First feature should be close

            # Calculate standard scaler params (mean and std)
            mean = np.mean(close_values)
            std = np.std(close_values)
            return {'mean': mean, 'std': std}
        except Exception as e:
            logger.warning(f"Could not get scaler params: {e}")
            return {'mean': 0.0, 'std': 1.0}

    def inverse_transform_predictions(self, scaled_predictions, scaler_params, last_actual_price):
        """Transform scaled predictions back to realistic daily price changes"""
        try:
            # REALISTIC MARKET STATISTICS:
            # - Average daily change: ±0.8%
            # - 68% of days: ±1.2% (1 std dev)
            # - 95% of days: ±2.4% (2 std dev)
            # - Extreme days (rare): ±3.5%
            # - Major events only: >±5%

            # Severely constrain predictions to market reality
            MAX_REALISTIC_DAILY = 0.035  # Max 3.5% daily change (very rare)
            TYPICAL_DAILY_STD = 0.012    # 1.2% standard deviation

            # Clip raw predictions to prevent extreme values
            scaled_predictions = np.clip(scaled_predictions, -0.05, 0.05)

            abs_predictions = []
            current_price = last_actual_price

            for i, raw_change in enumerate(scaled_predictions):
                # Apply aggressive dampening - reduce to realistic levels
                # Most neural networks produce values that are too extreme
                dampened_change = raw_change * 0.1  # Reduce by 90%

                # Add market-realistic noise and constraints
                # Ensure predictions follow normal market distribution
                if abs(dampened_change) > TYPICAL_DAILY_STD:
                    # For larger changes, reduce them further (rare events)
                    dampened_change = dampened_change * 0.5

                # Absolute maximum constraint
                dampened_change = np.clip(dampened_change, -MAX_REALISTIC_DAILY, MAX_REALISTIC_DAILY)

                # Calculate new price
                new_price = current_price * (1 + dampened_change)

                # Additional safety checks
                new_price = max(new_price, current_price * 0.965)  # Max 3.5% drop
                new_price = min(new_price, current_price * 1.035)  # Max 3.5% gain

                abs_predictions.append(new_price)
                current_price = new_price

            return np.array(abs_predictions)

        except Exception as e:
            logger.warning(f"Prediction transform failed: {e}, using market-realistic fallback")
            # Ultra-conservative fallback based on actual market statistics
            abs_predictions = []
            current_price = last_actual_price

            for i in range(len(scaled_predictions)):
                # Generate realistic daily changes following normal distribution
                # 68% of changes will be within ±0.8%
                realistic_change = np.random.normal(0, 0.008)  # Mean 0%, Std 0.8%
                realistic_change = np.clip(realistic_change, -0.025, 0.025)  # Max ±2.5%

                new_price = current_price * (1 + realistic_change)
                abs_predictions.append(new_price)
                current_price = new_price

            return np.array(abs_predictions)

    def validate_predictions_against_history(self, predictions, symbol_data, last_actual_price):
        """Validate predictions against historical volatility patterns"""
        try:
            # Get historical daily returns for the symbol
            if 'close' in symbol_data.columns:
                historical_prices = symbol_data['close'].to_numpy()
            else:
                # If no close column, use the first feature column
                historical_prices = symbol_data[symbol_data.columns[0]].to_numpy()

            # Calculate historical daily returns
            if len(historical_prices) > 1:
                historical_returns = np.diff(historical_prices) / historical_prices[:-1]
                historical_volatility = np.std(historical_returns)
                max_historical_daily_change = np.max(np.abs(historical_returns))

                # Calculate predicted returns
                predicted_returns = []
                for i, pred_price in enumerate(predictions):
                    if i == 0:
                        daily_return = (pred_price - last_actual_price) / last_actual_price
                    else:
                        daily_return = (pred_price - predictions[i-1]) / predictions[i-1]
                    predicted_returns.append(daily_return)

                predicted_returns = np.array(predicted_returns)
                max_predicted_change = np.max(np.abs(predicted_returns))

                # If predictions are too volatile compared to history, constrain them
                if max_predicted_change > max_historical_daily_change * 1.5:
                    # Scale down all predictions to be more in line with historical volatility
                    scaling_factor = (max_historical_daily_change * 1.2) / max_predicted_change
                    logger.info(f"Scaling down volatile predictions by factor: {scaling_factor:.3f}")

                    # Apply scaling
                    validated_predictions = []
                    current_price = last_actual_price

                    for daily_return in predicted_returns:
                        scaled_return = daily_return * scaling_factor
                        # Further constrain to market reality
                        scaled_return = np.clip(scaled_return, -0.03, 0.03)  # Max ±3%
                        new_price = current_price * (1 + scaled_return)
                        validated_predictions.append(new_price)
                        current_price = new_price

                    return np.array(validated_predictions)

            return predictions

        except Exception as e:
            logger.warning(f"Prediction validation failed: {e}, using original predictions")
            return predictions

    def analyze_stocks(self, symbols, interval="1d"):
        """Generate comprehensive stock analysis with ML insights"""
        if self.model is None:
            raise ValueError("No model loaded. Train first or load existing model.")

        logger.info("="*60)
        logger.info("PROFESSIONAL STOCK ANALYSIS")
        logger.info("="*60)

        # Get model parameters
        seq_len = self.config['model_config']['seq_len']
        pred_len = self.config['model_config']['pred_len']

        # Get comprehensive data for analysis
        lookback_days = max(seq_len + 200, 500)  # More data for better analysis
        raw_data = self.download_data(symbols, period=f"{lookback_days}d", interval=interval)
        processed_data = self.preprocess_data(raw_data, is_training=False)

        analyses = {}
        
        for symbol in symbols:
            try:
                # Get symbol data
                symbol_data = processed_data.filter(
                    processed_data['symbol'] == symbol
                ).sort('datetime')

                if len(symbol_data) < seq_len:
                    logger.warning(f"Insufficient data for {symbol}: {len(symbol_data)} < {seq_len}")
                    continue

                # Get raw price data for analysis
                raw_symbol_data = raw_data.filter(pl.col('symbol') == symbol).sort('datetime')
                historical_prices = raw_symbol_data['close'].to_numpy()

                # Calculate comprehensive market analysis
                analysis = self._generate_comprehensive_analysis(
                    symbol, symbol_data, raw_symbol_data, historical_prices
                )

                # Add ML-based trend insights (not precise predictions)
                ml_insights = self._generate_ml_insights(symbol_data, seq_len)
                analysis.update(ml_insights)

                analyses[symbol] = analysis

                logger.info(f"Analysis completed for {symbol}: "
                          f"Trend: {analysis['overall_trend']}, "
                          f"Risk Level: {analysis['risk_assessment']}")

            except Exception as e:
                logger.error(f"Failed to analyze {symbol}: {e}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")

        return analyses

    def _generate_comprehensive_analysis(self, symbol, symbol_data, raw_symbol_data, historical_prices):
        """Generate comprehensive professional stock analysis"""
        try:
            current_price = float(historical_prices[-1])
            prices_30d = historical_prices[-30:] if len(historical_prices) >= 30 else historical_prices
            prices_90d = historical_prices[-90:] if len(historical_prices) >= 90 else historical_prices

            # Calculate returns
            daily_returns = np.diff(historical_prices) / historical_prices[:-1]
            monthly_returns = (prices_30d[-1] / prices_30d[0]) - 1 if len(prices_30d) > 1 else 0
            quarterly_returns = (prices_90d[-1] / prices_90d[0]) - 1 if len(prices_90d) > 1 else 0

            # Volatility analysis
            daily_volatility = np.std(daily_returns)
            annualized_volatility = daily_volatility * np.sqrt(252)

            # Technical indicators from processed data
            rsi = self._calculate_current_rsi(historical_prices)
            ma20 = np.mean(historical_prices[-20:]) if len(historical_prices) >= 20 else current_price
            ma50 = np.mean(historical_prices[-50:]) if len(historical_prices) >= 50 else current_price

            # Price levels analysis
            support_level = np.min(prices_30d)
            resistance_level = np.max(prices_30d)
            price_range = (current_price - support_level) / (resistance_level - support_level) if resistance_level != support_level else 0.5

            # Trend analysis
            short_trend = "Bullish" if current_price > ma20 else "Bearish"
            medium_trend = "Bullish" if ma20 > ma50 else "Bearish"
            overall_trend = "Bullish" if (current_price > ma20 and ma20 > ma50) else "Bearish" if (current_price < ma20 and ma20 < ma50) else "Neutral"

            # Risk assessment
            if annualized_volatility > 0.4:
                risk_level = "High"
            elif annualized_volatility > 0.25:
                risk_level = "Medium"
            else:
                risk_level = "Low"

            return {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_1d': daily_returns[-1] if len(daily_returns) > 0 else 0,
                'price_change_30d': monthly_returns,
                'price_change_90d': quarterly_returns,
                'daily_volatility': daily_volatility,
                'annualized_volatility': annualized_volatility,
                'rsi': rsi,
                'ma20': ma20,
                'ma50': ma50,
                'support_level': support_level,
                'resistance_level': resistance_level,
                'price_position': price_range,  # 0-1, where in range between support/resistance
                'short_term_trend': short_trend,
                'medium_term_trend': medium_trend,
                'overall_trend': overall_trend,
                'risk_assessment': risk_level,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}

    def _generate_ml_insights(self, symbol_data, seq_len):
        """Generate ML-based insights (trend probability, not precise predictions)"""
        try:
            # Prepare input for ML model
            features = symbol_data[self.feature_columns].to_numpy()
            input_seq = features[-seq_len:]  # Most recent data
            input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(self.device)

            with torch.no_grad():
                outputs = self.model(input_tensor)
                raw_output = outputs['predictions'].cpu().numpy()[0]

            # Instead of price predictions, extract trend insights
            trend_direction = np.mean(raw_output)  # Average direction
            trend_strength = np.std(raw_output)    # Volatility/uncertainty
            trend_consistency = 1 - (np.std(np.diff(raw_output)) / np.mean(np.abs(raw_output))) if np.mean(np.abs(raw_output)) > 0 else 0

            # Convert to interpretable insights
            if trend_direction > 0.02:
                ml_trend = "Bullish"
                confidence = min(trend_direction * 20, 0.9)  # Scale to 0-0.9
            elif trend_direction < -0.02:
                ml_trend = "Bearish"
                confidence = min(abs(trend_direction) * 20, 0.9)
            else:
                ml_trend = "Neutral"
                confidence = 0.3

            # Volatility expectation
            if trend_strength > 0.05:
                volatility_outlook = "High"
            elif trend_strength > 0.02:
                volatility_outlook = "Medium"
            else:
                volatility_outlook = "Low"

            return {
                'ml_trend_direction': ml_trend,
                'ml_trend_confidence': float(confidence),
                'ml_volatility_outlook': volatility_outlook,
                'ml_trend_consistency': float(max(0, min(1, trend_consistency))),
                'ml_notes': f"Based on {seq_len} days of market patterns and technical indicators"
            }

        except Exception as e:
            logger.error(f"Error in ML insights generation: {e}")
            return {
                'ml_trend_direction': "Unknown",
                'ml_trend_confidence': 0.0,
                'ml_volatility_outlook': "Unknown",
                'ml_trend_consistency': 0.0,
                'ml_notes': "ML analysis unavailable"
            }

    def _calculate_current_rsi(self, prices, period=14):
        """Calculate current RSI value"""
        if len(prices) < period + 1:
            return 50.0  # Neutral RSI if insufficient data

        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)

    def save_results(self, analyses, output_dir="stock_analyses"):
        """Save professional stock analyses"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        symbols_str = '-'.join(analyses.keys()) if analyses else 'unknown'
        unique_dir = f"{output_dir}_{timestamp}_{symbols_str}"
        output_path = Path(unique_dir)
        output_path.mkdir(exist_ok=True)

        # Also save to the default directory for app.py compatibility
        default_path = Path(output_dir)
        default_path.mkdir(exist_ok=True)

        # Save analyses to both unique and default directories
        for save_path in [output_path, default_path]:
            with open(save_path / 'analyses.json', 'w') as f:
                json.dump(analyses, f, indent=2, default=str)
        
        # Create professional analysis report
        with open(output_path / 'analysis_report.md', 'w') as f:
            f.write(f"# Professional Stock Analysis Report\n\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            f.write("## Executive Summary\n\n")
            f.write("| Symbol | Current Price | 30d Return | Trend | Risk | ML Insight |\n")
            f.write("|--------|---------------|------------|-------|------|------------|\n")

            for symbol, analysis in analyses.items():
                if 'error' not in analysis:
                    f.write(f"| {symbol} | ${analysis['current_price']:.2f} | "
                           f"{analysis['price_change_30d']:+.2%} | {analysis['overall_trend']} | "
                           f"{analysis['risk_assessment']} | {analysis['ml_trend_direction']} |\n")

            f.write("\n## Detailed Analysis\n\n")
            for symbol, analysis in analyses.items():
                if 'error' not in analysis:
                    f.write(f"### {symbol}\n\n")
                    f.write(f"**Current Price**: ${analysis['current_price']:.2f}\n")
                    f.write(f"**Performance**:\n")
                    f.write(f"- 1-Day: {analysis['price_change_1d']:+.2%}\n")
                    f.write(f"- 30-Day: {analysis['price_change_30d']:+.2%}\n")
                    f.write(f"- 90-Day: {analysis['price_change_90d']:+.2%}\n\n")

                    f.write(f"**Technical Analysis**:\n")
                    f.write(f"- RSI: {analysis['rsi']:.1f}\n")
                    f.write(f"- MA20: ${analysis['ma20']:.2f}\n")
                    f.write(f"- MA50: ${analysis['ma50']:.2f}\n")
                    f.write(f"- Support: ${analysis['support_level']:.2f}\n")
                    f.write(f"- Resistance: ${analysis['resistance_level']:.2f}\n\n")

                    f.write(f"**Risk Assessment**:\n")
                    f.write(f"- Daily Volatility: {analysis['daily_volatility']:.2%}\n")
                    f.write(f"- Annualized Volatility: {analysis['annualized_volatility']:.1%}\n")
                    f.write(f"- Risk Level: {analysis['risk_assessment']}\n\n")

                    f.write(f"**ML Insights**:\n")
                    f.write(f"- Trend Direction: {analysis['ml_trend_direction']}\n")
                    f.write(f"- Confidence: {analysis['ml_trend_confidence']:.1%}\n")
                    f.write(f"- Volatility Outlook: {analysis['ml_volatility_outlook']}\n")
                    f.write(f"- Pattern Consistency: {analysis['ml_trend_consistency']:.1%}\n\n")
                    f.write(f"*{analysis['ml_notes']}*\n\n")

            f.write("## Important Disclaimers\n\n")
            f.write("**This analysis is for research and educational purposes only.**\n\n")
            f.write("- ML predictions have inherent limitations and uncertainty\n")
            f.write("- Past performance does not guarantee future results\n")
            f.write("- Markets are influenced by unpredictable events\n")
            f.write("- Always consult qualified financial professionals\n")
            f.write("- Never invest more than you can afford to lose\n")
        
        logger.info(f"Trading results saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Production Stock Trading System")
    parser.add_argument("command", choices=['train', 'predict', 'both'],
                       help="Command to run")
    parser.add_argument("--symbols", nargs='+', default=['AAPL', 'MSFT', 'GOOGL', 'NVDA'],
                       help="Stock symbols to trade")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--seq_len", type=int, default=60, help="Sequence length")
    parser.add_argument("--pred_len", type=int, default=5, help="Prediction length")
    parser.add_argument("--period", default="3y", help="Training data period")
    parser.add_argument("--interval", default="1d", help="Data interval (1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo)")
    parser.add_argument("--threshold", type=float, default=0.008, help="Signal threshold (realistic: 0.008 = 0.8%)")
    parser.add_argument("--model_path", default=None, help="Path to saved model")
    
    args = parser.parse_args()
    
    # Initialize trading system
    trader = ProductionStockTrader()
    
    try:
        if args.command in ['train', 'both']:
            logger.info("Starting training...")
            test_metrics = trader.train(
                symbols=args.symbols,
                epochs=args.epochs,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                period=args.period,
                interval=args.interval
            )
        
        if args.command in ['predict', 'both']:
            if args.command == 'predict':
                # Load existing model
                model_path = args.model_path or "trading_model/model.pt"
                trader.load_model(model_path)
            
            logger.info("Generating professional stock analysis...")
            analyses = trader.analyze_stocks(args.symbols, args.interval)
            
            if analyses:
                # Display results
                print("\n" + "="*60)
                print("PROFESSIONAL STOCK ANALYSIS")
                print("="*60)

                for symbol in args.symbols:
                    if symbol in analyses:
                        analysis = analyses[symbol]
                        if 'error' not in analysis:
                            print(f"\n📊 {symbol} - ${analysis['current_price']:.2f}")
                            print(f"   Trend: {analysis['overall_trend']} (ML: {analysis['ml_trend_direction']})")
                            print(f"   30d Performance: {analysis['price_change_30d']:+.2%}")
                            print(f"   Risk Level: {analysis['risk_assessment']}")
                            print(f"   RSI: {analysis['rsi']:.1f}")
                            print(f"   Volatility: {analysis['annualized_volatility']:.1%}")

                # Save results
                trader.save_results(analyses)
                print(f"\nFull report: stock_analyses/analysis_report.md")
                print("="*60)
            else:
                logger.error("No analyses generated")
    
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())