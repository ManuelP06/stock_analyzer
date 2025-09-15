import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import json
import logging
from pathlib import Path
import pickle
import hashlib
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from stock_trader import ProductionStockTrader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnalysisManager:
    """Manages persistent storage and retrieval of stock analyses"""

    def __init__(self, storage_dir="saved_analyses"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        self.analyses_file = self.storage_dir / "analyses_database.json"
        self.predictions_dir = self.storage_dir / "predictions"
        self.predictions_dir.mkdir(exist_ok=True)

    def get_analysis_key(self, symbols, training_period, training_interval):
        """Generate unique key for analysis based on symbols and parameters"""
        # Sort symbols for consistent key generation
        sorted_symbols = sorted(symbols) if isinstance(symbols, list) else [symbols]
        key_data = f"{'-'.join(sorted_symbols)}_{training_period}_{training_interval}"
        return hashlib.md5(key_data.encode()).hexdigest()[:12]

    def save_analysis(self, analysis_record, predictions, signals):
        """Save complete analysis to persistent storage"""
        try:
            symbols = analysis_record['symbols']
            period = analysis_record['training_period']
            interval = analysis_record['training_interval']

            # Generate unique key
            analysis_key = self.get_analysis_key(symbols, period, interval)

            # Load existing analyses database
            analyses_db = self.load_analyses_database()

            # Check if analysis with same symbols exists and remove it
            old_key_to_remove = None
            for key, stored_analysis in analyses_db.items():
                if (set(stored_analysis['symbols']) == set(symbols) and
                    stored_analysis['training_period'] == period and
                    stored_analysis['training_interval'] == interval):
                    old_key_to_remove = key
                    break

            # Remove old analysis files if exists
            if old_key_to_remove and old_key_to_remove != analysis_key:
                self._remove_analysis_files(old_key_to_remove)
                del analyses_db[old_key_to_remove]
                logger.info(f"Removed old analysis for {symbols}")

            # Save new analysis data
            analysis_record['analysis_key'] = analysis_key
            analyses_db[analysis_key] = analysis_record

            # Save predictions and signals
            pred_file = self.predictions_dir / f"{analysis_key}_predictions.json"
            signals_file = self.predictions_dir / f"{analysis_key}_signals.json"

            with open(pred_file, 'w') as f:
                json.dump(predictions, f, indent=2)
            with open(signals_file, 'w') as f:
                json.dump(signals, f, indent=2)

            # Update analyses database
            self.save_analyses_database(analyses_db)

            logger.info(f"Saved analysis {analysis_key} for {symbols}")
            return analysis_key

        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            return None

    def load_analysis(self, analysis_key):
        """Load specific analysis by key"""
        try:
            pred_file = self.predictions_dir / f"{analysis_key}_predictions.json"
            signals_file = self.predictions_dir / f"{analysis_key}_signals.json"

            if not (pred_file.exists() and signals_file.exists()):
                return None, None

            with open(pred_file, 'r') as f:
                predictions = json.load(f)
            with open(signals_file, 'r') as f:
                signals = json.load(f)

            return predictions, signals

        except Exception as e:
            logger.error(f"Failed to load analysis {analysis_key}: {e}")
            return None, None

    def load_analyses_database(self):
        """Load the analyses database"""
        try:
            if self.analyses_file.exists():
                with open(self.analyses_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Failed to load analyses database: {e}")
            return {}

    def save_analyses_database(self, analyses_db):
        """Save the analyses database"""
        try:
            with open(self.analyses_file, 'w') as f:
                json.dump(analyses_db, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save analyses database: {e}")

    def get_all_analyses(self):
        """Get all stored analyses"""
        return self.load_analyses_database()

    def _remove_analysis_files(self, analysis_key):
        """Remove analysis files for a given key"""
        try:
            pred_file = self.predictions_dir / f"{analysis_key}_predictions.json"
            signals_file = self.predictions_dir / f"{analysis_key}_signals.json"

            if pred_file.exists():
                pred_file.unlink()
            if signals_file.exists():
                signals_file.unlink()

        except Exception as e:
            logger.error(f"Failed to remove analysis files for {analysis_key}: {e}")

    def clear_all_analyses(self):
        """Clear all stored analyses (for testing/cleanup)"""
        try:
            # Remove all prediction files
            for file in self.predictions_dir.glob("*.json"):
                file.unlink()

            # Clear database
            self.save_analyses_database({})
            logger.info("Cleared all stored analyses")

        except Exception as e:
            logger.error(f"Failed to clear analyses: {e}")

# Initialize global analysis manager
analysis_manager = AnalysisManager()

# Page configuration for dashboard
st.set_page_config(
    page_title="Stock Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = True
if 'analysis_history' not in st.session_state:
    # Load stored analyses on startup
    stored_analyses = analysis_manager.get_all_analyses()
    st.session_state.analysis_history = list(stored_analyses.values())
if 'current_analysis_key' not in st.session_state:
    st.session_state.current_analysis_key = None

class Charts:
    @staticmethod
    def get_theme_colors(dark_mode=True):
        # Dark trading theme
        if dark_mode:
            return {
                # Chart colors (vibrant for data)
                'bull_green': '#00D4AA',
                'bear_red': '#FF4757',
                'accent': '#5B9BD5',
                'prediction_color': '#FFA726',

                # UI colors (professional dark)
                'background': '#0D1117',
                'surface': '#1C2128',
                'grid_color': '#21262D',
                'text_primary': '#F0F6FC',
                'text_secondary': '#9CA3AF',
                'border': '#30363D',
                'neutral': '#6B7280',
                'button_primary': '#2563EB',
                'button_hover': '#1D4ED8',
                'card_border': '#3B82F6',
                'success': '#10B981',
                'warning': '#F59E0B'
            }
        else:
            return {
                'bull_green': '#10B981',
                'bear_red': '#EF4444',
                'neutral': '#6B7280',
                'background': '#FFFFFF',
                'surface': '#FFFFFF',
                'grid_color': '#E5E7EB',
                'text_primary': '#111827',
                'text_secondary': '#6B7280',
                'border': '#D1D5DB',
                'accent': '#374151',
                'prediction_color': '#059669',
                'volume_color': '#9CA3AF',
                'button_primary': '#10B981',
                'button_hover': '#059669'
            }

    @staticmethod
    def create_price_chart(symbol, data, predictions=None, signals=None, dark_mode=True):
        """Create enhanced price chart with ML predictions"""
        colors = Charts.get_theme_colors(dark_mode)

        # Create subplot with volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.8, 0.2],
            vertical_spacing=0.02
        )

        # Main price chart (Candlesticks)
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price",
                increasing_line_color=colors['bull_green'],
                decreasing_line_color=colors['bear_red'],
                increasing_fillcolor=colors['bull_green'],
                decreasing_fillcolor=colors['bear_red']
            ),
            row=1, col=1
        )

        # Add moving averages
        ma_colors = [colors['accent'], colors['neutral'], colors['prediction_color']]
        for i, period in enumerate([20, 50, 200]):
            if len(data) >= period:
                ma = data['Close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma,
                        name=f"MA{period}",
                        line=dict(color=ma_colors[i], width=1.5, dash='dot'),
                        opacity=0.8
                    ),
                    row=1, col=1
                )

        # Add AI predictions
        if predictions and symbol in predictions:
            pred_data = predictions[symbol]

            if 'predictions' in pred_data and 'dates' in pred_data:
                pred_dates = pd.to_datetime(pred_data['dates'])
                pred_prices = pred_data['predictions']

                # Connection point
                last_price = pred_data.get('last_actual_price', data['Close'].iloc[-1])
                last_date = data.index[-1]

                # Prediction line
                prediction_x = [last_date] + list(pred_dates)
                prediction_y = [last_price] + pred_prices

                fig.add_trace(
                    go.Scatter(
                        x=prediction_x,
                        y=prediction_y,
                        name="AI PREDICTION",
                        line=dict(
                            color=colors['prediction_color'],
                            width=3,
                            dash='dot'
                        ),
                        marker=dict(
                            size=6,
                            color=colors['prediction_color'],
                            symbol='diamond'
                        )
                    ),
                    row=1, col=1
                )

        # Volume chart with colors
        volume_colors = [
            colors['bull_green'] if data['Close'].iloc[i] >= data['Open'].iloc[i]
            else colors['bear_red']
            for i in range(len(data))
        ]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="VOLUME",
                marker_color=volume_colors,
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=750,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text_primary'], size=11, family='JetBrains Mono'),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                bgcolor=colors['surface'],
                bordercolor=colors['border'],
                borderwidth=1,
                x=0.02,
                y=0.98,
                font=dict(size=10)
            ),
            hovermode='x unified',
            margin=dict(l=20, r=20, t=20, b=20)
        )

        fig.update_xaxes(
            gridcolor=colors['grid_color'],
            gridwidth=0.5,
            showgrid=True,
            linecolor=colors['border'],
            color=colors['text_secondary'],
            tickfont=dict(size=10)
        )
        fig.update_yaxes(
            gridcolor=colors['grid_color'],
            gridwidth=0.5,
            showgrid=True,
            linecolor=colors['border'],
            color=colors['text_secondary'],
            tickfont=dict(size=10),
            tickformat='.2f'
        )

        return fig

    @staticmethod
    def create_analysis_price_chart(symbol, data, analysis, dark_mode=True):
        """Create price chart with analysis insights (no trading signals)"""
        colors = Charts.get_theme_colors(dark_mode)

        # Create subplot with volume
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.8, 0.2],
            vertical_spacing=0.02
        )

        # Main price chart (Candlesticks)
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name="Price",
                increasing_line_color=colors['bull_green'],
                decreasing_line_color=colors['bear_red'],
                increasing_fillcolor=colors['bull_green'],
                decreasing_fillcolor=colors['bear_red']
            ),
            row=1, col=1
        )

        # Add moving averages
        ma_colors = [colors['accent'], colors['neutral']]
        for i, period in enumerate([20, 50]):
            if len(data) >= period:
                ma = data['Close'].rolling(window=period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=ma,
                        name=f"MA{period}",
                        line=dict(color=ma_colors[i], width=2, dash='dot'),
                        opacity=0.8
                    ),
                    row=1, col=1
                )

        # Add support and resistance lines if available in analysis
        if 'support_level' in analysis and 'resistance_level' in analysis:
            fig.add_hline(
                y=analysis['support_level'],
                line_dash="dash",
                line_color=colors['bull_green'],
                opacity=0.6,
                row=1, col=1,
                annotation_text="Support"
            )
            fig.add_hline(
                y=analysis['resistance_level'],
                line_dash="dash",
                line_color=colors['bear_red'],
                opacity=0.6,
                row=1, col=1,
                annotation_text="Resistance"
            )

        # Volume chart
        volume_colors = [
            colors['bull_green'] if data['Close'].iloc[i] >= data['Open'].iloc[i]
            else colors['bear_red']
            for i in range(len(data))
        ]

        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name="VOLUME",
                marker_color=volume_colors,
                opacity=0.6,
                showlegend=False
            ),
            row=2, col=1
        )

        fig.update_layout(
            height=750,
            title=f"{symbol} - Analysis",
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text_primary'], size=11, family='JetBrains Mono'),
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                bgcolor=colors['surface'],
                bordercolor=colors['border'],
                borderwidth=1,
                x=0.02,
                y=0.98,
                font=dict(size=10)
            ),
            hovermode='x unified',
            margin=dict(l=20, r=20, t=40, b=20)
        )

        fig.update_xaxes(
            gridcolor=colors['grid_color'],
            gridwidth=0.5,
            showgrid=True,
            linecolor=colors['border'],
            color=colors['text_secondary'],
            tickfont=dict(size=10)
        )
        fig.update_yaxes(
            gridcolor=colors['grid_color'],
            gridwidth=0.5,
            showgrid=True,
            linecolor=colors['border'],
            color=colors['text_secondary'],
            tickfont=dict(size=10),
            tickformat='.2f'
        )

        return fig

    @staticmethod
    def create_technical_dashboard(symbol, data, dark_mode=True):
        """Create comprehensive technical analysis dashboard"""
        colors = Charts.get_theme_colors(dark_mode)

        fig = make_subplots(
            rows=3, cols=3,
            shared_xaxes=True,
            subplot_titles=[
                "RSI", "MACD", "BOLLINGER %B",
                "STOCHASTIC", "WILLIAMS %R", "CCI",
                "ADX", "ATR", "VOLUME OBV"
            ],
            row_heights=[0.33, 0.33, 0.34],
            horizontal_spacing=0.06,
            vertical_spacing=0.08
        )

        # Calculate technical indicators
        close = data['Close']
        high = data['High']
        low = data['Low']
        volume = data['Volume']

        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        fig.add_trace(
            go.Scatter(x=data.index, y=rsi, name="RSI",
                      line=dict(color=colors['accent'], width=2)),
            row=1, col=1
        )
        fig.add_hline(y=70, line_dash="dot", line_color=colors['bear_red'], opacity=0.6, row=1, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color=colors['bull_green'], opacity=0.6, row=1, col=1)

        # MACD
        exp1 = close.ewm(span=12).mean()
        exp2 = close.ewm(span=26).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9).mean()
        histogram = macd_line - signal_line

        fig.add_trace(
            go.Scatter(x=data.index, y=macd_line, name="MACD",
                      line=dict(color=colors['accent'], width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=signal_line, name="SIGNAL",
                      line=dict(color=colors['prediction_color'], width=2)),
            row=1, col=2
        )

        colors_hist = [colors['bull_green'] if val >= 0 else colors['bear_red'] for val in histogram]
        fig.add_trace(
            go.Bar(x=data.index, y=histogram, name="HIST",
                   marker_color=colors_hist, opacity=0.5),
            row=1, col=2
        )

        # Bollinger %B
        bb_middle = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        bb_percent = (close - bb_lower) / (bb_upper - bb_lower)

        fig.add_trace(
            go.Scatter(x=data.index, y=bb_percent * 100, name="BB%B",
                      line=dict(color=colors['accent'], width=2)),
            row=1, col=3
        )
        fig.add_hline(y=80, line_dash="dot", line_color=colors['bear_red'], opacity=0.6, row=1, col=3)
        fig.add_hline(y=20, line_dash="dot", line_color=colors['bull_green'], opacity=0.6, row=1, col=3)

        # Stochastic
        lowest_low = low.rolling(window=14).min()
        highest_high = high.rolling(window=14).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=3).mean()

        fig.add_trace(
            go.Scatter(x=data.index, y=k_percent, name="%K",
                      line=dict(color=colors['accent'], width=2)),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=d_percent, name="%D",
                      line=dict(color=colors['prediction_color'], width=2)),
            row=2, col=1
        )
        fig.add_hline(y=80, line_dash="dot", line_color=colors['bear_red'], opacity=0.6, row=2, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color=colors['bull_green'], opacity=0.6, row=2, col=1)

        # Williams %R
        williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
        fig.add_trace(
            go.Scatter(x=data.index, y=williams_r, name="WILLIAMS%R",
                      line=dict(color=colors['prediction_color'], width=2)),
            row=2, col=2
        )
        fig.add_hline(y=-20, line_dash="dot", line_color=colors['bear_red'], opacity=0.6, row=2, col=2)
        fig.add_hline(y=-80, line_dash="dot", line_color=colors['bull_green'], opacity=0.6, row=2, col=2)

        # CCI
        typical_price = (high + low + close) / 3
        sma_tp = typical_price.rolling(20).mean()
        mad_tp = (typical_price - sma_tp).abs().rolling(20).mean()
        cci = (typical_price - sma_tp) / (0.015 * mad_tp)

        fig.add_trace(
            go.Scatter(x=data.index, y=cci, name="CCI",
                      line=dict(color=colors['accent'], width=2)),
            row=2, col=3
        )
        fig.add_hline(y=100, line_dash="dot", line_color=colors['bear_red'], opacity=0.6, row=2, col=3)
        fig.add_hline(y=-100, line_dash="dot", line_color=colors['bull_green'], opacity=0.6, row=2, col=3)

        # ADX
        tr_components = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift(1)).abs(),
            'lc': (low - close.shift(1)).abs()
        })
        tr = tr_components.max(axis=1)
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = ((up_move > down_move) & (up_move > 0)) * up_move
        minus_dm = ((down_move > up_move) & (down_move > 0)) * down_move
        atr_smooth = tr.ewm(alpha=1/14).mean()
        plus_di_smooth = plus_dm.ewm(alpha=1/14).mean()
        minus_di_smooth = minus_dm.ewm(alpha=1/14).mean()
        plus_di = 100 * plus_di_smooth / atr_smooth
        minus_di = 100 * minus_di_smooth / atr_smooth
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.ewm(alpha=1/14).mean()

        fig.add_trace(
            go.Scatter(x=data.index, y=adx, name="ADX",
                      line=dict(color=colors['accent'], width=2)),
            row=3, col=1
        )
        fig.add_hline(y=25, line_dash="dot", line_color=colors['neutral'], opacity=0.6, row=3, col=1)

        # ATR
        atr = tr.rolling(14).mean()
        atr_pct = (atr / close) * 100

        fig.add_trace(
            go.Scatter(x=data.index, y=atr_pct, name="ATR%",
                      line=dict(color=colors['prediction_color'], width=2)),
            row=3, col=2
        )

        # OBV
        price_change = close.diff()
        obv_change = volume * np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
        obv = obv_change.cumsum()
        obv_ma = obv.rolling(20).mean()

        fig.add_trace(
            go.Scatter(x=data.index, y=obv, name="OBV",
                      line=dict(color=colors['accent'], width=2)),
            row=3, col=3
        )
        fig.add_trace(
            go.Scatter(x=data.index, y=obv_ma, name="OBV_MA",
                      line=dict(color=colors['prediction_color'], width=1, dash='dot')),
            row=3, col=3
        )

        fig.update_layout(
            height=900,
            plot_bgcolor=colors['background'],
            paper_bgcolor=colors['background'],
            font=dict(color=colors['text_primary'], family='JetBrains Mono', size=9),
            showlegend=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )

        fig.update_xaxes(
            gridcolor=colors['grid_color'],
            gridwidth=0.3,
            showgrid=True,
            tickfont=dict(size=9)
        )
        fig.update_yaxes(
            gridcolor=colors['grid_color'],
            gridwidth=0.3,
            showgrid=True,
            tickfont=dict(size=9)
        )

        return fig

def load_css(dark_mode=True):
    colors = Charts.get_theme_colors(dark_mode)

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');

    /* Modern Dark Trading Layout */
    .stApp {{
        background: {colors['background']};
        color: {colors['text_primary']};
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }}

    .main .block-container {{
        padding: 1rem;
        max-width: 100%;
    }}

    #MainMenu, header, footer, .stDeployButton, .stDecoration {{
        display: none !important;
    }}

    div[data-baseweb="tab-list"] {{
        border-bottom: none !important;
    }}

    div[data-baseweb="tab-list"]:after {{
        display: none !important;
    }}

    .stTabs [data-testid="stTabs"] > div:first-child {{
        border-bottom: none !important;
        background: none !important;
    }}

    .css-1d391kg {{
        background: {colors['surface']} !important;
        border-right: 1px solid {colors['card_border']} !important;
        padding: 1rem !important;
    }}

    .css-1d391kg h4 {{
        color: {colors['text_primary']} !important;
        font-family: 'Inter', sans-serif !important;
        margin-bottom: 1rem !important;
    }}

    .streamlit-expanderHeader {{
        background: {colors['surface']} !important;
        border: 1px solid {colors['card_border']} !important;
        border-radius: 6px !important;
    }}

    .streamlit-expanderContent {{
        background: {colors['surface']} !important;
        border: 1px solid {colors['card_border']} !important;
        border-top: none !important;
        border-radius: 0 0 6px 6px !important;
    }}

    .metric-card {{
        background: {colors['surface']};
        border: 1px solid {colors['border']};
        border-radius: 8px;
        padding: 1.25rem;
        margin: 0.5rem 0;
        transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }}

    .metric-card:hover {{
        border-color: {colors['card_border']};
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.15);
    }}

    .metric-value {{
        font-size: 1.75rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        margin: 0;
        text-align: left;
    }}

    .metric-label {{
        font-size: 0.8rem;
        color: {colors['text_secondary']};
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-bottom: 0.5rem;
    }}

    .signal-badge {{
        padding: 0.75rem 1.25rem;
        border-radius: 6px;
        font-weight: 700;
        font-size: 1rem;
        font-family: 'Inter', sans-serif;
        text-align: center;
        margin: 0.75rem 0;
        border: 2px solid;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        background: {colors['surface']};
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }}

    .signal-buy {{
        color: {colors['bull_green']};
        border-color: {colors['bull_green']};
        background: rgba(0, 212, 170, 0.1);
        font-size: 1.1rem;
    }}

    .signal-sell {{
        color: {colors['bear_red']};
        border-color: {colors['bear_red']};
        background: rgba(255, 71, 87, 0.1);
        font-size: 1.1rem;
    }}

    .signal-hold {{
        color: {colors['text_secondary']};
        border-color: {colors['card_border']};
    }}

    .stButton > button {{
        background: {colors['button_primary']} !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-family: 'Inter', sans-serif !important;
        padding: 0.75rem 1.5rem !important;
        font-size: 0.9rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1) !important;
    }}

    .stButton > button:hover {{
        background: {colors['button_hover']} !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15) !important;
    }}

    .positive {{ color: {colors['bull_green']}; font-weight: 600; }}
    .negative {{ color: {colors['bear_red']}; font-weight: 600; }}
    .neutral {{ color: {colors['neutral']}; font-weight: 500; }}

    .stSelectbox > div > div > div,
    .stTextInput > div > div > input,
    .stMultiSelect > div > div > div {{
        background: {colors['surface']} !important;
        border: 1px solid {colors['border']} !important;
        color: {colors['text_primary']} !important;
        border-radius: 6px !important;
    }}

    .stSlider > div > div > div > div {{
        background: {colors['accent']} !important;
    }}

    .stDataFrame {{
        background: {colors['surface']};
        border: 1px solid {colors['border']};
    }}

    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: none !important;
        background: transparent !important;
    }}

    .stTabs [data-baseweb="tab-list"]::after {{
        display: none !important;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: {colors['surface']} !important;
        border: 1px solid {colors['border']} !important;
        border-bottom: none !important;
        border-radius: 6px !important;
        color: {colors['text_secondary']} !important;
        padding: 0.75rem 1.5rem !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 0.9rem !important;
        font-weight: 500 !important;
        position: relative !important;
        z-index: 10 !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        background: {colors['button_hover']} !important;
        color: white !important;
    }}

    .stTabs [aria-selected="true"] {{
        background: {colors['button_primary']} !important;
        color: white !important;
        border-color: {colors['button_primary']} !important;
        border-bottom: none !important;
        position: relative !important;
        z-index: 10 !important;
    }}

    .stTabs [data-baseweb="tab-border"] {{
        display: none !important;
    }}

    .stTabs::before,
    .stTabs::after,
    .stTabs [data-baseweb="tab"]::before,
    .stTabs [data-baseweb="tab"]::after {{
        display: none !important;
    }}

    .stTabs > div > div > div {{
        border: none !important;
        border-top: none !important;
        margin-top: 0 !important;
        padding-top: 1rem !important;
    }}

    .stTabs [role="tablist"] {{
        border-bottom: none !important;
        background: transparent !important;
    }}

    .stTabs [role="tablist"]::after {{
        content: none !important;
        display: none !important;
    }}

    .analysis-history {{
        background: linear-gradient(135deg, {colors['surface']} 0%, rgba(59, 130, 246, 0.05) 100%);
        border: 1px solid {colors['card_border']};
        border-radius: 6px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }}

    .batch-header {{
        color: {colors['card_border']};
        font-weight: 700;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }}

    .analysis-details {{
        color: {colors['text_secondary']};
        line-height: 1.4;
    }}

    </style>
    """, unsafe_allow_html=True)

def get_stock_info(symbol):
    """Get basic stock information"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            'name': info.get('longName', symbol),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 0),
        }
    except:
        return {'name': symbol, 'sector': 'N/A', 'market_cap': 0}

def validate_stock_symbol(symbol):
    """Validate if stock symbol exists"""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="5d")
        return len(hist) > 0
    except:
        return False

def format_market_cap(market_cap):
    """Format market cap in readable format"""
    if market_cap >= 1e12:
        return f"${market_cap/1e12:.1f}T"
    elif market_cap >= 1e9:
        return f"${market_cap/1e9:.1f}B"
    elif market_cap >= 1e6:
        return f"${market_cap/1e6:.1f}M"
    else:
        return f"${market_cap:,.0f}"

def render_analysis_overview(analyses):
    """Render professional analysis overview"""
    if not analyses:
        return

    # Create overview table
    cols = st.columns(min(4, len(analyses)))

    for i, (symbol, analysis) in enumerate(list(analyses.items())[:4]):
        if 'error' not in analysis:
            # Determine trend color
            trend = analysis.get('overall_trend', 'Neutral')
            if trend == 'Bullish':
                trend_class = 'positive'
            elif trend == 'Bearish':
                trend_class = 'negative'
            else:
                trend_class = 'neutral'

            with cols[i]:
                st.markdown(f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: 1.2rem; font-weight: 700; font-family: 'JetBrains Mono', monospace;">{symbol}</h3>
                    <div class="metric-value" style="font-size: 1.5rem;">${analysis['current_price']:.2f}</div>
                    <div class="metric-label">CURRENT PRICE</div>
                    <div style="margin-top: 1rem; color: #8B949E; font-size: 0.8rem; font-family: 'Inter', sans-serif;">
                        <div class="{trend_class}" style="font-weight: 600; margin: 0.2rem 0;">
                            {trend} Trend
                        </div>
                        <div>30d: <span class="{'positive' if analysis['price_change_30d'] > 0 else 'negative'}">{analysis['price_change_30d']:+.1%}</span></div>
                        <div>Risk: {analysis['risk_assessment']}</div>
                        <div>ML: {analysis['ml_trend_direction']}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

def render_stock_analysis(analysis):
    """Render detailed analysis for a single stock"""
    st.markdown(f"### {analysis['symbol']} - Detailed Analysis")

    # Performance metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Performance")
        st.metric("Current Price", f"${analysis['current_price']:.2f}")
        st.metric("1-Day Change", f"{analysis['price_change_1d']:+.2%}")
        st.metric("30-Day Return", f"{analysis['price_change_30d']:+.2%}")
        st.metric("90-Day Return", f"{analysis['price_change_90d']:+.2%}")

    with col2:
        st.markdown("#### Technical Indicators")
        st.metric("RSI (14)", f"{analysis['rsi']:.1f}")
        st.metric("MA20", f"${analysis['ma20']:.2f}")
        st.metric("MA50", f"${analysis['ma50']:.2f}")

        # Price position indicator
        position_pct = analysis['price_position'] * 100
        st.metric("Price Position", f"{position_pct:.0f}%",
                 help="Position between support (0%) and resistance (100%)")

    with col3:
        st.markdown("#### Risk Assessment")
        st.metric("Daily Volatility", f"{analysis['daily_volatility']:.2%}")
        st.metric("Annual Volatility", f"{analysis['annualized_volatility']:.1%}")
        st.metric("Risk Level", analysis['risk_assessment'])

        # Support/Resistance levels
        st.markdown("**Key Levels**")
        st.write(f"Support: ${analysis['support_level']:.2f}")
        st.write(f"Resistance: ${analysis['resistance_level']:.2f}")

    # Insights section
    st.markdown("#### Machine Learning Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Trend Direction", analysis['ml_trend_direction'])
        st.metric("Confidence Level", f"{analysis['ml_trend_confidence']:.0%}")

    with col2:
        st.metric("Volatility Outlook", analysis['ml_volatility_outlook'])
        st.metric("Pattern Consistency", f"{analysis['ml_trend_consistency']:.0%}")

    # Notes and disclaimers
    st.info(f"**Analysis Note:** {analysis['ml_notes']}")

    st.markdown("---")
    st.markdown(f"*Analysis generated: {analysis['analysis_date']}*")

@st.cache_data
def load_stock_data(symbol, period="1y"):
    """Load stock data with caching"""
    try:
        ticker = yf.Ticker(symbol)
        data = ticker.history(period=period)
        if len(data) == 0:
            return None
        return data
    except Exception as e:
        st.error(f"Error loading data for {symbol}: {e}")
        return None

def main():
    """Main dashboard application"""
    # Load dark CSS theme
    load_css(True)

    # Show control panel if toggled
    if st.session_state.get('show_controls', True):
        with st.expander("CONTROL PANEL", expanded=True):

            # Analysis History Section
            st.markdown("#### SAVED ANALYSES")
            if st.session_state.analysis_history:
                # Create analysis selector
                analysis_options = {}
                for analysis in st.session_state.analysis_history:
                    symbols_str = ', '.join(analysis['symbols'])
                    period = analysis.get('training_period', 'Unknown')
                    interval = analysis.get('training_interval', '1d')
                    timestamp = analysis.get('timestamp', 'Unknown')

                    display_name = f"{symbols_str} [{period}/{interval}] - {timestamp}"
                    analysis_key = analysis.get('analysis_key', None)
                    if analysis_key:
                        analysis_options[display_name] = analysis_key

                if analysis_options:
                    st.markdown("**Load Previous Analysis:**")
                    selected_analysis = st.selectbox(
                        "Select Analysis",
                        options=["None"] + list(analysis_options.keys()),
                        index=0,
                        key="analysis_selector"
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        if selected_analysis != "None":
                            if st.button("Load Selected", type="secondary"):
                                analysis_key = analysis_options[selected_analysis]
                                st.session_state.current_analysis_key = analysis_key
                                st.success(f"Loaded: {selected_analysis}")
                                st.rerun()
                    with col2:
                        if st.session_state.current_analysis_key:
                            if st.button("Clear Current", type="secondary"):
                                st.session_state.current_analysis_key = None
                                st.success("Cleared current analysis")
                                st.rerun()

                # Show recent analyses (display only)
                st.markdown("**Recent Analyses:**")
                for i, analysis in enumerate(reversed(st.session_state.analysis_history[-3:])):
                    batch_num = len(st.session_state.analysis_history) - i
                    training_period_display = analysis.get('training_period', 'Unknown')
                    training_interval_display = analysis.get('training_interval', '1d')
                    st.markdown(f"""
                    <div class="analysis-history">
                        <div class="batch-header">#{batch_num}</div>
                        <div class="analysis-details">
                            <strong>Symbols:</strong> {', '.join(analysis['symbols'])}<br>
                            <strong>Period:</strong> {training_period_display} ({training_interval_display})<br>
                            <strong>Time:</strong> {analysis['timestamp']}<br>
                            <strong>Model:</strong> E:{analysis['epochs']} S:{analysis['sequence_length']} P:{analysis['prediction_horizon']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No saved analyses yet")

            st.markdown("#### ANALYZE NEW STOCKS")

            # Popular stocks selection
            popular_stocks = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMZN', 'META', 'NFLX', 'AMD', 'CRM']
            selected_popular = st.multiselect(
                "Popular Stocks",
                options=popular_stocks,
                default=[]
            )

            # Custom stock input
            custom_input = st.text_input(
                "Custom Stocks",
                placeholder="Enter symbols: TSLA,GME,AMC"
            )

            # Parse and combine stocks
            custom_stocks = []
            if custom_input:
                custom_stocks = [s.strip().upper() for s in custom_input.split(',') if s.strip()]

            all_stocks = list(set(selected_popular + custom_stocks))

            # Show selected stocks
            if all_stocks:
                st.write("**Selected Stocks:**")
                valid_stocks = []
                for symbol in all_stocks:
                    if validate_stock_symbol(symbol):
                        valid_stocks.append(symbol)
                        info = get_stock_info(symbol)
                        st.success(f"✓ {symbol} - {info['name'][:25]}{'...' if len(info['name']) > 25 else ''}")
                    else:
                        st.error(f"✗ {symbol} - Invalid")

                if valid_stocks:
                    st.markdown("**Training Parameters:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        epochs = st.slider("Epochs", 10, 100, 25)
                        sequence_length = st.slider("Sequence Length", 30, 200, 60)
                    with col2:
                        prediction_horizon = st.slider("Prediction Days", 1, 30, 7)

                        # Time span selection for training data
                        period_options = {
                            "6 months": "6mo",
                            "1 year": "1y",
                            "2 years": "2y",
                            "3 years": "3y",
                            "5 years": "5y",
                            "10 years": "10y"
                        }
                        selected_period_name = st.selectbox(
                            "Training Data Period",
                            options=list(period_options.keys()),
                            index=2  # Default to "2 years"
                        )
                        training_period = period_options[selected_period_name]

                        # Interval selection for data granularity
                        interval_options = {
                            "1 minute": "1m",
                            "5 minutes": "5m",
                            "15 minutes": "15m",
                            "30 minutes": "30m",
                            "1 hour": "1h",
                            "1 day": "1d",
                            "1 week": "1wk",
                            "1 month": "1mo"
                        }
                        selected_interval_name = st.selectbox(
                            "Data Interval",
                            options=list(interval_options.keys()),
                            index=5  # Default to "1 day"
                        )
                        training_interval = interval_options[selected_interval_name]

                    if st.button("START ANALYSIS", type="primary", use_container_width=True):
                        symbols = valid_stocks
                        analyze_button = True
                    else:
                        symbols = None
                        analyze_button = False
                else:
                    symbols = None
                    analyze_button = False
            else:
                symbols = None
                analyze_button = False

    # Load existing analyses - check for loaded analysis first, then current session results
    analyses = {}

    # First, try to load a selected analysis
    if st.session_state.current_analysis_key:
        try:
            loaded_analyses, _ = analysis_manager.load_analysis(st.session_state.current_analysis_key)
            if loaded_analyses:
                analyses = loaded_analyses
                logger.info(f"Loaded analysis {st.session_state.current_analysis_key}")
        except Exception as e:
            logger.error(f"Error loading selected analysis: {e}")

    # If no loaded analysis, try current session results
    if not analyses:
        try:
            results_path = Path("stock_analyses")
            if results_path.exists():
                analysis_file = results_path / "analyses.json"
                if analysis_file.exists():
                    with open(analysis_file) as f:
                        analyses = json.load(f)
        except Exception as e:
            logger.error(f"Error loading current session results: {e}")

    # Run analysis
    if analyze_button and symbols:
        with st.spinner("Training AI model and generating predictions..."):
            try:
                trader = ProductionStockTrader()

                progress_bar = st.progress(0)
                status_text = st.empty()

                status_text.text("Training transformer model...")
                progress_bar.progress(25)

                # Train with error handling
                try:
                    trader.train(
                        symbols=symbols,
                        epochs=epochs,
                        seq_len=sequence_length,
                        pred_len=prediction_horizon,
                        period=training_period,
                        interval=training_interval
                    )
                except Exception as train_error:
                    st.error(f"Training failed: {train_error}")
                    logger.error(f"Training error: {train_error}")
                    return

                status_text.text("Generating professional analysis...")
                progress_bar.progress(75)

                try:
                    analyses = trader.analyze_stocks(symbols, interval=training_interval)
                    trader.save_results(analyses)
                except Exception as analysis_error:
                    st.error(f"Analysis failed: {analysis_error}")
                    logger.error(f"Analysis error: {analysis_error}")
                    return

                # Add to analysis history
                sample_data = load_stock_data(symbols[0], "1y")
                if sample_data is not None and len(sample_data) > 0:
                    date_range = f"{sample_data.index[0].strftime('%Y-%m-%d')} to {sample_data.index[-1].strftime('%Y-%m-%d')}"
                    timespan = f"{len(sample_data)} days"
                else:
                    date_range = "Unknown"
                    timespan = "Unknown"

                analysis_record = {
                    'symbols': symbols,
                    'epochs': epochs,
                    'sequence_length': sequence_length,
                    'prediction_horizon': prediction_horizon,
                    'training_period': training_period,
                    'training_interval': training_interval,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
                    'date_range': date_range,
                    'timespan': timespan
                }

                # Save analysis persistently
                analysis_key = analysis_manager.save_analysis(analysis_record, analyses, {})
                if analysis_key:
                    analysis_record['analysis_key'] = analysis_key
                    st.session_state.current_analysis_key = analysis_key

                    # Update session state with new analyses
                    stored_analyses = analysis_manager.get_all_analyses()
                    st.session_state.analysis_history = list(stored_analyses.values())

                progress_bar.progress(100)
                status_text.text("Analysis completed!")

                st.success("AI analysis completed successfully!")
                st.rerun()

            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                logger.error(f"Analysis error: {e}")

    # Display current analysis info
    if st.session_state.current_analysis_key and analyses:
        # Show which analysis is currently loaded
        current_analysis = None
        for analysis in st.session_state.analysis_history:
            if analysis.get('analysis_key') == st.session_state.current_analysis_key:
                current_analysis = analysis
                break

        if current_analysis:
            symbols_str = ', '.join(current_analysis['symbols'])
            period = current_analysis.get('training_period', 'Unknown')
            interval = current_analysis.get('training_interval', '1d')
            timestamp = current_analysis.get('timestamp', 'Unknown')

            st.info(f"Currently viewing: **{symbols_str}** [{period}/{interval}] - {timestamp}")

    # Display analysis results
    if analyses:
        st.markdown("## Stock Analysis")

        # Create overview metrics
        render_analysis_overview(analyses)

        # Stock selector for detailed analysis
        chart_symbol = st.selectbox(
            "Select Stock for Detailed Analysis",
            options=list(analyses.keys()),
            index=0
        )

        if chart_symbol and chart_symbol in analyses:
            analysis = analyses[chart_symbol]
            if 'error' not in analysis:
                # Display detailed analysis
                render_stock_analysis(analysis)

                # Show charts
                stock_data = load_stock_data(chart_symbol, "6mo")
                if stock_data is not None and len(stock_data) > 0:
                    tab1, tab2 = st.tabs(["Price Chart", "Technical Analysis"])

                    with tab1:
                        # Create simplified chart without trading signals
                        chart = Charts.create_analysis_price_chart(
                            chart_symbol, stock_data, analysis, True
                        )
                        st.plotly_chart(chart, use_container_width=True, config={'displayModeBar': False})

                    with tab2:
                        tech_chart = Charts.create_technical_dashboard(
                            chart_symbol, stock_data, True
                        )
                        st.plotly_chart(tech_chart, use_container_width=True, config={'displayModeBar': False})
                else:
                    st.error(f"Could not load chart data for {chart_symbol}")
            else:
                st.error(f"Analysis error for {chart_symbol}: {analysis.get('error', 'Unknown error')}")

    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 1rem; color: #8B949E;">
            <h2 style="font-family: 'JetBrains Mono', monospace; font-size: 2rem; margin-bottom: 1rem;">Stock Analyzer & Research</h2>
            <p style="font-size: 1rem;">Select stocks and configure analysis parameters to begin market research</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()