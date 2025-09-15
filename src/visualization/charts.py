"""
Professional charting library for stock analysis
Clean, minimalistic design focused on data visualization
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class StockCharts:
    """
    Professional stock charting library with clean design
    """
    
    # Professional trading color palette
    COLORS = {
        'primary': '#0066cc',      # Professional Blue
        'secondary': '#4a5568',    # Charcoal Gray
        'success': '#00c851',      # Market Green
        'danger': '#ff4757',       # Alert Red
        'warning': '#ffc107',      # Caution Amber
        'info': '#17a2b8',         # Information Teal
        'light': '#f8f9fa',        # Clean Background
        'dark': '#2c3e50',         # Dark Navy
        'background': '#ffffff',   # Pure White
        'grid': '#e9ecef',         # Subtle Grid
        'text': '#2c3e50',         # Professional Text
        'text_secondary': '#6c757d', # Muted Text
        'bullish': '#00c851',      # Bull Market Green
        'bearish': '#ff4757',      # Bear Market Red
        'neutral': '#6c757d',      # Neutral Gray
        'accent': '#007bff',       # Accent Blue
        'border': '#dee2e6'        # Border Gray
    }
    
    @staticmethod
    def create_price_chart(
        df: pd.DataFrame,
        predictions: Optional[Dict] = None,
        signals: Optional[Dict] = None,
        symbol: str = "Stock",
        show_volume: bool = True,
        height: int = 600
    ) -> go.Figure:
        """
        Create professional price chart with predictions and signals
        """
        
        # Create professional subplots
        if show_volume and 'volume' in df.columns:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.75, 0.25],
                subplot_titles=(f"{symbol} PRICE CHART", "VOLUME"),
                vertical_spacing=0.08
            )
        else:
            fig = make_subplots(rows=1, cols=1)
        
        # Main price chart (candlestick)
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            fig.add_trace(
                go.Candlestick(
                    x=df.index,
                    open=df['open'],
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    name="Price",
                    increasing_line_color=StockCharts.COLORS['success'],
                    decreasing_line_color=StockCharts.COLORS['danger'],
                    showlegend=False
                ),
                row=1, col=1
            )
        elif 'close' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['close'],
                    name="Price",
                    line=dict(color=StockCharts.COLORS['primary'], width=2),
                    showlegend=False
                ),
                row=1, col=1
            )
        
        # Add moving averages if available
        ma_columns = [col for col in df.columns if col.startswith('sma_') or col.startswith('ema_')]
        colors = [StockCharts.COLORS['secondary'], StockCharts.COLORS['info'], StockCharts.COLORS['warning']]
        
        for i, ma_col in enumerate(ma_columns[:3]):
            if ma_col in df.columns:
                ma_type = ma_col.split('_')[0].upper()
                ma_period = ma_col.split('_')[1]
                fig.add_trace(
                    go.Scatter(
                        x=df.index,
                        y=df[ma_col],
                        name=f"{ma_type}{ma_period}",
                        line=dict(
                            color=colors[i % len(colors)],
                            width=1.5,
                            dash='dot' if ma_type == 'EMA' else 'solid'
                        ),
                        opacity=0.8
                    ),
                    row=1, col=1
                )
        
        # Add predictions if available
        if predictions and symbol in predictions:
            pred_data = predictions[symbol]
            if 'dates' in pred_data and 'predictions' in pred_data:
                pred_dates = pd.to_datetime(pred_data['dates'])
                pred_values = pred_data['predictions']
                
                # Get proper connection point (CRITICAL FIX)
                try:
                    last_date = df.index[-1]
                    last_price = pred_data.get('last_actual_price', float(df['close'].iloc[-1]))
                except (IndexError, KeyError, AttributeError):
                    # Fallback if df indexing fails
                    last_date = df.index.max() if len(df.index) > 0 else datetime.now()
                    last_price = pred_data.get('last_actual_price', 100.0)
                
                # Create seamless connection
                pred_x = [last_date] + list(pred_dates)
                pred_y = [float(last_price)] + [float(p) for p in pred_values]
                
                # Add prediction line with enhanced visibility
                fig.add_trace(
                    go.Scatter(
                        x=pred_x,
                        y=pred_y,
                        name="AI Prediction",
                        line=dict(
                            color=StockCharts.COLORS['warning'],
                            width=4,
                            dash='dot'
                        ),
                        marker=dict(
                            size=8, 
                            color=StockCharts.COLORS['warning'],
                            symbol='circle',
                            line=dict(width=2, color='white')
                        ),
                        hovertemplate='<b>Prediction</b><br>Date: %{x}<br>Price: $%{y:.2f}<extra></extra>'
                    ),
                    row=1, col=1
                )
                
                # Add confidence interval if volatility data available
                if 'volatility' in pred_data:
                    volatility = pred_data['volatility']
                    upper_bound = [p * (1 + volatility) for p in pred_values]
                    lower_bound = [p * (1 - volatility) for p in pred_values]
                    
                    # Upper bound
                    fig.add_trace(
                        go.Scatter(
                            x=list(pred_dates),
                            y=upper_bound,
                            mode='lines',
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo='skip'
                        ),
                        row=1, col=1
                    )
                    
                    # Lower bound with fill
                    fig.add_trace(
                        go.Scatter(
                            x=list(pred_dates),
                            y=lower_bound,
                            mode='lines',
                            line=dict(width=0),
                            fillcolor='rgba(255, 193, 7, 0.2)',
                            fill='tonexty',
                            name='Confidence Interval',
                            hoverinfo='skip'
                        ),
                        row=1, col=1
                    )
        
        # Add trading signals if available
        if signals and symbol in signals:
            signal_data = signals[symbol]
            signal_type = signal_data.get('signal', 'HOLD')
            confidence = signal_data.get('confidence', 'LOW')
            
            if signal_type != 'HOLD':
                color = StockCharts.COLORS['success'] if signal_type == 'BUY' else StockCharts.COLORS['danger']
                arrow = '▲' if signal_type == 'BUY' else '▼'
                
                # Get position for signal annotation
                signal_y = df['close'].iloc[-1]
                if predictions and symbol in predictions:
                    # Position signal near first prediction point
                    signal_y = predictions[symbol].get('last_actual_price', signal_y)
                
                fig.add_annotation(
                    x=df.index[-1],
                    y=signal_y,
                    text=f"{arrow} {signal_type}<br>{confidence}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=color,
                    bgcolor=color,
                    bordercolor='white',
                    borderwidth=2,
                    font=dict(color='white', size=11, family='Arial Black'),
                    xshift=20,
                    yshift=20,
                    row=1, col=1
                )
        
        # Add volume chart
        if show_volume and 'volume' in df.columns:
            colors = [StockCharts.COLORS['success'] if df['close'].iloc[i] >= df['open'].iloc[i] 
                     else StockCharts.COLORS['danger'] for i in range(len(df))]
            
            fig.add_trace(
                go.Bar(
                    x=df.index,
                    y=df['volume'],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.7,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # Professional trading layout with enhanced prediction visibility
        fig.update_layout(
            title=dict(
                text=f"{symbol} - Professional Analysis",
                x=0.02,
                font=dict(size=24, color=StockCharts.COLORS['text'], weight='bold')
            ),
            height=height,
            xaxis_rangeslider_visible=False,
            plot_bgcolor=StockCharts.COLORS['background'],
            paper_bgcolor=StockCharts.COLORS['background'],
            font=dict(color=StockCharts.COLORS['text'], size=12),
            margin=dict(l=60, r=60, t=80, b=60),
            showlegend=True,
            legend=dict(
                bgcolor="rgba(255,255,255,0.95)",
                bordercolor=StockCharts.COLORS['border'],
                borderwidth=1,
                x=0.02,
                y=0.98,
                font=dict(size=11)
            ),
            hovermode='x unified'
        )
        
        # Clean axes
        fig.update_xaxes(
            gridcolor=StockCharts.COLORS['grid'],
            gridwidth=0.5,
            showgrid=True,
            zeroline=False
        )
        fig.update_yaxes(
            gridcolor=StockCharts.COLORS['grid'],
            gridwidth=0.5,
            showgrid=True,
            zeroline=False
        )
        
        return fig
    
    @staticmethod
    def create_indicators_chart(
        df: pd.DataFrame,
        symbol: str = "Stock",
        height: int = 500
    ) -> go.Figure:
        """
        Create technical indicators chart
        """
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            row_heights=[0.35, 0.35, 0.3],
            subplot_titles=("RELATIVE STRENGTH INDEX", "MACD OSCILLATOR", "BOLLINGER BAND POSITION"),
            vertical_spacing=0.08
        )
        
        # RSI
        if any(col.startswith('rsi_') for col in df.columns):
            rsi_col = next(col for col in df.columns if col.startswith('rsi_'))
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[rsi_col],
                    name="RSI",
                    line=dict(color=StockCharts.COLORS['primary'], width=2)
                ),
                row=1, col=1
            )
            
            # Add RSI levels
            for level, color in [(70, StockCharts.COLORS['danger']), (30, StockCharts.COLORS['success'])]:
                fig.add_hline(
                    y=level,
                    line_dash="dash",
                    line_color=color,
                    opacity=0.5,
                    row=1, col=1
                )
        
        # MACD
        if 'macd_line' in df.columns and 'macd_signal' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_line'],
                    name="MACD",
                    line=dict(color=StockCharts.COLORS['primary'], width=2)
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['macd_signal'],
                    name="Signal",
                    line=dict(color=StockCharts.COLORS['secondary'], width=2)
                ),
                row=2, col=1
            )
            
            if 'macd_histogram' in df.columns:
                colors = [StockCharts.COLORS['success'] if val >= 0 else StockCharts.COLORS['danger'] 
                         for val in df['macd_histogram']]
                fig.add_trace(
                    go.Bar(
                        x=df.index,
                        y=df['macd_histogram'],
                        name="Histogram",
                        marker_color=colors,
                        opacity=0.6
                    ),
                    row=2, col=1
                )
        
        # Bollinger Band Position
        if 'bb_percent' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df['bb_percent'],
                    name="BB %",
                    line=dict(color=StockCharts.COLORS['info'], width=2)
                ),
                row=3, col=1
            )
            
            for level in [1, -1]:
                fig.add_hline(
                    y=level,
                    line_dash="dash",
                    line_color=StockCharts.COLORS['warning'],
                    opacity=0.5,
                    row=3, col=1
                )
        
        # Professional indicators layout
        fig.update_layout(
            title=dict(
                text=f"{symbol} Technical Indicators",
                x=0.02,
                font=dict(size=22, color=StockCharts.COLORS['text'], weight='bold')
            ),
            height=height,
            plot_bgcolor=StockCharts.COLORS['background'],
            paper_bgcolor=StockCharts.COLORS['background'],
            font=dict(color=StockCharts.COLORS['text'], size=11),
            showlegend=False,
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        fig.update_xaxes(
            gridcolor=StockCharts.COLORS['grid'],
            gridwidth=0.5,
            showgrid=True
        )
        fig.update_yaxes(
            gridcolor=StockCharts.COLORS['grid'],
            gridwidth=0.5,
            showgrid=True
        )
        
        return fig
    
    @staticmethod
    def create_signal_gauge(
        signal_data: Dict,
        symbol: str = "Stock"
    ) -> go.Figure:
        """
        Create gauge chart for trading signals
        """
        
        signal = signal_data.get('signal', 'HOLD')
        confidence = signal_data.get('confidence', 'LOW')
        predicted_return = signal_data.get('predicted_return', 0)
        
        # Fix gauge calculation
        signal_value = {'SELL': -50, 'HOLD': 0, 'BUY': 50}.get(signal, 0)
        confidence_multiplier = {'LOW': 0.5, 'MEDIUM': 0.75, 'HIGH': 1.0}.get(confidence, 0.5)
        gauge_value = signal_value * confidence_multiplier
        
        # Color
        if signal == 'BUY':
            color = StockCharts.COLORS['success']
        elif signal == 'SELL':
            color = StockCharts.COLORS['danger']
        else:
            color = StockCharts.COLORS['info']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={'font': {'size': 16}},
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{symbol}", 'font': {'size': 18, 'color': StockCharts.COLORS['text'], 'weight': 'bold'}},
            gauge={
                'axis': {'range': [-50, 50], 'tickwidth': 1},
                'bar': {'color': color, 'thickness': 0.8},
                'steps': [
                    {'range': [-50, -15], 'color': 'rgba(255, 69, 58, 0.1)'},
                    {'range': [-15, 15], 'color': 'rgba(142, 142, 147, 0.1)'},
                    {'range': [15, 50], 'color': 'rgba(48, 209, 88, 0.1)'}
                ],
                'threshold': {
                    'line': {'color': color, 'width': 3},
                    'thickness': 0.75,
                    'value': gauge_value
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font=dict(color=StockCharts.COLORS['text'], size=12),
            plot_bgcolor=StockCharts.COLORS['background'],
            paper_bgcolor=StockCharts.COLORS['background'],
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        return fig
    
    @staticmethod
    def create_performance_summary(
        predictions: Dict,
        signals: Dict
    ) -> go.Figure:
        """
        Create performance summary chart
        """
        
        symbols = list(predictions.keys())
        returns = [signals[s].get('predicted_return', 0) for s in symbols if s in signals]
        volatilities = [predictions[s].get('volatility', 0.01) for s in symbols if s in predictions]
        signal_types = [signals[s].get('signal', 'HOLD') for s in symbols if s in signals]
        
        # Ensure all lists have same length
        min_len = min(len(returns), len(volatilities), len(signal_types))
        returns = returns[:min_len]
        volatilities = volatilities[:min_len]
        signal_types = signal_types[:min_len]
        symbols = symbols[:min_len]
        
        # Color mapping
        colors = []
        for signal in signal_types:
            if signal == 'BUY':
                colors.append(StockCharts.COLORS['success'])
            elif signal == 'SELL':
                colors.append(StockCharts.COLORS['danger'])
            else:
                colors.append(StockCharts.COLORS['info'])
        
        fig = go.Figure(data=go.Scatter(
            x=volatilities,
            y=returns,
            mode='markers+text',
            marker=dict(
                size=[abs(r)*500 + 15 for r in returns],
                color=colors,
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            text=symbols,
            textposition="top center"
        ))
        
        # Add reference lines
        fig.add_hline(y=0, line_dash="dash", line_color=StockCharts.COLORS['grid'])
        if volatilities:
            fig.add_vline(x=np.mean(volatilities), line_dash="dash", line_color=StockCharts.COLORS['grid'])
        
        fig.update_layout(
            title=dict(
                text="Risk vs Return Analysis",
                font=dict(size=22, color=StockCharts.COLORS['text'], weight='bold')
            ),
            xaxis_title="Risk (Volatility)",
            yaxis_title="Expected Return (%)",
            height=450,
            plot_bgcolor=StockCharts.COLORS['background'],
            paper_bgcolor=StockCharts.COLORS['background'],
            font=dict(color=StockCharts.COLORS['text'], size=12),
            margin=dict(l=60, r=40, t=60, b=50)
        )
        
        fig.update_xaxes(
            gridcolor=StockCharts.COLORS['grid'],
            gridwidth=0.5,
            showgrid=True
        )
        fig.update_yaxes(
            gridcolor=StockCharts.COLORS['grid'],
            gridwidth=0.5,
            showgrid=True
        )
        
        return fig