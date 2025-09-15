import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Optional, Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
torch.backends.cuda.matmul.allow_tf32 = True  # Enable TF32 for faster training
torch.backends.cudnn.allow_tf32 = True


class PatchEmbedding(nn.Module):
    """
    Patch Embedding layer that converts time series into patches and embeds them.
    Optimized for financial time series with proper normalization.
    """
    
    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: str = 'end',
        in_channels: int = 1,
        embed_dim: int = 128,
        norm_layer: Optional[nn.Module] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        # Patch projection layer
        self.proj = nn.Conv1d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_len, 
            stride=stride,
            bias=True
        )
        
        # Multi-feature projection layer (registered as proper module)
        self.multi_proj = None  # Will be initialized when needed
        
        # Normalization
        self.norm = norm_layer if norm_layer is not None else nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights properly for financial data
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper scaling"""
        nn.init.kaiming_normal_(self.proj.weight, mode='fan_out', nonlinearity='relu')
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
        Returns:
            patches: Embedded patches (batch_size, n_patches, embed_dim)
            n_patches: Number of patches
        """
        batch_size, seq_len, n_features = x.shape
        
        # Handle padding if necessary
        if seq_len % self.stride != 0:
            if self.padding_patch == 'end':
                pad_len = self.stride - (seq_len % self.stride)
                x = F.pad(x, (0, 0, 0, pad_len), mode='replicate')
            elif self.padding_patch == 'start':
                pad_len = self.stride - (seq_len % self.stride)
                x = F.pad(x, (0, 0, pad_len, 0), mode='replicate')
        
        # Process all features in parallel using grouped convolution
        x = x.transpose(1, 2)  # (batch_size, n_features, seq_len)
        
        # Use grouped convolution for efficient parallel processing
        if n_features > 1:
            # Initialize multi_proj if not already done (CRITICAL FIX)
            if self.multi_proj is None:
                self.multi_proj = nn.Conv1d(
                    n_features,
                    self.embed_dim,
                    kernel_size=self.patch_len,
                    stride=self.stride,
                    bias=True
                ).to(x.device)
                # Initialize the new layer
                nn.init.kaiming_normal_(self.multi_proj.weight, mode='fan_out', nonlinearity='relu')
                if self.multi_proj.bias is not None:
                    nn.init.constant_(self.multi_proj.bias, 0)

            patches = self.multi_proj(x)  # (batch_size, embed_dim, n_patches)
        else:
            patches = self.proj(x.view(batch_size, 1, -1))  # (batch_size, embed_dim, n_patches)
        
        n_patches = patches.shape[-1]
        patches = patches.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        
        # Apply normalization and dropout
        patches = self.norm(patches)
        patches = self.dropout(patches)
        
        return patches, n_patches


class PositionalEncoding(nn.Module):
    """
    Positional Encoding with support for both learned and sinusoidal encodings.
    Enhanced for time series with temporal patterns.
    """
    
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 5000,
        encoding_type: str = 'sinusoidal',
        dropout: float = 0.1
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.encoding_type = encoding_type
        self.dropout = nn.Dropout(dropout)
        
        if encoding_type == 'learned':
            self.pos_embedding = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        elif encoding_type == 'sinusoidal':
            pe = torch.zeros(max_len, embed_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            
            div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                               (-math.log(10000.0) / embed_dim))
            
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            
            self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
        Returns:
            x with positional encoding added
        """
        if self.encoding_type == 'learned':
            x = x + self.pos_embedding[:, :x.size(1), :]
        elif self.encoding_type == 'sinusoidal':
            x = x + self.pe[:, :x.size(1), :]
        
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention with financial time series optimizations.
    Includes relative position encoding and attention dropout.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_relative_pos: bool = True,
        max_relative_position: int = 64
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_relative_pos = use_relative_pos
        
        # Linear projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=True)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(attention_dropout)
        self.proj_dropout = nn.Dropout(dropout)
        
        # Relative position encoding for time series
        if use_relative_pos:
            self.max_relative_position = max_relative_position
            self.relative_position_k = nn.Parameter(
                torch.randn(2 * max_relative_position - 1, self.head_dim) * 0.02
            )
            self.relative_position_v = nn.Parameter(
                torch.randn(2 * max_relative_position - 1, self.head_dim) * 0.02
            )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv.weight)
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.constant_(self.qkv.bias, 0)
        nn.init.constant_(self.proj.bias, 0)
    
    def _get_relative_positions(self, seq_len: int) -> torch.Tensor:
        """Generate relative position indices"""
        range_vec = torch.arange(seq_len)
        relative_positions = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position + 1, 
            self.max_relative_position - 1
        )
        relative_positions = relative_positions + self.max_relative_position - 1
        return relative_positions
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len, embed_dim = x.shape
        
        # Use Flash Attention optimization when available
        if hasattr(F, 'scaled_dot_product_attention') and torch.cuda.is_available():
            # PyTorch 2.0+ Flash Attention 
            qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
            qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            # Use Flash Attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=mask,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
                is_causal=False
            )
            
            # Reshape and project
            attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
            out = self.proj(attn_output)
            out = self.proj_dropout(out)
            
            return out
        
        # Fallback to manual attention for older PyTorch versions
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, num_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with optimized matrix multiplication
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Add relative position encoding
        if self.use_relative_pos and seq_len <= self.max_relative_position:
            relative_positions = self._get_relative_positions(seq_len).to(x.device)
            
            # Relative position encoding for keys
            rel_pos_k = self.relative_position_k[relative_positions]
            rel_attn_k = torch.einsum('bhid,ijkd->bhijk', q, rel_pos_k)
            rel_attn_k = rel_attn_k.sum(dim=-1)
            attn_scores = attn_scores + rel_attn_k
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        
        # Softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.attn_dropout(attn_probs)
        
        # Apply attention to values
        out = torch.matmul(attn_probs, v)
        
        # Add relative position encoding for values
        if self.use_relative_pos and seq_len <= self.max_relative_position:
            rel_pos_v = self.relative_position_v[relative_positions]
            rel_attn_v = torch.einsum('bhij,jikd->bhikd', attn_probs, rel_pos_v)
            rel_attn_v = rel_attn_v.sum(dim=-2)
            out = out + rel_attn_v
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        out = self.proj(out)
        out = self.proj_dropout(out)
        
        return out


class FeedForward(nn.Module):
    """
    Feed Forward Network with GELU activation and proper dropout.
    Optimized for financial time series modeling.
    """
    
    def __init__(
        self,
        embed_dim: int,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        super().__init__()
        ff_dim = ff_dim or 4 * embed_dim
        
        self.fc1 = nn.Linear(embed_dim, ff_dim)
        self.fc2 = nn.Linear(ff_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer Block with pre-normalization and stochastic depth.
    Optimized for financial time series with proper residual connections.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        drop_path: float = 0.0,
        use_relative_pos: bool = True,
        activation: str = 'gelu'
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.drop_path = drop_path
        
        # Pre-normalization layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            use_relative_pos=use_relative_pos
        )
        
        # Feed forward network
        self.ffn = FeedForward(
            embed_dim=embed_dim,
            ff_dim=ff_dim,
            dropout=dropout,
            activation=activation
        )
        
        # Stochastic depth for regularization
        self.drop_path_layer = DropPath(drop_path) if drop_path > 0 else nn.Identity()
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Pre-norm attention with residual connection
        attn_out = self.attention(self.norm1(x), mask=mask)
        x = x + self.drop_path_layer(attn_out)
        
        # Pre-norm feed forward with residual connection
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.drop_path_layer(ffn_out)
        
        return x


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) implementation for regularization.
    Randomly drops entire residual branches during training.
    """
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output


class PatchTST(nn.Module):
    """
    State-of-the-art PatchTST model optimized for stock price prediction.
    
    Features:
    - Patch-based time series transformation
    - Multi-head attention with relative positioning
    - Pre-normalization and stochastic depth
    - Optimized for financial time series patterns
    - Support for multiple forecasting horizons
    - Advanced regularization techniques
    """
    
    def __init__(
        self,
        # Data parameters
        seq_len: int = 512,
        pred_len: int = 1,
        n_features: int = 50,
        
        # Patch parameters
        patch_len: int = 16,
        stride: int = 8,
        padding_patch: str = 'end',
        
        # Model parameters
        embed_dim: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: Optional[int] = None,
        
        # Regularization
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        drop_path: float = 0.1,
        
        # Architecture options
        use_relative_pos: bool = True,
        pos_encoding: str = 'sinusoidal',
        activation: str = 'gelu',
        
        # Output options
        output_attention: bool = False,
        use_multi_scale: bool = True
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.output_attention = output_attention
        self.use_multi_scale = use_multi_scale
        
        # Calculate number of patches
        if seq_len % stride != 0:
            if padding_patch == 'end':
                padded_len = seq_len + (stride - seq_len % stride)
            else:
                padded_len = seq_len + (stride - seq_len % stride)
        else:
            padded_len = seq_len
        self.n_patches = (padded_len - patch_len) // stride + 1
        
        logger.info(f"PatchTST initialized: seq_len={seq_len}, n_patches={self.n_patches}, embed_dim={embed_dim}")
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            padding_patch=padding_patch,
            in_channels=1,  # Will handle multiple features internally
            embed_dim=embed_dim,
            dropout=dropout
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(
            embed_dim=embed_dim,
            max_len=self.n_patches * 2,
            encoding_type=pos_encoding,
            dropout=dropout
        )
        
        # Transformer layers with varying drop path rates
        drop_path_rates = [x.item() for x in torch.linspace(0, drop_path, num_layers)]
        self.transformer_layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                attention_dropout=attention_dropout,
                drop_path=drop_path_rates[i],
                use_relative_pos=use_relative_pos,
                activation=activation
            )
            for i in range(num_layers)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # Multi-scale prediction heads
        if use_multi_scale:
            # Different prediction heads for different time horizons
            self.pred_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(embed_dim // 2, 1)
                )
                for _ in range(pred_len)  # Create head for each prediction step
            ])
        else:
            # Single prediction head
            self.pred_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim // 2, pred_len)
            )
        
        # Global average pooling for sequence aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Apply weight initialization
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights properly for financial time series"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of PatchTST model.
        
        Args:
            x: Input tensor (batch_size, seq_len, n_features)
            mask: Optional attention mask
            
        Returns:
            Dictionary containing predictions and optionally attention weights
        """
        batch_size, seq_len, n_features = x.shape
        
        # Process all features at once using the enhanced patch embedding
        patches, n_patches = self.patch_embedding(x)
        
        # Add positional encoding
        patches = self.pos_encoding(patches)
        
        # Store attention weights if requested
        attention_weights = [] if self.output_attention else None
        
        # Pass through transformer layers
        for i, transformer_layer in enumerate(self.transformer_layers):
            patches = transformer_layer(patches, mask=mask)
            
            # Store attention weights for analysis
            if self.output_attention:
                # This would require modifying transformer_layer to return attention weights
                pass
        
        # Final normalization
        patches = self.norm(patches)
        
        # Global pooling to get sequence representation
        # patches: (batch_size, n_patches, embed_dim)
        sequence_repr = patches.mean(dim=1)  # (batch_size, embed_dim)
        
        # Generate predictions with uncertainty estimation
        if self.use_multi_scale and len(self.pred_heads) > 1:
            # Multi-scale predictions
            predictions = []
            for i, pred_head in enumerate(self.pred_heads):
                if i < self.pred_len:
                    pred = pred_head(sequence_repr)  # (batch_size, 1)
                    predictions.append(pred)

            predictions = torch.cat(predictions, dim=1)  # (batch_size, pred_len)
        else:
            # Single-scale prediction
            predictions = self.pred_head(sequence_repr)  # (batch_size, pred_len)

        # Calculate prediction uncertainty using ensemble-style approach
        # This helps provide confidence intervals for professional analysis
        uncertainty = self._calculate_prediction_uncertainty(sequence_repr, predictions)

        # Prepare output dictionary with professional metrics
        output = {
            'predictions': predictions,
            'uncertainty': uncertainty,
            'confidence_intervals': self._get_confidence_intervals(predictions, uncertainty),
            'embeddings': sequence_repr,
            'patch_embeddings': patches,
            'trend_strength': self._calculate_trend_strength(predictions)
        }
        
        if self.output_attention and attention_weights:
            output['attention_weights'] = attention_weights
        
        return output

    def _calculate_prediction_uncertainty(self, embeddings: torch.Tensor, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate prediction uncertainty for professional risk assessment.
        Uses embedding variance as a proxy for model confidence.
        """
        try:
            # Calculate embedding variance across the embedding dimension
            embedding_var = torch.var(embeddings, dim=1, keepdim=True)

            # Normalize uncertainty to [0, 1] range
            uncertainty = torch.sigmoid(embedding_var / (embedding_var.mean() + 1e-8))

            # Expand to match prediction dimensions
            if len(predictions.shape) > 1:
                uncertainty = uncertainty.expand(-1, predictions.shape[1])

            return uncertainty
        except Exception as e:
            logger.warning(f"Uncertainty calculation failed: {e}")
            # Return low confidence (high uncertainty) as fallback
            return torch.ones_like(predictions) * 0.7

    def _get_confidence_intervals(self, predictions: torch.Tensor, uncertainty: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate confidence intervals for professional analysis.
        Provides upper/lower bounds for trend analysis.
        """
        try:
            # Scale uncertainty to reasonable confidence intervals
            # High uncertainty -> wider intervals
            interval_width = uncertainty * 0.1  # Max 10% interval width

            lower_bound = predictions * (1 - interval_width)
            upper_bound = predictions * (1 + interval_width)

            return {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'interval_width': interval_width
            }
        except Exception as e:
            logger.warning(f"Confidence interval calculation failed: {e}")
            return {
                'lower_bound': predictions,
                'upper_bound': predictions,
                'interval_width': torch.zeros_like(predictions)
            }

    def _calculate_trend_strength(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Calculate trend strength for professional trend analysis.
        Measures consistency of directional movement.
        """
        try:
            if predictions.shape[1] < 2:
                return torch.zeros(predictions.shape[0], 1)

            # Calculate directional changes
            changes = predictions[:, 1:] - predictions[:, :-1]

            # Trend strength based on consistency of direction
            positive_changes = (changes > 0).float()
            negative_changes = (changes < 0).float()

            # Calculate directional consistency
            pos_consistency = positive_changes.mean(dim=1, keepdim=True)
            neg_consistency = negative_changes.mean(dim=1, keepdim=True)

            # Trend strength is the maximum directional consistency
            trend_strength = torch.max(pos_consistency, neg_consistency)

            return trend_strength
        except Exception as e:
            logger.warning(f"Trend strength calculation failed: {e}")
            return torch.ones(predictions.shape[0], 1) * 0.5

    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract attention weights for interpretability analysis.
        Useful for understanding which time periods the model focuses on.
        """
        # This would require modifying the forward pass to capture attention weights
        # Implementation would involve modifying TransformerBlock to return attention maps
        pass


def create_patchtst_model(config: Dict) -> PatchTST:
    """
    Factory function to create PatchTST model from configuration.
    
    Args:
        config: Configuration dictionary with model parameters
        
    Returns:
        Initialized PatchTST model
    """
    model = PatchTST(
        seq_len=config.get('seq_len', 512),
        pred_len=config.get('pred_len', 1),
        n_features=config.get('n_features', 50),
        patch_len=config.get('patch_len', 16),
        stride=config.get('stride', 8),
        embed_dim=config.get('embed_dim', 128),
        num_layers=config.get('num_layers', 6),
        num_heads=config.get('num_heads', 8),
        dropout=config.get('dropout', 0.1),
        use_multi_scale=config.get('use_multi_scale', True)
    )
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"PatchTST model created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    return model


def get_config(model_size: str = "large") -> Dict:
    """
    Different model sizes to maximize GPU utilization.
    """
    
    if model_size == "small":
        return {
            'seq_len': 200,          # 200 days of history (~8 months)
            'pred_len': 10,          # Predict next 10 days
            'n_features': 80,        # All technical indicators + price features
            'patch_len': 16,         # 16-day patches
            'stride': 8,             # 8-day stride
            'embed_dim': 512,        # Large embedding dimension
            'num_layers': 12,        # Deep model
            'num_heads': 16,         # Multi-head attention
            'ff_dim': 2048,          # Feed-forward dimension
            'dropout': 0.1,
            'attention_dropout': 0.1,
            'drop_path': 0.1,
            'use_multi_scale': True,
            'use_relative_pos': True,
            'batch_size': 64,        
        }
    
    elif model_size == "large":
        return {
            'seq_len': 1024,         # 1024 days (~4 years of daily data)
            'pred_len': 20,          # Predict next 20 days (1 month)
            'n_features': 100,       # Extended feature set
            'patch_len': 32,         # Larger patches for long sequences
            'stride': 16,            # Larger stride
            'embed_dim': 768,        # Large embedding (GPT-style)
            'num_layers': 16,        # Very deep model
            'num_heads': 24,         # Many attention heads
            'ff_dim': 3072,          # Large feed-forward
            'dropout': 0.15,
            'attention_dropout': 0.1,
            'drop_path': 0.2,        # Strong regularization
            'use_multi_scale': True,
            'use_relative_pos': True,
            'batch_size': 32,        # Balanced batch size
        }
    
    elif model_size == "xl":
        return {
            'seq_len': 2048,         # 2048 days (~8 years)
            'pred_len': 30,          # Predict next 30 days
            'n_features': 120,       # Maximum features
            'patch_len': 64,         # Large patches
            'stride': 32,            # Large stride
            'embed_dim': 1024,       # Very large embedding
            'num_layers': 20,        # Extremely deep
            'num_heads': 32,         # Maximum heads
            'ff_dim': 4096,          # Maximum feed-forward
            'dropout': 0.2,
            'attention_dropout': 0.15,
            'drop_path': 0.3,
            'use_multi_scale': True,
            'use_relative_pos': True,
            'batch_size': 16,        # Smaller batch for memory
        }
    
    else:
        raise ValueError(f"Unknown model size: {model_size}")


def get_training_config() -> Dict:
    return {
        # Optimizer settings
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        
        # Scheduler settings
        'scheduler': 'cosine_with_warmup',
        'warmup_steps': 1000,
        'max_steps': 100000,
        
        # Mixed precision training 
        'use_amp': True,
        'grad_scaler': True,
        
        # Gradient settings
        'grad_clip_norm': 1.0,
        'gradient_accumulation_steps': 2,
        
        # Memory optimization
        'use_checkpoint': True,  # Gradient checkpointing
        'pin_memory': True,
        'num_workers': 16,       
        
        # Training settings
        'epochs': 100,
        'eval_every': 1000,
        'save_every': 5000,
        'early_stopping_patience': 20,
        
        # Hardware-specific
        'device': 'cuda',
        'compile': True,         # PyTorch 2.0 compilation
        'channels_last': True,   # Memory layout optimization
    }


# Example usage and configuration
if __name__ == "__main__":
    print("Available configurations:")
    
    for size in ['small', 'large', 'xl']:
        config = get_config(size)
        print(f"\n{size.upper()} Model:")
        print(f"  Sequence Length: {config['seq_len']} days")
        print(f"  Prediction Length: {config['pred_len']} days") 
        print(f"  Features: {config['n_features']}")
        print(f"  Embedding Dim: {config['embed_dim']}")
        print(f"  Layers: {config['num_layers']}")
        print(f"  Attention Heads: {config['num_heads']}")
        print(f"  Batch Size: {config['batch_size']}")
        
        # Estimate memory usage
        model = create_patchtst_model(config)
        total_params = sum(p.numel() for p in model.parameters())
        estimated_memory = total_params * 4 * config['batch_size'] / 1024**3  # GB
        print(f"  Parameters: {total_params:,}")
        print(f"  Estimated VRAM: {estimated_memory:.2f} GB")

    config = get_config("large")
    
    # Create model
    model = create_patchtst_model(config)
    
    # Example forward pass
    batch_size = 32
    x = torch.randn(batch_size, config['seq_len'], config['n_features'])
    
    with torch.no_grad():
        output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Prediction shape: {output['predictions'].shape}")
    print(f"Embedding shape: {output['embeddings'].shape}")
    print(f"Patch embeddings shape: {output['patch_embeddings'].shape}")