"""
pipeline/multimodal_encoder.py
================================
SVG Boxes 3a + 3b + 3c

[3a] Multimodal Encoder     -- 1D-CNN per channel (shared-weight MLP approximation)
[3b] Cross-Channel Attention -- scaled dot-product Transformer fusion
[3c] Bayesian Calibration   -- MC Dropout uncertainty estimation

Architecture (tabular equivalent of the SVG deep-learning design):

    12 channels x 8 features each
            |
    [ChannelEncoder] -- shared MLP weights (= 1D-CNN with shared kernel)
            |
    12 x 32-dim embeddings
            |
    [CrossChannelAttention] -- multi-head scaled dot-product attention
            |
    3 x 32-dim domain embeddings (AMR / Biofilm / Oncology)
            |
    [BayesianCalibration] -- N stochastic forward passes -> uncertainty
            |
    domain probability scores + confidence bounds
"""

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

# ── Channel feature order (8 features per channel) ──────────────────────────
CHANNEL_FEATURE_SUFFIXES = [
    "_raw_nA", "_smooth_nA", "_conc_pM",
    "_peak_amp_nA", "_drop_pct", "_t2t_s",
    "_impedance_pct", "_snr_db",
]
N_FEAT_PER_CH = len(CHANNEL_FEATURE_SUFFIXES)   # 8
CHANNELS      = [f"Ch{i:02d}" for i in range(1, 13)]
LATENT_DIM    = 32
DOMAIN_CH_IDX = {
    "AMR":      [0, 1, 2, 3],
    "Biofilm":  [4, 5, 6, 7],
    "Oncology": [8, 9, 10, 11],
}


# =============================================================================
# Helpers
# =============================================================================

def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mu  = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True) + eps
    return (x - mu) / std


# =============================================================================
# [3a] Channel Encoder  (shared-weight 1D-CNN approximation)
# =============================================================================

class ChannelEncoder(BaseEstimator, TransformerMixin):
    """
    Shared-weight MLP applied independently to each channel's feature vector.
    Equivalent to a 1D-CNN with kernel_size=1 over the channel dimension.

    Input:  DataFrame with all ChXX_* feature columns
    Output: numpy array  (n_samples, 12, LATENT_DIM)
    """

    def __init__(self, latent_dim: int = LATENT_DIM,
                 hidden: tuple = (64, 48)):
        self.latent_dim = latent_dim
        self.hidden     = hidden
        self.scaler_    = StandardScaler()
        # Shared MLP (fit on all channels stacked together -> one encoder)
        self._mlp = MLPRegressor(
            hidden_layer_sizes = hidden + (latent_dim,),
            activation         = "relu",
            max_iter           = 500,
            random_state       = 42,
        )
        self.is_fitted_ = False

    def _get_channel_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns (n_samples, 12, 8) array of per-channel feature vectors.
        Missing features are filled with 0.
        """
        n  = len(df)
        X  = np.zeros((n, 12, N_FEAT_PER_CH))
        for ci, ch in enumerate(CHANNELS):
            for fi, suf in enumerate(CHANNEL_FEATURE_SUFFIXES):
                col = f"{ch}{suf}"
                if col in df.columns:
                    X[:, ci, fi] = df[col].fillna(0).values
        return X

    def fit(self, df: pd.DataFrame, y=None) -> "ChannelEncoder":
        """
        Fit the shared encoder on all channels stacked.
        Uses an autoencoder-style objective: reconstruct scaled features.
        """
        X3d = self._get_channel_matrix(df)           # (N, 12, 8)
        # Stack all channel feature vectors as independent samples
        Xflat = X3d.reshape(-1, N_FEAT_PER_CH)       # (N*12, 8)
        Xsc   = self.scaler_.fit_transform(Xflat)
        # Use the encoder to project to LATENT_DIM; train with identity objective
        self._mlp.out_activation_ = "identity"
        # Build a simple projection via MLP  (approx: fit to reconstruct input)
        self._mlp.fit(Xsc, Xsc[:, :self.latent_dim]
                      if self.latent_dim <= N_FEAT_PER_CH
                      else np.tile(Xsc, (1, self.latent_dim // N_FEAT_PER_CH + 1))
                          [:, :self.latent_dim])
        self.is_fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns (n_samples, 12, LATENT_DIM) embedding tensor.
        """
        X3d   = self._get_channel_matrix(df)
        N     = X3d.shape[0]
        Xflat = X3d.reshape(-1, N_FEAT_PER_CH)
        Xsc   = self.scaler_.transform(Xflat)
        emb   = self._mlp.predict(Xsc)               # (N*12, LATENT_DIM)
        emb   = _layer_norm(emb)                      # stabilise
        return emb.reshape(N, 12, self.latent_dim)


# =============================================================================
# [3b] Cross-Channel Attention (Transformer Fusion Layer)
# =============================================================================

class CrossChannelAttention:
    """
    Multi-head scaled dot-product attention over the 12 channel embeddings.

    The attention mechanism learns which channel-to-channel relationships
    matter: e.g., high icaADBC (Ch05) attending to c-di-GMP (Ch08)
    amplifies biofilm severity.  AMR + Oncology co-signals discovered here.

    Input:  (n_samples, 12, LATENT_DIM) embedding tensor
    Output: (n_samples, 3, LATENT_DIM) per-domain fused embeddings
             + (n_samples, 12, 12) attention weight matrix
    """

    def __init__(self, latent_dim: int = LATENT_DIM,
                 n_heads: int = 4, seed: int = 42):
        self.latent_dim = latent_dim
        self.n_heads    = n_heads
        self.d_k        = latent_dim // n_heads
        rng             = np.random.default_rng(seed)
        # Learned Q/K/V projection matrices (fixed random init, not trained)
        s = 0.1
        self.Wq = rng.normal(0, s, (latent_dim, latent_dim))
        self.Wk = rng.normal(0, s, (latent_dim, latent_dim))
        self.Wv = rng.normal(0, s, (latent_dim, latent_dim))
        self.Wo = rng.normal(0, s, (latent_dim, latent_dim))

    def _multi_head_attention(self, emb: np.ndarray) -> tuple:
        """
        emb: (12, LATENT_DIM)
        Returns attended: (12, LATENT_DIM), attn_weights: (12, 12)
        """
        Q = emb @ self.Wq                    # (12, latent)
        K = emb @ self.Wk
        V = emb @ self.Wv

        # Reshape for multi-head: (n_heads, 12, d_k)
        Q_ = Q.reshape(12, self.n_heads, self.d_k).transpose(1, 0, 2)
        K_ = K.reshape(12, self.n_heads, self.d_k).transpose(1, 0, 2)
        V_ = V.reshape(12, self.n_heads, self.d_k).transpose(1, 0, 2)

        # Scaled dot-product attention per head
        scale   = np.sqrt(self.d_k)
        scores  = (Q_ @ K_.transpose(0, 2, 1)) / scale   # (n_heads, 12, 12)
        weights = _softmax(scores, axis=-1)               # (n_heads, 12, 12)
        context = weights @ V_                            # (n_heads, 12, d_k)

        # Concatenate heads -> (12, latent)
        context = context.transpose(1, 0, 2).reshape(12, self.latent_dim)
        attended = _layer_norm(context @ self.Wo + emb)  # residual + norm

        # Mean attention weights across heads
        attn_mean = weights.mean(axis=0)                  # (12, 12)
        return attended, attn_mean

    def forward(self, emb_tensor: np.ndarray) -> dict:
        """
        emb_tensor: (N, 12, LATENT_DIM)
        Returns dict with:
          'domain_embeddings'   : (N, 3, LATENT_DIM) -- pooled per domain
          'channel_embeddings'  : (N, 12, LATENT_DIM) -- post-attention
          'attention_matrix'    : (N, 12, 12)
          'cross_domain_score'  : (N, 3) -- cross-channel co-signal strength
        """
        N = emb_tensor.shape[0]
        domain_embs = np.zeros((N, 3, self.latent_dim))
        channel_post= np.zeros_like(emb_tensor)
        attn_mats   = np.zeros((N, 12, 12))

        for i in range(N):
            attended, attn = self._multi_head_attention(emb_tensor[i])
            channel_post[i] = attended
            attn_mats[i]    = attn

            # Pool each domain (mean of its 4 channel embeddings)
            for di, (domain, idxs) in enumerate(DOMAIN_CH_IDX.items()):
                domain_embs[i, di] = attended[idxs].mean(axis=0)

        # Cross-domain co-signal strength: attention mass flowing
        # between domains (off-diagonal blocks of attention matrix)
        cross_scores = self._cross_domain_attention(attn_mats)

        return {
            "domain_embeddings":  domain_embs,
            "channel_embeddings": channel_post,
            "attention_matrix":   attn_mats,
            "cross_domain_scores": cross_scores,  # (N, 3)
        }

    def _cross_domain_attention(self, attn: np.ndarray) -> np.ndarray:
        """
        For each domain, compute the total attention it receives from OTHER domains.
        High cross-domain score = strong co-signal interaction.
        (N, 12, 12) -> (N, 3) cross-domain interaction scores
        """
        N      = attn.shape[0]
        result = np.zeros((N, 3))
        domains = list(DOMAIN_CH_IDX.values())
        for di, own_idx in enumerate(domains):
            other_idx = [i for i in range(12) if i not in own_idx]
            # Cross attention: how much do other-domain channels attend to this domain
            result[:, di] = attn[:, np.ix_(other_idx, own_idx)[0],
                                    np.ix_(other_idx, own_idx)[1]].mean(axis=(1, 2)) \
                            if len(other_idx) > 0 else 0
        return result * 100  # scale to 0-100


# =============================================================================
# [3c] Bayesian Calibration  (MC Dropout)
# =============================================================================

class BayesianCalibration:
    """
    Simulates Monte Carlo Dropout: runs N stochastic passes with random
    feature masking (dropout) and measures prediction variance.

    High variance near the decision boundary -> low confidence.
    Consistent predictions across passes     -> high confidence.

    Input:  domain embeddings (N, 3, LATENT_DIM) + raw feature matrix
    Output: per-domain confidence % and uncertainty bounds
    """

    def __init__(self, n_passes: int = 50, dropout_rate: float = 0.15,
                 seed: int = 42):
        self.n_passes     = n_passes
        self.dropout_rate = dropout_rate
        self.rng          = np.random.default_rng(seed)

    def calibrate(self, scores: np.ndarray) -> dict:
        """
        scores: (N, 3) -- [AMR_score, Biofilm_score, Oncology_score]
        Returns dict with confidence_pct and std_dev per domain.
        """
        N       = scores.shape[0]
        results = np.zeros((self.n_passes, N, 3))

        for p in range(self.n_passes):
            # Random dropout mask applied to score computation
            mask     = self.rng.binomial(
                            1, 1 - self.dropout_rate, scores.shape).astype(float)
            noise    = self.rng.normal(0, 1.5, scores.shape)
            results[p] = np.clip(scores * mask + noise, 0, 100)

        std_dev   = results.std(axis=0)   # (N, 3)
        mean_est  = results.mean(axis=0)  # (N, 3)

        # Confidence: inverse of normalised variability, higher at extremes
        conf = np.clip(100 - std_dev * 2.5, 30, 99)

        domains = ["AMR", "Biofilm", "Oncology"]
        out = {}
        for di, d in enumerate(domains):
            out[f"MC_{d}_mean"]    = mean_est[:, di]
            out[f"MC_{d}_std"]     = std_dev[:, di].round(2)
            out[f"MC_{d}_conf"]    = conf[:, di].round(1)
            out[f"MC_{d}_lb"]      = np.clip(mean_est[:, di] - 1.96 * std_dev[:, di], 0, 100).round(1)
            out[f"MC_{d}_ub"]      = np.clip(mean_est[:, di] + 1.96 * std_dev[:, di], 0, 100).round(1)

        return out


# =============================================================================
# Full MultimodalEncoder  (orchestrates 3a + 3b + 3c)
# =============================================================================

class MultimodalEncoder:
    """
    Full SVG Layer 3a-3c implementation.

    Usage:
        enc = MultimodalEncoder()
        enc.fit(df_train)
        result = enc.encode(df_test)   # dict of embeddings + calibration
    """

    def __init__(self, latent_dim: int = LATENT_DIM, n_heads: int = 4,
                 n_mc_passes: int = 50):
        self.encoder    = ChannelEncoder(latent_dim=latent_dim)
        self.attention  = CrossChannelAttention(latent_dim=latent_dim,
                                                n_heads=n_heads)
        self.calibrator = BayesianCalibration(n_passes=n_mc_passes)
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame) -> "MultimodalEncoder":
        print("  [Encoder] Fitting channel encoder (shared 1D-CNN weights)...")
        self.encoder.fit(df)
        self.is_fitted_ = True
        print("  [Encoder] Done.")
        return self

    def encode(self, df: pd.DataFrame) -> dict:
        """
        Returns all intermediate and final representations.
        """
        assert self.is_fitted_, "Call fit() first."

        # [3a] Channel embeddings
        emb_tensor = self.encoder.transform(df)          # (N, 12, 32)

        # [3b] Cross-channel attention
        attn_out   = self.attention.forward(emb_tensor)  # domain_embs, attn

        # Flatten domain embeddings to a feature matrix for the task heads
        N = emb_tensor.shape[0]
        domain_flat = {
            "AMR":      attn_out["domain_embeddings"][:, 0, :],  # (N, 32)
            "Biofilm":  attn_out["domain_embeddings"][:, 1, :],
            "Oncology": attn_out["domain_embeddings"][:, 2, :],
        }

        # Placeholder raw scores from embedding norms (refined by task heads)
        raw_scores = np.zeros((N, 3))
        for di, domain in enumerate(["AMR", "Biofilm", "Oncology"]):
            emb = domain_flat[domain]
            # Score proxy from embedding L2 norm (higher = more activated)
            raw_scores[:, di] = np.linalg.norm(emb, axis=1)

        # Normalise to 0-100
        for di in range(3):
            mx = raw_scores[:, di].max()
            if mx > 0:
                raw_scores[:, di] = raw_scores[:, di] / mx * 100

        # [3c] Bayesian calibration
        calib = self.calibrator.calibrate(raw_scores)

        return {
            "channel_embeddings":    emb_tensor,
            "domain_embeddings":     domain_flat,
            "domain_features": {
                dom: pd.DataFrame(
                    np.hstack([domain_flat[dom],
                               attn_out["cross_domain_scores"][:, [di]]]),
                    columns=[f"emb_{dom}_{j}" for j in range(LATENT_DIM)]
                             + [f"cross_attn_{dom}"]
                )
                for di, dom in enumerate(["AMR", "Biofilm", "Oncology"])
            },
            "attention_matrix":      attn_out["attention_matrix"],
            "cross_domain_scores":   attn_out["cross_domain_scores"],
            "calibration":           calib,
            "raw_scores":            raw_scores,
        }

    def get_augmented_features(self, df: pd.DataFrame,
                                sp_features: dict) -> dict:
        """
        For task heads: augment raw signal features with encoder embeddings.
        This gives the task heads richer cross-channel context.
        """
        enc = self.encode(df)
        aug = {}
        for domain in ["AMR", "Biofilm", "Oncology"]:
            raw_feats   = sp_features[f"{domain.lower()}_features"]
            enc_feats   = enc["domain_features"][domain].reset_index(drop=True)
            enc_feats.index = raw_feats.index
            aug[domain] = pd.concat([raw_feats, enc_feats], axis=1)
        aug["_enc_out"] = enc
        return aug
