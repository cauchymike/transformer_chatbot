import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# Setting up the device for GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Embeddings(nn.Module):
    """
    Implements word embeddings and adds their positional encodings.
    
    Args:
        vocab_size (int): The size of the vocabulary (number of unique tokens).
        d_model (int): The dimension of each embedding vector.
        max_len (int): The maximum sequence length for positional encodings.
    """
    def __init__(self, vocab_size: int, d_model: int, max_len: int = 50):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding(vocab_size, d_model)  # Embedding layer
        self.pe = self.create_positional_encoding(max_len, self.d_model)  # Positional encoding tensor
        self.dropout = nn.Dropout(0.1)

    def create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Creates positional encoding to capture the position of words in a sequence.

        Args:
            max_len (int): Maximum sequence length for positional encoding.
            d_model (int): Embedding dimension size.

        Returns:
            torch.Tensor: Positional encoding of shape (1, max_len, d_model).
        """
        pe = torch.zeros(max_len, d_model).to(device)  # Positional encoding matrix
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                # Sinusoidal encoding for even indices
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / d_model)))
                # Cosine encoding for odd indices
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model)))
        pe = pe.unsqueeze(0)  # Add batch size dimension
        return pe

    def forward(self, encoded_words: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the embedding layer.

        Args:
            encoded_words (torch.Tensor): Tensor containing the word indices of shape (batch_size, max_len).

        Returns:
            torch.Tensor: Word embeddings with positional encoding added, shape (batch_size, max_len, d_model).
        """
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model)  # Scale embeddings
        embedding += self.pe[:, :embedding.size(1)]  # Add positional encoding
        embedding = self.dropout(embedding)
        return embedding


class MultiHeadAttention(nn.Module):
    """
    Implements Multi-Head Self-Attention mechanism.
    
    Args:
        heads (int): Number of attention heads.
        d_model (int): Dimension of the model (embedding size).
    """
    def __init__(self, heads: int, d_model: int):
        super(MultiHeadAttention, self).__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"
        
        self.d_k = d_model // heads  # Dimension per head
        self.heads = heads
        self.dropout = nn.Dropout(0.1)
        
        # Linear layers to compute query, key, value vectors
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.concat = nn.Linear(d_model, d_model)  # Layer to combine multi-head outputs

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-head attention.
        
        Args:
            query (torch.Tensor): Query tensor of shape (batch_size, max_len, d_model).
            key (torch.Tensor): Key tensor of shape (batch_size, max_len, d_model).
            value (torch.Tensor): Value tensor of shape (batch_size, max_len, d_model).
            mask (torch.Tensor): Attention mask to prevent attending to certain positions.
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, max_len, d_model).
        """
        # Linear projections of query, key, and value
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        
        # Reshape into (batch_size, heads, max_len, d_k)
        query = query.view(query.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.heads, self.d_k).permute(0, 2, 1, 3)

        # Scaled dot-product attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)  # (batch_size, heads, max_len, max_len)
        scores = scores.masked_fill(mask == 0, -1e9)  # Apply mask (batch_size, heads, max_len, max_len)
        weights = F.softmax(scores, dim=-1)  # Normalize scores
        weights = self.dropout(weights)

        # Compute attention output
        context = torch.matmul(weights, value)  # (batch_size, heads, max_len, d_k)

        # Concatenate attention heads and project back to d_model
        context = context.permute(0, 2, 1, 3).contiguous().view(context.shape[0], -1, self.heads * self.d_k)
        interacted = self.concat(context)  # Final output after concatenation

        return interacted


class FeedForward(nn.Module):
    """
    Implements a Position-Wise Feed Forward Network.
    
    Args:
        d_model (int): Input/output dimension of the model.
        middle_dim (int): Dimension of the hidden layer (default: 2048).
    """
    def __init__(self, d_model: int, middle_dim: int = 2048):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, middle_dim)
        self.fc2 = nn.Linear(middle_dim, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the feedforward network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, max_len, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, max_len, d_model).
        """
        out = F.relu(self.fc1(x))  # First linear layer + ReLU
        out = self.fc2(self.dropout(out))  # Second linear layer + dropout
        return out


class EncoderLayer(nn.Module):
    """
    Single Encoder layer composed of Multi-Head Attention and FeedForward layers.
    
    Args:
        d_model (int): Dimension of the model.
        heads (int): Number of attention heads.
    """
    def __init__(self, d_model: int, heads: int):
        super(EncoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)  # Layer normalization
        self.self_multihead = MultiHeadAttention(heads, d_model)  # Self-attention layer
        self.feed_forward = FeedForward(d_model)  # Feed-forward layer
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the encoder layer.
        
        Args:
            embeddings (torch.Tensor): Input embeddings of shape (batch_size, max_len, d_model).
            mask (torch.Tensor): Attention mask of shape (batch_size, 1, 1, max_len).
        
        Returns:
            torch.Tensor: Encoded output of shape (batch_size, max_len, d_model).
        """
        # Self-attention with residual connection and layer normalization
        interacted = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, mask))
        interacted = self.layernorm(interacted + embeddings)

        # Feed-forward with residual connection and layer normalization
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        encoded = self.layernorm(feed_forward_out + interacted)
        
        return encoded


class DecoderLayer(nn.Module):
    """
    Single Decoder layer composed of Self-Attention, Encoder-Decoder Attention, and FeedForward layers.
    
    Args:
        d_model (int): Dimension of the model.
        heads (int): Number of attention heads.
    """
    def __init__(self, d_model: int, heads: int):
        super(DecoderLayer, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
        self.self_multihead = MultiHeadAttention(heads, d_model)  # Self-attention for the target sequence
        self.src_multihead = MultiHeadAttention(heads, d_model)   # Encoder-Decoder attention
        self.feed_forward = FeedForward(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, embeddings: torch.Tensor, encoded: torch.Tensor, src_mask: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the decoder layer.
        
        Args:
            embeddings (torch.Tensor): Target sequence embeddings of shape (batch_size, max_len, d_model).
            encoded (torch.Tensor): Encoder output of shape (batch_size, max_len, d_model).
            src_mask (torch.Tensor): Mask for the source sequence.
            target_mask (torch.Tensor): Mask for the target sequence.
        
        Returns:
            torch.Tensor: Decoded output of shape (batch_size, max_len, d_model).
        """
        # Self-attention (for the target sequence)
        query = self.dropout(self.self_multihead(embeddings, embeddings, embeddings, target_mask))
        query = self.layernorm(query + embeddings)

        # Encoder-Decoder attention (attending to the encoded input)
        interacted = self.dropout(self.src_multihead(query, encoded, encoded, src_mask))
        interacted = self.layernorm(interacted + query)

        # Feed-forward layer with residual connection
        feed_forward_out = self.dropout(self.feed_forward(interacted))
        decoded = self.layernorm(feed_forward_out + interacted)
        
        return decoded


class Transformer(nn.Module):
    """
    Full Transformer model combining both the Encoder and Decoder layers.
    
    Args:
        d_model (int): Dimension of the model.
        heads (int): Number of attention heads.
        num_layers (int): Number of Encoder and Decoder layers.
        word_map (dict): Vocabulary word map.
    """
    def __init__(self, d_model: int, heads: int, num_layers: int, word_map: dict):
        super(Transformer, self).__init__()
        
        self.d_model = d_model
        self.vocab_size = len(word_map)
        
        # Embedding layer for both encoder and decoder
        self.embed = Embeddings(self.vocab_size, d_model)
        
        # Stacking multiple encoder and decoder layers
        self.encoder = nn.ModuleList([EncoderLayer(d_model, heads) for _ in range(num_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model, heads) for _ in range(num_layers)])
        
        # Linear layer to project the final output to vocabulary size
        self.logit = nn.Linear(d_model, self.vocab_size)

    def encode(self, src_words: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Encoder pass for the source words.
        
        Args:
            src_words (torch.Tensor): Source sequence of word indices.
            src_mask (torch.Tensor): Mask for the source sequence.
        
        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, max_len, d_model).
        """
        src_embeddings = self.embed(src_words)
        for layer in self.encoder:
            src_embeddings = layer(src_embeddings, src_mask)
        return src_embeddings

    def decode(self, target_words: torch.Tensor, target_mask: torch.Tensor, src_embeddings: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Decoder pass for the target words.
        
        Args:
            target_words (torch.Tensor): Target sequence of word indices.
            target_mask (torch.Tensor): Mask for the target sequence.
            src_embeddings (torch.Tensor): Encoded source sequence from the encoder.
            src_mask (torch.Tensor): Mask for the source sequence.
        
        Returns:
            torch.Tensor: Decoded output of shape (batch_size, max_len, d_model).
        """
        tgt_embeddings = self.embed(target_words)
        for layer in self.decoder:
            tgt_embeddings = layer(tgt_embeddings, src_embeddings, src_mask, target_mask)
        return tgt_embeddings

    def forward(self, src_words: torch.Tensor, src_mask: torch.Tensor, target_words: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the full Transformer model.
        
        Args:
            src_words (torch.Tensor): Source sequence of word indices.
            src_mask (torch.Tensor): Mask for the source sequence.
            target_words (torch.Tensor): Target sequence of word indices.
            target_mask (torch.Tensor): Mask for the target sequence.
        
        Returns:
            torch.Tensor: Log softmax output of shape (batch_size, max_len, vocab_size).
        """
        encoded = self.encode(src_words, src_mask)
        decoded = self.decode(target_words, target_mask, encoded, src_mask)
        out = F.log_softmax(self.logit(decoded), dim=2)  # Apply log softmax over the vocabulary
        return out
