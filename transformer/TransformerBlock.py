from tensorflow.keras import layers
from tensorflow import keras
import tensorflow as tf

def causal_attention_mask(batch_size, n_dest, n_src, dtype):
    """
    Creates a causal attention mask to ensure that each token in a sequence only attends to previous and current tokens,
    but not to future tokens. This is crucial for autoregressive models where each token should not be influenced
    by tokens that come after it in the sequence.

    The mask is designed to be applied to the attention weights in self-attention mechanisms, such as those used
    in Transformer models. It prevents information from flowing from future tokens to the current token, ensuring
    that predictions for each token depend only on tokens that precede it.

    Parameters:
    - batch_size (int): The number of sequences in the batch.
    - n_dest (int): The length of the destination sequence (number of tokens in the sequence being processed).
    - n_src (int): The length of the source sequence (typically equal to n_dest in self-attention).
    - dtype (tf.DType): The data type for the mask tensor (e.g., tf.float32, tf.int32).

    Returns:
    - tf.Tensor: A tensor of shape [batch_size, n_dest, n_src] where the upper triangle of the dot product matrix
      is masked out with zeros, and the lower triangle (including the diagonal) is filled with ones. This tensor
      can be used to mask the attention weights in a self-attention mechanism, ensuring that each token attends only
      to earlier tokens and itself, but not to future tokens.

    Example:
    >>> causal_mask = causal_attention_mask(2, 4, 4, tf.float32)
    >>> print(causal_mask)
    <tf.Tensor: shape=(2, 4, 4), dtype=float32, numpy=
    array([[[1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 1.]],

           [[1., 0., 0., 0.],
            [1., 1., 0., 0.],
            [1., 1., 1., 0.],
            [1., 1., 1., 1.]]], dtype=float32)>
    """
    i = tf.range(n_dest)[:, None]
    j = tf.range(n_src)
    m = i >= j - n_src + n_dest
    mask = tf.cast(m, dtype)
    mask = tf.reshape(mask, [1, n_dest, n_src])
    mult = tf.concat(
        [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)], 0
    )
    return tf.tile(mask, mult)

class TransformerBlock(layers.Layer):
    """
    A single block of the Transformer model architecture. This block combines multi-head self-attention
    and feed-forward neural networks to process input sequences.

    The TransformerBlock is designed to capture complex dependencies in sequential data by using self-attention
    mechanisms. It also includes feed-forward layers to further process the attention outputs, along with normalization
    and dropout layers to stabilize training and prevent overfitting.

    Attributes:
    - embed_dim (int): The dimension of the embedding space.
    - num_heads (int): The number of attention heads in the multi-head attention mechanism.
    - ff_dim (int): The dimension of the feed-forward network hidden layer.
    - rate (float): The dropout rate applied to the attention and feed-forward layers (default is 0.1).

    Methods:
    - call(inputs): Executes the forward pass of the Transformer block. It applies the multi-head attention, adds
      residual connections, normalizes the outputs, and processes them through a feed-forward network.

    Parameters:
    - inputs (tf.Tensor): Input tensor with shape (batch_size, seq_len, embed_dim). Represents the sequence of embeddings.

    Returns:
    - tf.Tensor: Output tensor with shape (batch_size, seq_len, embed_dim). The processed sequence after attention,
      feed-forward operations, and normalization.

    Example:
    >>> transformer_block = TransformerBlock(embed_dim=64, num_heads=4, ff_dim=128)
    >>> inputs = tf.random.uniform((32, 10, 64))  # Example input tensor with batch_size=32, seq_len=10, embed_dim=64
    >>> output = transformer_block(inputs)
    >>> print(output.shape)
    (32, 10, 64)
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super().__init__(**kwargs)  # Initializes the parent class (layers.Layer).
        # MultiHeadAttention layer to capture relationships between different positions in the sequence.
        self.att = layers.MultiHeadAttention(num_heads, embed_dim)

        # Feed-forward network with a ReLU activation function followed by a linear layer.
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),  # Dense layer with ReLU activation.
                layers.Dense(embed_dim),  # Dense layer to project back to the embedding dimension.
            ]
        )

        # Layer normalization applied before and after the residual connection.
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)  # First layer normalization.

        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)  # Second layer normalization.

        # Dropout layers to prevent overfitting by randomly setting a fraction of input units to zero.
        self.dropout1 = layers.Dropout(rate)  # Dropout after the attention layer.

        self.dropout2 = layers.Dropout(rate)  # Dropout after the feed-forward network.

    def call(self, inputs):
        """
        Defines the forward pass of the Transformer block.

        Arguments:
        inputs -- The input tensor to the Transformer block.

        Returns:
        The output tensor of the Transformer block after applying attention, dropout, and feed-forward network.
        """
        input_shape = tf.shape(inputs)  # Get the shape of the input tensor.
        batch_size = input_shape[0]  # Number of sequences in the batch.
        seq_len = input_shape[1]  # Length of each sequence.

        # Create a causal attention mask to prevent attending to future tokens.
        causal_mask = causal_attention_mask(batch_size, seq_len, seq_len, tf.bool)

        # Apply multi-head attention with the causal mask.
        attention_output = self.att(inputs, inputs, attention_mask=causal_mask)

        # Apply dropout to the attention output.
        attention_output = self.dropout1(attention_output)

        # Add the input (residual connection) to the attention output and normalize.
        out1 = self.layernorm1(inputs + attention_output)

        # Apply the feed-forward network to the normalized output.
        ffn_output = self.ffn(out1)

        # Apply dropout to the feed-forward network output.
        ffn_output = self.dropout2(ffn_output)

        # Add the normalized output of the feed-forward network to the residual connection and normalize.
        return self.layernorm2(out1 + ffn_output)