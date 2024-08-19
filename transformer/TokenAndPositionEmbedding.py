from tensorflow.keras.layers import Embedding, Layer
import tensorflow as tf

class TokenAndPositionEmbedding(Layer):
    """
    A custom layer that combines token embeddings and positional embeddings for sequences.
    This layer is designed to convert input tokens into dense vectors and add positional information
    to each token embedding to capture the order of tokens in a sequence.

    The TokenAndPositionEmbedding layer is crucial for models that process sequential data, such as
    natural language processing models, where understanding the position of each token in the sequence
    is essential for interpreting the context and meaning.

    Attributes:
    - maxlen (int): The maximum length of the input sequences. This determines the size of the positional
      embeddings.
    - vocab_size (int): The size of the vocabulary, which determines the number of possible tokens.
    - embed_dim (int): The dimensionality of the embedding space. Each token and position is mapped to a vector of
      this size.

    Methods:
    - call(x): Applies the token and positional embeddings to the input sequences. It generates embeddings for each
      token and adds positional embeddings to these token embeddings to encode the order of tokens in the sequence.

    Parameters:
    - x (tf.Tensor): Input tensor of shape (batch_size, sequence_length), where each value represents a token index
      in the input sequences.

    Returns:
    - tf.Tensor: Output tensor of shape (batch_size, sequence_length, embed_dim), where each token index in the input
      sequences has been converted into an embedding vector, with positional information added to it.

    Example:
    >>> embedding_layer = TokenAndPositionEmbedding(maxlen=100, vocab_size=5000, embed_dim=64)
    >>> input_seq = tf.constant([[1, 5, 9], [2, 6, 3]])
    >>> output = embedding_layer(input_seq)
    >>> print(output.shape)
    (2, 3, 64)
    """
    def __init__(self, maxlen, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)  # Initializes the parent class (layers.Layer).

        # Embedding layer for tokens, maps token indices to dense vectors of size `embed_dim`.
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim)

        # Embedding layer for positional encodings, maps position indices to dense vectors of size `embed_dim`.
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        """
        Applies token and positional embeddings to the input tensor.

        Arguments:
        x -- The input tensor containing token indices.

        Returns:
        The tensor after adding token and positional embeddings.
        """
        maxlen = tf.shape(x)[-1]  # Get the length of the sequences from the input tensor shape.

        # Generate position indices from 0 to maxlen - 1.
        positions = tf.range(start=0, limit=maxlen, delta=1)

        # Apply the positional embedding layer to position indices.
        positions = self.pos_emb(positions)

        # Apply the token embedding layer to the input tensor.
        x = self.token_emb(x)

        # Add the token embeddings and positional embeddings.
        return x + positions