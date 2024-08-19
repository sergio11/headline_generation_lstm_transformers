import numpy as np
import tensorflow as tf
from tensorflow import keras
import TokenAndPositionEmbedding
import TransformerBlock

class TransformersHeadlineGenerator:
    def __init__(self, model_path='trained_model.model.h5', weights_path='trained_model.weights.h5', vocab='vocabulary.txt', maxlen=80, top_k=10):
        """
        Initializes the TransformersHeadlineGenerator class.
        
        Arguments:
        - model_path (str): Path to the file containing the saved model.
        - weights_path (str): Path to the file containing the saved model weights.
        - vocab (list): List of vocabulary used in the model.
        - maxlen (int): Maximum length of input sequences.
        - top_k (int): Number of most probable predictions to consider during generation.
        """
        self.model_path = model_path
        self.weights_path = weights_path
        self.maxlen = maxlen
        self.top_k = top_k

        # Load the saved model with custom objects
        self.model = keras.models.load_model(model_path, custom_objects={
            'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
            'TransformerBlock': TransformerBlock
        })
        self.model.load_weights(weights_path)

        # Load and set vocab
        self.vocab = self._load_vocab(vocab)
        print(self.vocab[:30])
        self.word_to_index = {word: index for index, word in enumerate(self.vocab)}
        self.index_to_word = {index: word for word, index in self.word_to_index.items()}

    def _load_vocab(self, vocab_file):
        """
        Load vocabulary from a file.
        """
        with open(vocab_file, 'r') as file:
            vocab = file.read().splitlines()
        return vocab

    def _sample_from(self, logits):
        """
        Samples a token from the model's predictions using a soft probability distribution.
        """
        logits, indices = tf.math.top_k(logits, k=self.top_k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def _detokenize(self, token):
        """
        Converts a token index to its corresponding word.
        """
        return self.index_to_word[token]

    def generate_text_from_prompt(self, prompt, max_tokens=8):
        """
        Generates text based on an input prompt.
        
        Arguments:
        - prompt (str): The initial text from which the sequence will be generated.
        - max_tokens (int): Maximum number of tokens to generate.
        
        Returns:
        - str: The generated text.
        """

        start_tokens = [self.word_to_index.get(_, 1) for _ in prompt.lower().split()]
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= max_tokens:
            pad_len = self.maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:self.maxlen]
                sample_index = self.maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self._sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self._detokenize(_) for _ in start_tokens + tokens_generated]
        )
        return txt
