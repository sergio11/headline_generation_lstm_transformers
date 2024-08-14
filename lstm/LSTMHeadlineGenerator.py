import string
import unicodedata
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class LstmWrapper:
    """
    A wrapper class for generating text using a pre-trained LSTM model.

    Attributes:
        model (tf.keras.Model): The loaded LSTM model.
        tokenizer (Tokenizer): The tokenizer used for text processing.
        max_sequence_len (int): Maximum length of the input sequences.
    """
    
    def __init__(self, model_path='trained_model.model.h5', weights_path='trained_model.weights.h5', tokenizer_path='tokenizer.pkl', max_sequence_len=100):
        """
        Initializes the LstmWrapper instance.

        Args:
            model_path (str): Path to the saved model file (`.h5` format). Default is 'trained_model.model.h5'.
            weights_path (str): Path to the saved weights file (`.h5` format). Default is 'trained_model.weights.h5'.
            tokenizer_path (str): Path to the saved tokenizer file (`.pkl` format). Default is 'tokenizer.pkl'.
            max_sequence_len (int): Maximum length of the input sequences. Default is 100.
        """
        # Load the pre-trained model
        self.model = load_model(model_path)
        # Load the model weights
        self.model.load_weights(weights_path)
        # Load the tokenizer
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        # Set the maximum sequence length
        self.max_sequence_len = max_sequence_len

    def clean_and_normalize_text(self, txt):
        """
        Cleans and normalizes the input text.

        Args:
            txt (str): The text to be cleaned and normalized.

        Returns:
            str: The cleaned and normalized text.
        """
        # Remove punctuation and convert text to lowercase
        txt = "".join(c for c in txt if c not in string.punctuation).lower()
        # Normalize unicode characters and encode to ASCII
        txt = unicodedata.normalize('NFKD', txt).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        return txt

    def generate_text_from_prompt(self, prompt, num_words):
        """
        Generates text based on a given starting prompt.

        Args:
            prompt (str): The starting text prompt for text generation.
            num_words (int): The number of words to generate following the prompt.

        Returns:
            str: The generated text with the specified number of words appended to the prompt.
        """
        generated_text = prompt

        for _ in range(num_words):
            # Preprocess the prompt
            prompt_proc = self.clean_and_normalize_text(generated_text)
            prompt_proc = self.tokenizer.texts_to_sequences([prompt_proc])[0]
            prompt_proc = pad_sequences([prompt_proc], maxlen=self.max_sequence_len-1, padding='pre')

            # Predict the next word
            predict = self.model.predict(prompt_proc, verbose=0)
            predicted_index = np.argmax(predict, axis=1)[0]

            # Convert predicted index to word
            next_word = ""
            for word, index in self.tokenizer.word_index.items():
                if index == predicted_index:
                    next_word = word
                    break

            # Append the predicted word to the generated text
            generated_text += " " + next_word

        return generated_text.title()