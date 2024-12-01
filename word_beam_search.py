import numpy as np
import tensorflow as tf


class WordBeamSearch:
    def __init__(self, beam_width=50, lexicon=None, word_chars=None, corpus=None, vocab=None):
        """
        Word Beam Search Decoder for CTC.
        Args:
            beam_width: Number of hypotheses to consider during decoding.
            lexicon: A list of valid words.
            word_chars: A string of valid characters forming words.
            corpus: A string representing the corpus for word candidates.
            vocab: A vocabulary of characters (including blank) from the model.
        """
        self.beam_width = beam_width
        self.lexicon = lexicon  # Optional: list of valid words
        self.word_chars = word_chars  # Optional: string of characters that form words
        self.corpus = corpus  # Corpus of valid words
        self.vocab = vocab  # Vocabulary of characters

    def decode(self, logits):
        """
        Perform Word Beam Search decoding over CTC logits.
        Args:
            logits: The predicted logits from the CTC network (softmax probabilities).
        Returns:
            best_sequence: The most probable sequence of words decoded from the logits.
        """
        # Initialize the beam with the first time step's logits
        beams = [([], 0.0)]  # list of (sequence, score)

        for t in range(logits.shape[0]):  # Iterate through each time step
            new_beams = []
            for seq, score in beams:  # For each hypothesis in the beam
                for token, token_prob in enumerate(logits[t]):  # For each possible token at time step
                    new_seq = seq + [token]
                    new_score = score + np.log(token_prob + 1e-10)  # Add a small value to prevent log(0)
                    new_beams.append((new_seq, new_score))

            # Sort by score and keep only the top `beam_width` sequences
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_width]

        # Decode the best beam to words using the lexicon or corpus
        best_sequence = self.convert_to_words(beams[0][0])
        return best_sequence

    def convert_to_words(self, sequence):
        """
        Convert the token sequence (indices) to words using the lexicon.
        Args:
            sequence: The sequence of tokens (indices) from the decoder.
        Returns:
            decoded_words: The decoded word sequence.
        """
        words = []
        current_word = []
        for token in sequence:
            char = self.vocab[token]  # Convert token index to character
            if char == ' ' or char in self.word_chars:
                if current_word:
                    words.append(''.join(current_word))
                current_word = [char]
            else:
                current_word.append(char)

        if current_word:
            words.append(''.join(current_word))
        
        return ' '.join(words)

    def compute(self, logits):
        """
        Compute the most probable word sequence using Word Beam Search.
        Args:
            logits: The raw output from the neural network (softmax probabilities).
        Returns:
            decoded_words: The best decoded word sequence.
        """
        return self.decode(logits)