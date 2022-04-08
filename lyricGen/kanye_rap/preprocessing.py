import numpy as np
import re
import os

def load_file():
    base_path = "kanye.txt"
    file = open(base_path, "r", encoding = "utf8")
    text = file.read()
    text = text.replace("\n\n", "\n")
    return text

def clean_lyric(lyric):
    return re.sub("[^a-z' ]", "", lyric).replace("'", "")

def cleaned_lyrics(text):
    lyrics = text.lower().split("\n")
    lyrics = np.unique(lyrics)[1:].tolist()
    cleaned_lyrics = [clean_lyric(lyric) for lyric in lyrics]
    return cleaned_lyrics

def create_sequences(lyric, seq_len):
    sequences = []
    if len(lyric.split()) <= seq_len:
        return [lyric]
    
    for i in range(seq_len, len(lyric.split())):
        curr_seq = lyric.split()[i - seq_len:i + 1]
        sequences.append(" ".join(curr_seq))
    
    return sequences

def get_seq_idx(word_to_idx,seq):
    return [word_to_idx[word] for word in seq.split()]

def get_preprocessed_data():
    text = load_file()
    lyrics = cleaned_lyrics(text)
    seq_size = 5
    raw_sequences = [create_sequences(lyric, seq_size) for lyric in lyrics]
    sequences = np.unique(np.array(sum(raw_sequences, []))).tolist()

    uniq_words = np.unique(np.array(" ".join(sequences).split(" ")))
    uniq_words_idx = np.arange(uniq_words.size)

    word_to_idx = dict(zip(uniq_words.tolist(), uniq_words_idx.tolist()))
    idx_to_word = dict(zip(uniq_words_idx.tolist(), uniq_words.tolist()))

    vocab_size = len(word_to_idx)

    x_word = []
    y_word = []

    # iterate through every sequence
    for seq in sequences:
        
        # stop if the sequence isn't long enough
        if (len(seq.split()) != seq_size + 1):
            continue
        
        # add the words to the sequences
        x_word.append(" ".join(seq.split()[:-1]))
        y_word.append(" ".join(seq.split()[1:]))
    
    x_idx = np.array([get_seq_idx(word_to_idx ,word) for word in x_word])
    y_idx = np.array([get_seq_idx(word_to_idx, word) for word in y_word])
    return x_idx, y_idx, vocab_size, word_to_idx, idx_to_word, uniq_words