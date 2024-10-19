import json
import logging
from collections import Counter
from typing import List, Dict, Tuple

# Logging setup to monitor progress and debug information
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Paths to the corpus files
corpus_movie_conv: str = 'data/corpus.json'
corpus_movie_lines: str = 'data/speakers.json'

# Hyperparameters
max_len: int = 25  # Maximum number of words per question/answer pair
min_word_freq: int = 5  # Minimum frequency for a word to be included in the vocabulary

def read_file(file_path: str) -> List[str]:
    """
    Reads a file and returns the lines as a list of strings.

    Args:
        file_path (str): Path to the file.

    Returns:
        List[str]: List of lines from the file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.readlines()
    except FileNotFoundError:
        logging.error(f"File {file_path} not found.")
        return []
    except Exception as e:
        logging.error(f"Error reading {file_path}: {e}")
        return []

# Reading movie conversations and lines
conv: List[str] = read_file(corpus_movie_conv)
lines: List[str] = read_file(corpus_movie_lines)

# Parsing movie lines into a dictionary
lines_dic: Dict[str, str] = {}
for line in lines:
    objects = line.split(" +++$+++ ")
    if len(objects) == 5:  # Ensure correct format
        lines_dic[objects[0]] = objects[-1].strip()

def remove_punc(string: str) -> str:
    """
    Remove punctuation from a string and convert it to lowercase.

    Args:
        string (str): Input string.

    Returns:
        str: String without punctuation and in lowercase.
    """
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    table = str.maketrans('', '', punctuations)
    return string.translate(table).lower()

# Create pairs of consecutive lines (question-answer)
pairs: List[Tuple[List[str], List[str]]] = []
for con in conv:
    try:
        ids: List[str] = eval(con.split(" +++$+++ ")[-1])  # Extract the list of line IDs
        for i in range(len(ids) - 1):  # Create (question, answer) pairs
            first: List[str] = remove_punc(lines_dic.get(ids[i], '')).split()[:max_len]
            second: List[str] = remove_punc(lines_dic.get(ids[i + 1], '')).split()[:max_len]
            if first and second:  # Only consider non-empty pairs
                pairs.append((first, second))
    except (KeyError, SyntaxError) as e:
        logging.warning(f"Skipping invalid conversation entry: {con} due to {e}")

# Word frequency counter
word_freq: Counter = Counter()
for pair in pairs:
    word_freq.update(pair[0])
    word_freq.update(pair[1])

# Creating the word map (vocabulary)
words: List[str] = [w for w in word_freq.keys() if word_freq[w] >= min_word_freq]
word_map: Dict[str, int] = {k: v + 1 for v, k in enumerate(words)}
word_map['<unk>'] = len(word_map) + 1
word_map['<start>'] = len(word_map) + 1
word_map['<end>'] = len(word_map) + 1
word_map['<pad>'] = 0  # Padding token is assigned index 0

logging.info(f"Vocabulary size (words appearing > {min_word_freq} times): {len(word_map)}")

# Save the word map to a JSON file
with open('WORDMAP_corpus.json', 'w') as j:
    json.dump(word_map, j)
logging.info(f"Word map saved to 'WORDMAP_corpus.json'")

def encode_question(words: List[str], word_map: Dict[str, int]) -> List[int]:
    """
    Encodes the question into a fixed-length vector using word_map. Pads with '<pad>' tokens.

    Args:
        words (List[str]): List of words in the question.
        word_map (Dict[str, int]): Mapping of words to indices.

    Returns:
        List[int]: Encoded question, padded to max_len.
    """
    enc_q: List[int] = [word_map.get(word, word_map['<unk>']) for word in words]
    enc_q += [word_map['<pad>']] * (max_len - len(enc_q))
    return enc_q

def encode_reply(words: List[str], word_map: Dict[str, int]) -> List[int]:
    """
    Encodes the reply with start and end tokens, padded to max_len.

    Args:
        words (List[str]): List of words in the reply.
        word_map (Dict[str, int]): Mapping of words to indices.

    Returns:
        List[int]: Encoded reply, including <start> and <end> tokens, padded to max_len.
    """
    enc_r: List[int] = [word_map['<start>']] + [word_map.get(word, word_map['<unk>']) for word in words] + [word_map['<end>']]
    enc_r += [word_map['<pad>']] * (max_len - len(enc_r))
    return enc_r

# Encode all pairs (questions and replies)
pairs_encoded: List[Tuple[List[int], List[int]]] = []
for pair in pairs:
    qus: List[int] = encode_question(pair[0], word_map)
    ans: List[int] = encode_reply(pair[1], word_map)
    pairs_encoded.append((qus, ans))

# Save the encoded pairs to a JSON file
with open('pairs_encoded.json', 'w') as p:
    json.dump(pairs_encoded, p)
logging.info(f"Encoded pairs saved to 'pairs_encoded.json'")
