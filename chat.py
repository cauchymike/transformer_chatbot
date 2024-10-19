import json
import torch
import torch.utils.data
from models import *  
from utils import * 

# Load the word map from the JSON file
with open('WORDMAP_corpus.json', 'r') as j:
    word_map = json.load(j)

# Flag to load a pre-trained transformer model checkpoint
load_checkpoint = True

# Path to the model checkpoint
ckpt_path = 'checkpoint.pth.tar'


def evaluate(transformer: torch.nn.Module, question: torch.Tensor, question_mask: torch.Tensor, max_len: int, word_map: dict) -> str:
    """
    Performs Greedy Decoding with a batch size of 1 to generate a response for the given input.

    Args:
        transformer (torch.nn.Module): The transformer model to use for encoding and decoding.
        question (torch.Tensor): The input question encoded as a tensor of word indices.
        question_mask (torch.Tensor): A mask for the question tensor, indicating non-zero tokens.
        max_len (int): The maximum number of tokens to generate for the response.
        word_map (dict): A dictionary mapping words to their indices.

    Returns:
        str: The generated sentence as a string.
    """
    # Create a reverse word map to map indices back to words for decoding
    rev_word_map = {v: k for k, v in word_map.items()}

    # Set the transformer model to evaluation mode (disables dropout, etc.)
    transformer.eval()

    # Start decoding with the '<start>' token
    start_token = word_map['<start>']

    # Encoder step: encode the question with the transformer
    encoded = transformer.encode(question, question_mask)

    # Initialize the words tensor with the start token, on the correct device
    words = torch.LongTensor([[start_token]]).to(device)

    # Iteratively decode the response up to the max length
    for step in range(max_len - 1):
        # Prepare the target mask (causal mask) for the decoder
        size = words.shape[1]
        target_mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        target_mask = target_mask.to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, size, size)

        # Decode the current words tensor to get predictions for the next token
        decoded = transformer.decode(words, target_mask, encoded, question_mask)

        # Get the predictions (logits) for the next word
        predictions = transformer.logit(decoded[:, -1])

        # Use greedy decoding (select the word with the highest logit)
        _, next_word = torch.max(predictions, dim=1)
        next_word = next_word.item()

        # Break if the '<end>' token is predicted
        if next_word == word_map['<end>']:
            break

        # Append the predicted word to the words tensor
        words = torch.cat([words, torch.LongTensor([[next_word]]).to(device)], dim=1)  # (1, step + 2)

    # Convert the word indices back to words to form the sentence
    if words.dim() == 2:
        words = words.squeeze(0)  # Remove batch dimension
        words = words.tolist()  # Convert to list

    # Remove the '<start>' token from the sentence and form the final sentence
    sen_idx = [w for w in words if w not in {word_map['<start>']}]
    sentence = ' '.join([rev_word_map[sen_idx[k]] for k in range(len(sen_idx))])

    return sentence  # Return the generated sentence


# Load a checkpoint if the flag is set
if load_checkpoint:
    # Load the model and other state information from the checkpoint
    checkpoint = torch.load(ckpt_path)
    transformer = checkpoint['transformer']  # Transformer model

# Main loop for the chatbot interaction
while True:
    question = input("Question: ")  # Get user input for a question
    if question == 'quit':
        break  # Exit the loop if the user types 'quit'

    max_len = input("Maximum Reply Length: ")  # Get the max response length from the user

    # Encode the question using the word map, handling unknown words with the '<unk>' token
    enc_qus = [word_map.get(word, word_map['<unk>']) for word in question.split()]

    # Convert the encoded question to a tensor and add a batch dimension
    question = torch.LongTensor(enc_qus).to(device).unsqueeze(0)

    # Create a mask for the question tensor (non-zero tokens)
    question_mask = (question != 0).to(device).unsqueeze(1).unsqueeze(1)  # Shape: (1, 1, 1, seq_len)

    # Generate a response using the evaluate function
    sentence = evaluate(transformer, question, question_mask, int(max_len), word_map)

    # Print the generated response
    print(sentence)
