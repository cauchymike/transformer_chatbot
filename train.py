import json
import torch
from torch.utils.data import Dataset, DataLoader
from models import Transformer
from utils import AdamWarmup, LossWithLS, create_masks

# Constants
BATCH_SIZE = 100
d_model = 512
heads = 8
num_layers = 6
epochs = 10

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load word map
with open('WORDMAP_corpus.json', 'r') as j:
    word_map = json.load(j)

# Dataset Loader
train_loader = DataLoader(Dataset(), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Model, Optimizer, Criterion setup
transformer = Transformer(d_model=d_model, heads=heads, num_layers=num_layers, word_map=word_map).to(device)
adam_optimizer = torch.optim.Adam(transformer.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
transformer_optimizer = AdamWarmup(model_size=d_model, warmup_steps=4000, optimizer=adam_optimizer)
criterion = LossWithLS(len(word_map), 0.1)


def train_model(train_loader: DataLoader, transformer: Transformer, criterion: LossWithLS, epoch: int) -> None:
    """
    Trains the transformer model for one epoch.

    Args:
        train_loader (DataLoader): DataLoader providing batches of training data.
        transformer (Transformer): Transformer model to be trained.
        criterion (LossWithLS): Loss function with label smoothing.
        epoch (int): Current epoch number.

    Returns:
        None
    """
    transformer.train()
    sum_loss = 0
    count = 0

    for i, (question, reply) in enumerate(train_loader):
        samples = question.shape[0]

        # Move data to device
        question = question.to(device)
        reply = reply.to(device)

        # Prepare target data
        reply_input = reply[:, :-1]
        reply_target = reply[:, 1:]

        # Create masks
        question_mask, reply_input_mask, reply_target_mask = create_masks(question, reply_input, reply_target)

        # Get model output
        out = transformer(question, question_mask, reply_input, reply_input_mask)

        # Compute the loss
        loss = criterion(out, reply_target, reply_target_mask)

        # Backpropagation
        transformer_optimizer.optimizer.zero_grad()
        loss.backward()
        transformer_optimizer.step()

        # Accumulate loss
        sum_loss += loss.item() * samples
        count += samples

        if i % 100 == 0:
            print(f"Epoch [{epoch}][{i}/{len(train_loader)}]\tLoss: {sum_loss/count:.3f}")


def save_checkpoint(state: dict, filename: str) -> None:
    """
    Saves the model and optimizer state to a file.

    Args:
        state (dict): The state dictionary containing the model and optimizer.
        filename (str): Path to save the checkpoint.

    Returns:
        None
    """
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


if __name__ == "__main__":
    # Run the training loop for the specified number of epochs
    for epoch in range(epochs):
        train_model(train_loader, transformer, criterion, epoch)

        # Save the model checkpoint
        state = {
            'epoch': epoch,
            'transformer': transformer.state_dict(),
            'transformer_optimizer': transformer_optimizer.optimizer.state_dict()
        }
        save_checkpoint(state, f'checkpoint_{epoch}.pth.tar')
