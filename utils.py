import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Dataset(Dataset):
    """
    Custom Dataset class for loading encoded question-reply pairs.

    Attributes:
        pairs (list): List of encoded question-reply pairs.
        dataset_size (int): The number of question-reply pairs in the dataset.
    """

    def __init__(self) -> None:
        """
        Initialize the dataset by loading the encoded pairs from a JSON file.
        """
        self.pairs = json.load(open('pairs_encoded.json'))
        self.dataset_size = len(self.pairs)

    def __getitem__(self, i: int) -> tuple:
        """
        Fetches a specific question-reply pair.

        Args:
            i (int): Index of the pair to fetch.

        Returns:
            tuple: The question and reply tensors.
        """
        question = torch.LongTensor(self.pairs[i][0])
        reply = torch.LongTensor(self.pairs[i][1])
        return question, reply

    def __len__(self) -> int:
        """
        Returns the total number of pairs in the dataset.

        Returns:
            int: The size of the dataset.
        """
        return self.dataset_size


def create_masks(question: torch.Tensor, reply_input: torch.Tensor, reply_target: torch.Tensor) -> tuple:
    """
    Create masks for the input questions and replies for the transformer model.

    Args:
        question (torch.Tensor): Tensor containing the question input sequence (batch_size, max_words).
        reply_input (torch.Tensor): Tensor containing the reply input sequence (batch_size, max_words).
        reply_target (torch.Tensor): Tensor containing the reply target sequence (batch_size, max_words).

    Returns:
        tuple: Question mask, reply input mask, and reply target mask.
    """

    def subsequent_mask(size: int) -> torch.Tensor:
        """
        Create a mask to prevent attending to future positions in the sequence.

        Args:
            size (int): Size of the sequence.

        Returns:
            torch.Tensor: Subsequent mask of shape (1, size, size).
        """
        mask = torch.triu(torch.ones(size, size)).transpose(0, 1).type(dtype=torch.uint8)
        return mask.unsqueeze(0)

    # Question mask
    question_mask = (question != 0).to(device).unsqueeze(1).unsqueeze(1)  # (batch_size, 1, 1, max_words)

    # Reply input mask
    reply_input_mask = reply_input != 0
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words)
    reply_input_mask = reply_input_mask & subsequent_mask(reply_input.size(-1)).type_as(reply_input_mask.data)
    reply_input_mask = reply_input_mask.unsqueeze(1)  # (batch_size, 1, max_words, max_words)

    # Reply target mask
    reply_target_mask = (reply_target != 0)  # (batch_size, max_words)

    return question_mask, reply_input_mask, reply_target_mask


class AdamWarmup:
    """
    Custom learning rate scheduler for the Adam optimizer with warmup steps.

    Attributes:
        model_size (int): Dimension of the model.
        warmup_steps (int): Number of warmup steps.
        optimizer (torch.optim.Optimizer): Adam optimizer.
        current_step (int): Current step of the optimizer.
        lr (float): Current learning rate.
    """

    def __init__(self, model_size: int, warmup_steps: int, optimizer: torch.optim.Optimizer) -> None:
        """
        Initializes the AdamWarmup class.

        Args:
            model_size (int): Size of the model.
            warmup_steps (int): Number of warmup steps.
            optimizer (torch.optim.Optimizer): Adam optimizer.
        """
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.optimizer = optimizer
        self.current_step = 0
        self.lr = 0.0

    def get_lr(self) -> float:
        """
        Compute the learning rate for the current step.

        Returns:
            float: The current learning rate.
        """
        return (self.model_size ** (-0.5) *
                min(self.current_step ** (-0.5), self.current_step * self.warmup_steps ** (-1.5)))

    def step(self) -> None:
        """
        Update the optimizer learning rate and perform an optimization step.
        """
        self.current_step += 1
        self.lr = self.get_lr()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr

        self.optimizer.step()


class LossWithLS(nn.Module):
    """
    Loss function with label smoothing.

    Attributes:
        criterion (nn.KLDivLoss): KLDivLoss to compute the loss.
        confidence (float): Confidence score for the correct labels.
        smooth (float): Smoothing factor for the incorrect labels.
        size (int): Size of the vocabulary.
    """

    def __init__(self, size: int, smooth: float) -> None:
        """
        Initialize the loss function with label smoothing.

        Args:
            size (int): Size of the vocabulary.
            smooth (float): Smoothing factor.
        """
        super(LossWithLS, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.confidence = 1.0 - smooth
        self.smooth = smooth
        self.size = size

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss with label smoothing.

        Args:
            prediction (torch.Tensor): Model output predictions (batch_size, max_words, vocab_size).
            target (torch.Tensor): Target labels (batch_size, max_words).
            mask (torch.Tensor): Mask to apply on the loss (batch_size, max_words).

        Returns:
            torch.Tensor: The computed loss.
        """
        prediction = prediction.view(-1, prediction.size(-1))  # (batch_size * max_words, vocab_size)
        target = target.contiguous().view(-1)  # (batch_size * max_words)
        mask = mask.float().view(-1)  # (batch_size * max_words)

        # Create smoothed labels
        labels = prediction.data.clone()
        labels.fill_(self.smooth / (self.size - 1))
        labels.scatter_(1, target.data.unsqueeze(1), self.confidence)

        # Compute the loss
        loss = self.criterion(prediction, labels)
        loss = (loss.sum(1) * mask).sum() / mask.sum()  # Normalize loss by mask

        return loss
