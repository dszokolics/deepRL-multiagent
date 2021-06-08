from collections import deque
import random
from utils import transpose_list


class ReplayBuffer:
    """Buffer that holds the experience of the agents.
    
    Args:
        size (int): Number of experiences to hold.
        
    """
    def __init__(self, size):
        """Initialize an empty buffer."""
        self.size = size
        self.deque = deque(maxlen=self.size)

    def push(self, transition):
        """Save experience
        
        Args:
            transition (tuple of torch.FloatTensor): The experience
            
        """
        self.deque.append(transition)

    def sample(self, batchsize):
        """Sample from the buffer
        
        Args:
            batchsize (int): Number of experiences to return.
        
        Returns:
            tuple of torch.FloatTensor: The sampled experiences.
        """
        samples = random.sample(self.deque, batchsize)
        return transpose_list(samples)

    def __len__(self):
        return len(self.deque)
