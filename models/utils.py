import torch
from torch.nn import functional as F


def attention_weights(query, keys):
    """

    Args:
        query (batch_size, match_dim): attention query object
        keys (batch_size, timesteps, match_dim): attention key objects

    Returns:
        attention weights of shape (batch_size, timesteps) which are probability
        distributions along last dimension
    """
    query = query.unsqueeze(1)
    sim = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
    # sim.shape == (batch_size, timesteps)
    sim = sim - sim.max(1, keepdim=True)[0]
    return F.softmax(sim, 1)


def attention(query, keys, values):
    """

    Args:
        query (batch_size, match_dim): attention query object
        keys (batch_size, timesteps, match_dim): attention key objects
        values (batch_size, timesteps, some_dim): attention values

    Returns:
        attended values
    """
    attn = attention_weights(query, keys).unsqueeze(1)
    return torch.bmm(attn, values).squeeze()
