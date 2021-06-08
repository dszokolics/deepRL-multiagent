import numpy as np
import torch
from torch.optim import Adam

from models import Actor, Critic


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPGAgent():
    """Interacts with and learns from the environment.
        
    Args:
        state_size (int): Dimension of each state
        action_size (int): Dimension of each action
        random_seed (int): Random seed
        num_agents (int): Number of DDPG agents within the MADDPG
        agent_params (dict): Parameters of the agent
        network_params (dict): Layer sizes for the actors and critics
        noise_params (dict): Noise parameters

    """
    
    def __init__(self, state_size, action_size, random_seed, num_agents, agent_params, network_params, noise_params):
        """Create DDPG agent."""
        self.params = agent_params
        self.tau = agent_params["tau"]

        # Initialize the actor
        self.actor = Actor(state_size, action_size, random_seed, **network_params["actor"]).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed, **network_params["actor"]).to(device)
        self.hard_update(self.actor, self.actor_target)
        
        self.actor_optimizer = Adam(self.actor.parameters(), lr=agent_params["lr_actor"], weight_decay=0)

        # Initialize the critic
        self.critic = (
            Critic(state_size * num_agents, action_size*num_agents, random_seed, **network_params["critic"])
            .to(device)
        )
        self.critic_target = (
            Critic(state_size * num_agents, action_size*num_agents, random_seed, **network_params["critic"])
            .to(device)
        )
        self.hard_update(self.critic, self.critic_target)
        
        self.critic_optimizer = Adam(self.critic.parameters(), lr=agent_params["lr_critic"], weight_decay=0)

        # Initialize the noise
        self.noise = OUNoise(action_size, noise_params)
    
    def act(self, obs):
        """Act with the local network. A noise is used to introduce randomness to the process.
        
        Args:
            obs (torch.FloatTensor): Observation space.

        """
        obs = obs.to(device)
        action = self.actor(obs) + self.noise.noise()
        return torch.clamp(action, -1, 1)

    def target_act(self, obs):
        """Act with the target network.
        
        Args:
            obs (torch.FloatTensor): Observation space.

        """
        obs = obs.to(device)
        action = self.actor_target(obs)
        return action
    
    def reset(self):
        """Resets the noise"""
        self.noise.reset()
        
    def reduce_noise(self):
        self.noise.reduce()

    def hard_update(self, local_model, target_model):
        """Copy network parameters from source to target

        Args:
            local_model (torch.nn.Module): Net whose parameters to copy
            target_model (torch.nn.Module): Net to copy parameters to

        """
        for target_param, param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(param.data)
        
    def soft_update(self, local_model, target_model):
        """Perform soft update (move target params toward source based on weight factor tau)
        
        Args:
            local_model (torch.nn.Module): Net whose parameters to copy
            target_model (torch.nn.Module): Net to copy parameters to

        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)
            
    def save(self, path):
        torch.save(self.actor.state_dict(), path + '_ckpt_actor.pth')
        torch.save(self.actor_target.state_dict(), path + '_ckpt_actor_target.pth')
        torch.save(self.critic.state_dict(), path + '_ckpt_critic.pth')
        torch.save(self.critic_target.state_dict(), path + '_ckpt_critic_target.pth')
        
    def load(self, path):
        self.actor.load_state_dict(torch.load(path + '_ckpt_actor.pth'))
        self.actor_target.load_state_dict(torch.load(path + '_ckpt_actor_target.pth'))
        self.critic.load_state_dict(torch.load(path + '_ckpt_critic.pth'))
        self.critic_target.load_state_dict(torch.load(path + '_ckpt_critic_target.pth'))


class OUNoise:
    """Implements the Ornstein-Uhlenback process for noise sampling.
    
    Args:
        action_dimension (int): Size of the outputted noise
        params (dict): Parameters that control the process
        
    """

    def __init__(self, action_dimension, params):
        """Initialize the noise."""
        self.action_dimension = action_dimension
        self.scale = params["scale"]
        self.mu = params["mu"]
        self.theta = params["theta"]
        self.sigma = params["sigma"]
        self.reduction = params["noise_reduction"]
        self.min_noise = params["min_noise"]

        self.reset()

    def reset(self):
        """Reset the noise to its original state."""
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        """Generate noise."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx

        return torch.tensor(self.state * self.scale).float()
    
    def reduce(self):
        self.scale = max(self.scale * self.reduction, self.min_noise)
