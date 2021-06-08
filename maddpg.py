import torch

from utils import transpose_to_tensor
from ddpg import DDPGAgent


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """Coordinates multiple competing / collaborating DDPG agents
    
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
        """Create and initialize the base agents."""
        super(MADDPG, self).__init__()
        self.params = agent_params
        self.gamma = agent_params["gamma"]

        # Generate a num_agents amount of players
        self.maddpg_agent = [
            DDPGAgent(state_size, action_size, random_seed, num_agents, agent_params, network_params, noise_params)
            for _
            in range(num_agents)
        ]
    
    def save(self, checkpoint_path):
        """Save the params of the underlying agents.
        
        Args:
            checkpoint_path (str): Path used for saving the agent
            
        """
        for agent_num, agent in enumerate(self.maddpg_agent):
            agent.save(checkpoint_path +'_'+ str(agent_num))

    def load(self, checkpoint_path):
        """Load the params of the underlying agents.
        
        Args:
            checkpoint_path (str): Path used for loading the agent
            
        """
        for agent_num, agent in enumerate(self.maddpg_agent):
            agent.load(checkpoint_path +'_'+ str(agent_num))
    
    def act(self, obs_all_agents):
        """Get actions from all agents in the MADDPG object.
        
        Args:
            obs_all_agents (torch.FloatTensor): The observations for all agents.
            noise (float): Scale of the noise. Set it to zero if the agent is not used for training.
                Default is zero.
                
        Returns:
            torch.FloatTensor: Proposed actions
            
        """
        actions = [agent.act(obs) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents):
        """Get target network actions from all the agents in the MADDPG object.
        
        Args:
            obs_all_agents (torch.FloatTensor): The observations for all agents.
            noise (float): Scale of the noise. Set it to zero if the agent is not used for training.
                Default is zero.
                
        Returns:
            torch.FloatTensor: Proposed actions
            
        """
        target_actions = [agent.target_act(obs) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return target_actions
    
    def reset(self):
        """Reset the noise for each agents."""
        for a in self.maddpg_agent:
            a.reset()
            
    def reduce_noise(self):
        for a in self.maddpg_agent:
            a.reduce_noise()

    def update(self, samples, agent_number):
        """Update the critics and actors of all the agents.
        
        Args:
            samples (tuple of torch.FloatTensor): Experience to learn from.
            agent_number (int): The agent to update.
            
        """

        obs, action, reward, next_obs, done = transpose_to_tensor(samples) # get data

        # full versions of obs and actions are needed for the critics
        obs_full = torch.cat((obs[0], obs[1]), 1)
        next_obs_full = torch.cat((next_obs[0], next_obs[1]), 1)
        action_full = torch.cat((action[0], action[1]), 1)
        
        agent = self.maddpg_agent[agent_number]
        agent.critic_optimizer.zero_grad()

        #critic loss = batch mean of (y- Q(s,a) from target network)^2
        #y = reward of this timestep + discount * Q(st+1,at+1) from target network
        target_actions = self.target_act(next_obs)
        target_actions = torch.cat(target_actions, dim=1)
        
        q_next = agent.critic_target(next_obs_full, target_actions).detach()
                
        y = (
            reward[agent_number].unsqueeze(-1)
            + (self.gamma * q_next * (1 - done[agent_number].unsqueeze(-1)))
        )
        q = agent.critic(obs_full, action_full)

        loss = torch.nn.MSELoss()
        critic_loss = loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()

        #update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = [ self.maddpg_agent[i].actor(ob) if i == agent_number \
                   else self.maddpg_agent[i].actor(ob).detach()
                   for i, ob in enumerate(obs) ]
                
        q_input = torch.cat(q_input, dim=1)
        
        # get the policy gradient
        actor_loss = -agent.critic(obs_full, q_input).mean()
        actor_loss.backward()
        agent.actor_optimizer.step()
        
        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic, agent.critic_target)
        agent.soft_update(agent.actor, agent.actor_target)
