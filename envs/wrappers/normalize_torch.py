import gym
import torch


# taken from https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_normalize.py
class RunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = torch.zeros(shape).to('cuda:0')
        self.var = torch.ones(shape).to('cuda:0')
        self.count = epsilon

    def update(self, x):
        """ update from a batch of samples"""
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.mean, self.var, self.count = self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        new_count = tot_count
        return new_mean, new_var, new_count


class NormalizeObservation(gym.core.Wrapper):
    """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
    Note:
        The normalization depends on past trajectories and observations
        will not be normalized correctly if the wrapper was
        newly instantiated or the policy was changed recently.
    """

    def __init__(self, env: gym.Env, epsilon: float = 1e-8):
        """This wrapper will normalize observations s.t. each coordinate is centered with unit variance.
        Args:
            env (Env): The environment to apply the wrapper
            epsilon: A stability parameter that is used when scaling the observations.
        """
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = True
        self.obs_rms = RunningMeanStd(shape=self.single_observation_space.shape)
        self.epsilon = epsilon

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, dones, info = self.env.step(action)
        obs = self.normalize(obs)
        return obs, rews, dones, info

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs = self.env.reset(**kwargs)
        return self.normalize(obs)

    def normalize(self, obs):
        """Normalises the observation using the running mean and variance of the observations."""
        self.obs_rms.update(obs)
        return (obs - self.obs_rms.mean) / torch.sqrt(self.obs_rms.var + self.epsilon)
    
    def save_obs_rms(self, path):
        """Saves the observation running mean and variance to a file."""
        torch.save(self.obs_rms, path)

    def load_obs_rms(self, path):
        """Loads the observation running mean and variance from a file."""
        self.obs_rms = torch.load(path)


class NormalizeReward(gym.core.Wrapper):
    r"""This wrapper will normalize immediate rewards s.t. their exponential moving average has a fixed variance.
    The exponential moving average will have variance :math:`(1 - \gamma)^2`.
    Note:
        The scaling depends on past trajectories and rewards will not be scaled correctly if the wrapper was newly
        instantiated or the policy was changed recently.
    """
    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.is_vector_env = True
        self.return_rms = RunningMeanStd(shape=())
        self.returns = torch.zeros((self.num_envs, 1)).to('cuda:0')
        self.gamma = gamma
        self.epsilon = epsilon
        self.total_reward = torch.zeros((self.num_envs,)).to('cuda:0')

    def step(self, action):
        """Steps through the environment, normalizing the rewards returned."""
        obs, rews, dones, info = self.env.step(action)
        self.total_reward += rews
        self.returns = self.returns * self.gamma + rews
        rews = self.normalize(rews)
        self.returns[dones.long()] = 0.0
        return obs, rews, dones, info

    def normalize(self, rews):
        """Normalizes the rewards with the running mean rewards and their variance."""
        self.return_rms.update(self.returns)
        return rews / torch.sqrt(self.return_rms.var + self.epsilon)

    def reset(self, **kwargs):
        """Resets the environment and normalizes the observation."""
        obs = self.env.reset(**kwargs)
        self.total_reward = torch.zeros((self.num_envs,)).to('cuda:0')
        return obs
    
    def save_return_rms(self, path):
        """Saves the reward running mean and variance to a file."""
        torch.save(self.return_rms, path)

    def load_return_rms(self, path):
        """Loads the reward running mean and variance from a file."""
        self.return_rms = torch.load(path)
