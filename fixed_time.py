import random

class Agent():
    def __init__(self, obs_shape, act_shape, env_id="fixed_time_agent", fixed_time=4, name="") -> None:
        self.obs_shape = obs_shape
        self.act_shape = act_shape
        self.env_id = env_id
        self.fixed_time = fixed_time
        self._counter = 0
        self.action = random.choice(range(self.act_shape))

    def act(self, obs, rew, done=False):
        self._counter += 1
        if self._counter % self.fixed_time == 0:
            self.action += 1

        return self.action % self.act_shape

    def episode_end(self, env_id):
        pass


