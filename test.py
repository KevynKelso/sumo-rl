import sys
from utils_plot import trial_post_proc
import os
from numpy import mean

import sumo_rl
sys.path.append("../StateStreetSumo")
from dqn import Agent as DQNAgent
from fixed_time import Agent as FixedAgent

def main():
    env_id="_dqn_multi_2"
    data_dir = "Data"+env_id
    info_csv = data_dir+os.sep+"infos/info"

    env = sumo_rl.parallel_env(
        net_file='nets/RESCO/grid4x4/grid4x4.net.xml',
        route_file='myroute.rou.xml',
        out_csv_name=info_csv,
        use_gui=False,
        num_seconds=1800,
        # add_system_info=True,
        # render_mode="human",
    )
    observations, infos = env.reset()
    myagents = {}
    for agent in env.agents:
        myagents[agent] = DQNAgent(observations[agent].shape, 6, env_id=env_id, name=agent)

    done = False
    total_reward = 0
    for episode in range(50):
        print(f"Episode {episode} start")
        while not done:
            actions = {agent: myagents[agent].act(observations[agent], total_reward, done) for agent in env.agents}
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            done = True if True in terminations.values() or True in truncations.values() else False
            total_reward = mean(list(rewards.values()))
            observations = next_observations

        [myagent.episode_end(env_id) for myagent in myagents.values()]
        done = False
        observations, infos = env.reset()

    trial_post_proc(data_dir, "DQN")

if __name__ == '__main__':
    main()
