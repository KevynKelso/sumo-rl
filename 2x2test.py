import os
from sumo_rl import SumoEnvironment
from utils_plot import trial_post_proc
import sys
sys.path.append("../StateStreetSumo")
from dqn import Agent as DQNAgent
from fixed_time import Agent as FixedAgent


def main():
    env_id="_fixed_single_1"
    data_dir = "Data"+env_id
    info_csv = data_dir+os.sep+"infos/info"
    env = SumoEnvironment(
        net_file="nets/2way-single-intersection/single-intersection.net.xml",
        route_file="nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name=info_csv,
        single_agent=True,
        use_gui=False,
        num_seconds=1800,
        # render_mode="human",
    )
    observations, infos = env.reset()

    myagent = FixedAgent(observations.shape, 4, env_id=env_id)
    done = False
    reward = 0
    for episode in range(5):
        while not done:
            action = myagent.act(observations, reward, done)
            next_observation, reward, termination, truncation, infos = env.step(action)
            done = termination or truncation
            observations = next_observation

        myagent.episode_end(env_id)
        done = False
        observations, infos = env.reset()

    trial_post_proc(data_dir, "Fixed Time")

if __name__ == '__main__':
    main()
