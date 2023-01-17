#!/usr/bin/env python
import os,sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import argparse
import time

from multiagent.environment import MultiAgentEnv
from multiagent.policy import InteractivePolicy
import multiagent.scenarios as scenarios

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-s', '--scenario', default='simple.py', help='Path of the scenario Python script.')
    args = parser.parse_args()

    # load scenario from script
    scenario = scenarios.load(args.scenario).Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, info_callback=None, shared_viewer=False, discrete_action=True)
    print ('\033[1;32mobs: {}\nacs: {}\033[1;0m'.format(env.observation_space, env.action_space))
    # render call to create viewer window (necessary only for interactive policies)
    env.render()
    # create interactive policies for each agent
    policies = [InteractivePolicy(env,i) for i in range(env.n)]
    # execution loop
    obs_n = env.reset()
    episode_limit = 25
    step_idx = 0
    while True:
        start = time.time()
        # query for action from each agent's policy
        act_n = []
        for i, policy in enumerate(policies):
            act_n.append(policy.action(obs_n[i]))
        # print ('[{}] observation: {}'.format(step_idx, obs_n))
        print ('[{}] action: {}'.format(step_idx, act_n))
        # step environment
        obs_n, reward_n, done_n, _ = env.step(act_n)
        step_idx += 1

        if step_idx >= episode_limit:
            break

        # render all agent views
        env.render()
        # end = time.time()
        # elapsed = end - start
        # time.sleep(max(1 / 30 - elapsed, 0))
        # display rewards
        #for agent in env.world.agents:
        #    print(agent.name + " reward: %0.3f" % env._get_reward(agent))
