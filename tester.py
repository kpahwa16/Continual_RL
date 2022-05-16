import json
import os
from core.util import get_output_folder


class Tester(object):

    def __init__(self, agent, env, model_path, load_weights=True, num_episodes=20, max_ep_steps=600000, test_ep_steps=2000):
        self.num_episodes = num_episodes
        self.max_ep_steps = max_ep_steps
        self.test_ep_steps = test_ep_steps
        self.agent = agent
        self.env = env
        self.env_name = self.env.unwrapped.spec.id
        self.outputdir = get_output_folder(self.agent.config.output, self.agent.config.env)
        self.agent.is_training = False
        # self.agent.load_weights(model_path)
        self.policy = lambda x: agent.act(x)
        if load_weights:
            self.agent.load_weights(model_path)

    def test(self, debug=False, save_result=False): # , visualize=True):
        avg_reward = 0
        reward_history = [] # debug
        for episode in range(self.num_episodes):
            s0 = self.env.reset()
            episode_steps = 0
            episode_reward = 0.

            done = False
            
            reward_history = []
            while not done:
                # if visualize:
                #     self.env.render()

                action = self.policy(s0)
                s0, reward, done, info = self.env.step(action)
                episode_reward += reward
                episode_steps += 1

                reward_history.append(episode_reward)

                if episode_steps + 1 > self.test_ep_steps:
                    done = True
            # with open("test_reward_by_episode_{}.json".format(self.env_name), 'w') as f:
            #     json.dump(reward_history, f)
            # break
        
            if debug:
                print('[Test] episode: %3d, episode_reward: %5f' % (episode, episode_reward))
                reward_history.append(episode_reward)

            avg_reward += episode_reward
        avg_reward /= self.num_episodes 
        print("avg reward: %5f" % (avg_reward))
        # if save_result:
        #     with open(os.path.join(self.outputdir, "test_reward_by_episode_{}.json".format(self.env_name)), 'w') as f:
        #         json.dump(reward_history, f)

        return avg_reward

