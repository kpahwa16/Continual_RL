import math
import torch.nn as nn
import numpy as np
from config import Config
from core.logger import TensorBoardLogger
from core.util import get_output_folder, get_common_membuf_location, get_output_folder_MA
from tester import Tester
from socket_scripts import *
import pickle
import random
import json
import gym
import os
import pdb
import time


class Trainer:
    def __init__(self, agent, env, config, loss_fn=None, test_env=None, 
                 eval_model=False, num_test_times=1,
                 sample_thres=.9, sim_agent=None, 
                 sim_env=None, outputdir=None,
                 agent_id=-1, task_no=-1, default_logic=True, connection=None, 
                 s_sock=None, r_sock=None):
        self.agent = agent
        self.env = env
        self.env_name = env.unwrapped.spec.id
        self.config = config
        # self.test_env = test_env
        self.eval_model = eval_model
        self.sim_agent = sim_agent
        self.sim_env = sim_env
        self.connection = connection
        self.s_sock = s_sock
        self.r_sock = r_sock

        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)
        self.agent_id = agent_id if agent_id !=-1 else self.config.agent_id
        self.task_no = task_no if task_no != -1 else self.config.task_no
        # self.outputdir = outputdir if outputdir is not None else get_output_folder(self.config.output, self.config.env, agent_id=self.agent_id, task_no=self.task_no)
        self.outputdir = outputdir if outputdir is not None else get_output_folder_MA(self.config.output, agent_id=self.agent_id)
        self.agent.save_config(self.outputdir)
        self.board_logger = TensorBoardLogger(self.outputdir)
        self.loss_fn = nn.MSELoss() if loss_fn is None else loss_fn
        self.num_test_times = 1
        self.apply_sample_thres = self.config.apply_sample_thres
        self.apply_lsc_membuf = self.config.apply_lsc_membuf
        self.sample_thres = self.config.sample_thres
        self.default_logic = default_logic
        self.share_info = self.config.share_info
        if self.config.use_membuf:
            self.membufdir = get_common_membuf_location(self.config.membuf_parent_savedir, self.config.membuf_savedir)
        
    def train(self, 
              pre_fr=0, 
              apply_ewc=False, 
              learn_new_env=True, 
              use_membuf=False,
              use_simnet=False):
        losses = []
        all_rewards = []
        all_train_rewards = []
        train_mean_rewards = []
        test_rewards = []
        episode_reward = 0
        ep_num = 0
        is_win = False
        final_fr = -1
        total_frames = self.config.frames if learn_new_env else len(self.agent.buffer.prev_buffer)
        if use_simnet:
            agent = self.sim_agent
            env = self.sim_env
            total_frames = 10000
            agent.init_simnet(self.config.simnet_weight_dir)
        else:
            agent = self.agent
            env = self.env
            # agent.save_model(self.outputdir, 'init_{}'.format(self.env_name))
            agent.buffer.set_curr_idx(agent.buffer.size())
            shared_buffer_path = os.path.join(self.membufdir, "membuf_lsc_{}_task-{}.pkl".format(self.env_name, self.config.task_no))
            if self.config.apply_lsc_membuf and not os.path.exists(shared_buffer_path):
                with open(shared_buffer_path, 'wb') as f:
                    pickle.dump(agent.buffer.shared_buffer, f)
        train_starttime = time.time()
        state = env.reset()
        interaction_time_duration = 0.0
        for fr in range(pre_fr + 1, total_frames + 1):
            epsilon = self.epsilon_by_frame(fr)
            
            if learn_new_env:
                query_starttime = time.time()
                action = agent.act(state, epsilon, apply_ewc=apply_ewc)
                next_state, reward, done, _ = env.step(action)
                query_duration = time.time() - query_starttime
                interaction_time_duration += query_duration
                agent.buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
            else:
                query_starttime = time.time()
                state, action, reward, next_state, done = agent.buffer.get_past_buffer_samples()
                query_duration = time.time() - query_starttime
                interaction_time_duration += query_duration
                agent.buffer.add(state[0], action, reward, next_state[0], done)
                episode_reward += reward
            
            loss = 0
            buffer_size = agent.buffer.size()
            if self.apply_sample_thres and (buffer_size - agent.buffer.curr_start_idx) > self.config.batch_size:
                # pdb.set_trace()
                loss = self.learn_by_thres(agent, fr, apply_ewc=apply_ewc)
            elif self.apply_lsc_membuf and (buffer_size - agent.buffer.curr_start_idx) > self.config.batch_size:
                # pdb.set_trace()
                loss = self.learn_by_thres(agent, fr, apply_ewc=apply_ewc, apply_lsc=self.apply_lsc_membuf)
            elif buffer_size > self.config.batch_size: # without biasing towards samples of current task
                loss = agent.learning(fr, self.loss_fn, apply_ewc=apply_ewc)
            losses.append(loss)
            
            self.board_logger.scalar_summary('Loss per frame', fr, loss)
            if fr % self.config.print_interval == 0:
                print("frames: %5d, reward: %5f, loss: %4f episode: %4d" % (fr, np.mean(all_rewards[-100:]), loss, ep_num))
                train_mean_rewards.append((fr, np.mean(all_rewards[-100:])))
                
            if not use_membuf:
                # low-switching cost on memory
                if self.config.apply_lsc_membuf and fr % self.config.num_frames_save_buf == 0:
                    print("save contents to shared buffer")
                    agent.buffer.update_shared_buffer(self.config.num_frames_save_buf, shared_buffer_path)
                if self.config.apply_lsc_membuf and fr % self.config.num_frames_load_buf == 0:
                    print("load contents from shared buffer")
                    agent.buffer.load_shared_buffer(shared_buffer_path)
                
                # logging
                if fr % self.config.log_interval == 0:
                    # print('Reward per episode: ep_num = {}, reward = {}'.format(ep_num, all_rewards[-1]))
                    self.board_logger.scalar_summary('Reward per episode', ep_num, all_rewards[-1])
                if self.config.checkpoint and fr % self.config.checkpoint_interval == 0:
                    agent.save_checkpoint(fr, self.outputdir)
                if fr % 10000 == 0:
                    # agent.save_model(self.outputdir, 'fr_{}_{}'.format(fr, self.env_name))
                    if self.eval_model:
                        test_avg_reward = self.evaluate(num_episodes=self.num_test_times) # default is 1
                        test_rewards.append(test_avg_reward)
                        print("frames {}: test reward {}, on {}".format(fr, test_avg_reward, self.test_env.unwrapped.spec.id))
            if done:
                state = env.reset()
                all_rewards.append(episode_reward)
                episode_reward = 0
                ep_num += 1
                avg_reward = float(np.mean(all_rewards[-100:]))
                bst_reward = float(np.max(all_rewards[-100:]))
                self.board_logger.scalar_summary('Best 100-episodes average reward', ep_num, avg_reward)
                if len(all_rewards) >= 100 and avg_reward >= self.config.win_reward and all_rewards[-1] > self.config.win_reward: # and all_rewards[-1] > best_reward:
                    best_reward = all_rewards[-1]
                    is_win = True
                    out_model_fn = 'best_{}'.format(self.env_name) if not use_simnet else 'best_simnet_{}'.format(self.env_name)
                    agent.save_model(self.outputdir, out_model_fn)
                    final_fr = fr # update final frame number for saving fisher matrix
                    print('Ran %d episodes best 100-episodes average reward is %3f. Best 100-episode reward is %3f. Solved after %d trials ✔' % (ep_num, avg_reward, bst_reward, ep_num - 100))
                    if self.config.win_break:
                        break
        train_duration = time.time() - train_starttime
        train_other_time = train_duration - interaction_time_duration
        print("Duration of training in Trainer.train(): {}".format(train_duration))
        print("Duration of interacting with env: {} / {}".format(interaction_time_duration, train_duration))
        print("Duration of remaining activities: {} / {}".format(train_other_time, train_duration))
        
        if apply_ewc:
            with open(os.path.join(self.outputdir, "debug_ewc_loss.json"), 'w') as f6:
                json.dump(agent.debug_ewc_loss, f6)
            agent.estimate_fisher_matrix(agent.config.batch_size, self.loss_fn)
            agent.save_fisher_matrix(final_fr, self.outputdir)

        if not use_simnet:
            # get portion of buffer that has samples of the current task
            agent.buffer.update_prev_buffer()
            # save history memory buffer
            # with open(os.path.join(self.outputdir, "membuf_history_{}_agent-{}.pkl".format(self.env_name, self.agent_id)), 'wb') as f:
            #     pickle.dump(agent.buffer.buffer, f)
            
            # if learn_new_env: # only saves memory replay buffer to file when learning a new task
            #     agent.save_task_membuf(self.outputdir, self.env_name, self.agent_id, self.task_no)
            agent.save_task_membuf(self.outputdir, self.env_name, self.agent_id, self.task_no)

            # 0223: add share info logic to allow communication among agents
            # pdb.set_trace()
            if self.share_info:
                comm_starttime = time.time()
                # agent.save_task_membuf(self.membufdir, self.env_name, self.agent_id, self.task_no)
                membuf_fn = "membuf_{}_agent-{}-task-{}.pkl".format(self.env_name, self.agent_id, self.task_no)
                
                # send the buffer file name to the other agent
                server_send_simp(self.connection, membuf_fn)
                received_fn = client_receive_simp(self.r_sock)
                # pdb.set_trace()

                local_membuf_path = os.path.join(self.outputdir, membuf_fn)
                other_agent_id = 2 if self.agent_id % 2 else 1
                if self.config.send_first:
                    # print("sending membuf to the other agent")
                    server_send(self.s_sock, self.connection, 
                                send_file_path=local_membuf_path)
                    # print("waiting...")
                    # time.sleep(15)
                    # print("receive")
                    if self.default_logic:
                        recv_membuf_fn = "membuf_{}_agent-{}-task-{}.pkl".format("FOO", other_agent_id, self.task_no) # + 1)
                    else:
                        # pdb.set_trace()
                        # we asssume knowing the task identity
                        recv_membuf_fn = received_fn
                    recv_membuf_path = os.path.join(self.membufdir, recv_membuf_fn)
                    client_receive(self.r_sock, recv_file_path=recv_membuf_path)
                    # print("membuf received")
                else:
                    # print("waiting...")
                    # time.sleep(15)
                    # print("receiving membuf from other agent")
                    if self.default_logic:
                        recv_membuf_fn = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", other_agent_id, self.task_no) # + 1)
                    else:
                        recv_membuf_fn = received_fn
                    recv_membuf_path = os.path.join(self.membufdir, recv_membuf_fn)
                    client_receive(self.r_sock, recv_file_path=recv_membuf_path)
                    # print("membuf received")
                    # print("then send")
                    server_send(self.s_sock, self.connection,
                                send_file_path=local_membuf_path)
                    # print("membuf sent")
                print("Duration of communication in Trainer.train(): {}".format(time.time() - comm_starttime))

            with open(os.path.join(self.outputdir, "all_rewards_{}.json".format(self.env_name)), 'w') as f:
                json.dump(all_rewards, f)
            with open(os.path.join(self.outputdir, "train_mean_reward_per_{}_frame_{}.json".format(self.config.print_interval, self.env_name)), 'w') as f4:
                json.dump(train_mean_rewards, f4)
            if not is_win:
                print('Did not solve after %d episodes' % ep_num)
                agent.save_model(self.outputdir, 'last_{}_agent-{}-task-{}'.format(self.env_name, self.agent_id, self.task_no))
                # agent.save_model(self.membufdir, 'last_{}_agent-{}-task-{}'.format(self.env_name, self.agent_id, self.task_no))
                # agent.save_model(self.membufdir, 'last_agent_{}_task_{}_{}'.format(self.agent_id, self.task_no, self.env_name))
        else:
            if not is_win:
                print('Did not solve after %d episodes' % ep_num)
                print("Saved simnet to: {}".format(self.outputdir))
                agent.save_model(self.outputdir, 'last_simnet_agent_{}_task_{}_{}'.format(self.agent_id, self.task_no, self.env_name))
        
    """def evaluate(self, debug=False, num_episodes=50, test_ep_steps=600000): # now the dafault is 1
        avg_reward = 0
        policy = lambda x: self.agent.act(x)
        for episode in range(num_episodes):
            s0 = self.test_env.reset()
            episode_steps = 0
            episode_reward = 0.
            done = False
            while not done:
                action = policy(s0)
                s0, reward, done, info = self.test_env.step(action)
                episode_reward += reward
                episode_steps += 1
                if episode_steps + 1 > test_ep_steps:
                    done = True
            if debug:
                print('[Test] episode: %3d, episode_reward: %5f' % (episode, episode_reward))
            avg_reward += episode_reward
        avg_reward /= num_episodes 
        return avg_reward"""
    
    def learn_by_thres(self, agent, fr, apply_ewc=False, apply_lsc=False):
        if random.random() >= self.sample_thres:
            if apply_lsc:
                return agent.learning(fr, self.loss_fn, apply_ewc=apply_ewc, focus_curr=False, use_lsc=apply_lsc)
            else:
                return agent.learning(fr, self.loss_fn, apply_ewc=apply_ewc, focus_curr=True)
        else:
            return agent.learning(fr, self.loss_fn, apply_ewc=apply_ewc, focus_curr=False)
    