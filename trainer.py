import math
import torch.nn as nn
import numpy as np
from config import Config
from core.logger import TensorBoardLogger
from core.util import get_output_folder, get_common_membuf_location, get_output_folder_MA
from tester import Tester
from socket_scripts import *
from gym_server_util import *
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
                 sample_thres=.9, sim_agent=None, outputdir=None,
                 agent_id=-1, task_no=-1, default_logic=True, connection=None, 
                 s_sock=None, r_sock=None, atari_client=None):
        self.agent = agent
        # self.env = env
        self.config = config
        self.eval_model = eval_model
        self.sim_agent = sim_agent
        self.connection = connection
        self.s_sock = s_sock
        self.r_sock = r_sock
        self.atari_client = atari_client # atari_env_sock
        self.env_name = self.atari_client.atari_get_env_id() # env.unwrapped.spec.id
        
        # non-Linear epsilon decay
        epsilon_final = self.config.epsilon_min
        epsilon_start = self.config.epsilon
        epsilon_decay = self.config.eps_decay
        self.epsilon_by_frame = lambda frame_idx: epsilon_final + (epsilon_start - epsilon_final) * math.exp(
            -1. * frame_idx / epsilon_decay)
        self.agent_id = agent_id if agent_id !=-1 else self.config.agent_id
        self.task_no = task_no if task_no != -1 else self.config.task_no
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
            self.atari_client.atari_select_env("sim")
            total_frames = 10000
            agent.init_simnet(self.config.simnet_weight_dir)
        else:
            agent = self.agent
            time.sleep(1)
            self.atari_client.atari_select_env("train")
            agent.buffer.set_curr_idx(agent.buffer.size())
            shared_buffer_path = os.path.join(self.membufdir, "membuf_lsc_{}_task-{}.pkl".format(self.env_name, self.config.task_no))
            if self.config.apply_lsc_membuf and not os.path.exists(shared_buffer_path):
                with open(shared_buffer_path, 'wb') as f:
                    pickle.dump(agent.buffer.shared_buffer, f)
        train_starttime = time.time()
        time.sleep(0.1)
        state = self.atari_client.atari_reset() # env.reset()
        interaction_time_duration = 0.0
        for fr in range(pre_fr + 1, total_frames + 1):
            if fr % 100 == 0:
                print(fr)
            epsilon = self.epsilon_by_frame(fr)
            
            if learn_new_env:
                query_starttime = time.time()
                action = agent.act(state, epsilon, apply_ewc=apply_ewc)
                result = self.atari_client.atari_step(action)
                next_state, reward, done, _ = result
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
                state = self.atari_client.atari_reset() # env.reset()
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

            if self.share_info:
                comm_starttime = time.time()
                # agent.save_task_membuf(self.membufdir, self.env_name, self.agent_id, self.task_no)
                membuf_fn = "membuf_{}_agent-{}-task-{}.pkl".format(self.env_name, self.agent_id, self.task_no)
                # pdb.set_trace()
                local_membuf_path = os.path.join(self.outputdir, membuf_fn)
                if self.config.send_first:
                    server_send_simp(self.connection["connection_1_2"], membuf_fn)
                    server_send_simp(self.connection["connection_1_3"], membuf_fn)
                    server_send_simp(self.connection["connection_1_4"], membuf_fn)
                    server_send_simp(self.connection["connection_1_5"], membuf_fn)
                    server_send_simp(self.connection["connection_1_6"], membuf_fn)
                    server_send_simp(self.connection["connection_1_7"], membuf_fn)
                    
                    # pdb.set_trace()
                    time.sleep(7)
                    received_fn_agent2 = client_receive_simp(self.connection["connection_1_2"])
                    # pdb.set_trace()
                    time.sleep(7)
                    received_fn_agent3 = client_receive_simp(self.connection["connection_1_3"])
                    time.sleep(7)
                    received_fn_agent4 = client_receive_simp(self.connection["connection_1_4"])
                    # pdb.set_trace()
                    time.sleep(7)
                    received_fn_agent5 = client_receive_simp(self.connection["connection_1_5"])
                    time.sleep(7)
                    received_fn_agent6 = client_receive_simp(self.connection["connection_1_6"])
                    time.sleep(7)
                    received_fn_agent7 = client_receive_simp(self.connection["connection_1_7"])
                    time.sleep(7)
                    # print("waiting...")
                    # time.sleep(15)
                    if self.default_logic:
                        recv_membuf_fn_agent2 = "membuf_{}_agent-{}-task-{}.pkl".format("FOO", 2, self.task_no) # + 1)
                        recv_membuf_fn_agent3 = "membuf_{}_agent-{}-task-{}.pkl".format("FOO", 3, self.task_no)
                        recv_membuf_fn_agent4 = "membuf_{}_agent-{}-task-{}.pkl".format("FOO", 4, self.task_no)
                        recv_membuf_fn_agent5 = "membuf_{}_agent-{}-task-{}.pkl".format("FOO", 5, self.task_no)
                        recv_membuf_fn_agent6 = "membuf_{}_agent-{}-task-{}.pkl".format("FOO", 6, self.task_no)
                        recv_membuf_fn_agent7 = "membuf_{}_agent-{}-task-{}.pkl".format("FOO", 7, self.task_no)
                    else:
                        # pdb.set_trace()
                        # we asssume knowing the task identity
                        recv_membuf_fn_agent2 = received_fn_agent2
                        recv_membuf_fn_agent3 = received_fn_agent3
                        recv_membuf_fn_agent4 = received_fn_agent4
                        recv_membuf_fn_agent5 = received_fn_agent5
                        recv_membuf_fn_agent6 = received_fn_agent6
                        recv_membuf_fn_agent7 = received_fn_agent7
                    recv_membuf_path_agent2 = os.path.join(self.membufdir, recv_membuf_fn_agent2) # receive from agent 2
                    recv_membuf_path_agent3 = os.path.join(self.membufdir, recv_membuf_fn_agent3) # receive from agent 3
                    recv_membuf_path_agent4 = os.path.join(self.membufdir, recv_membuf_fn_agent4) # receive from agent 4
                    recv_membuf_path_agent5 = os.path.join(self.membufdir, recv_membuf_fn_agent5) # receive from agent 5
                    recv_membuf_path_agent6= os.path.join(self.membufdir, recv_membuf_fn_agent6) # receive from agent 6
                    recv_membuf_path_agent7 = os.path.join(self.membufdir, recv_membuf_fn_agent7) # receive from agent 7
                    
                    # print("Line 264")
                    # pdb.set_trace()
                    server_send(None, self.connection["connection_1_2"], # send to agent 2
                                send_file_path=local_membuf_path)
                    # print("Line 267")
                    server_send(None, self.connection["connection_1_3"], # send to agent 3
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_1_4"],
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_1_5"],
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_1_6"],
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_1_7"],
                                send_file_path=local_membuf_path)
                    time.sleep(14)
                    # print("Line 271")
                    client_receive(self.connection["connection_1_3"], recv_file_path=recv_membuf_path_agent3) # receive from agent 3
                    time.sleep(7)
                    # print("Line 274")
                    client_receive(self.connection["connection_1_2"], recv_file_path=recv_membuf_path_agent2) # receive from agent 2
                    # print("Line 276")
                    time.sleep(7)
                    client_receive(self.connection["connection_1_4"], recv_file_path=recv_membuf_path_agent4) # receive from agent 4
                    time.sleep(7)
                    client_receive(self.connection["connection_1_5"], recv_file_path=recv_membuf_path_agent5) # receive from agent 5
                    time.sleep(7)
                    client_receive(self.connection["connection_1_6"], recv_file_path=recv_membuf_path_agent6) # receive from agent 6
                    time.sleep(7)
                    client_receive(self.connection["connection_1_7"], recv_file_path=recv_membuf_path_agent7) # receive from agent 7
                    time.sleep(7)
                    
                    time.sleep(1)

                if self.config.send_second:
                    server_send_simp(self.r_sock["recv_sock_one_two"], membuf_fn)
                    server_send_simp(self.connection["connection_2_3"], membuf_fn)
                    server_send_simp(self.connection["connection_2_4"], membuf_fn)
                    server_send_simp(self.connection["connection_2_5"], membuf_fn)
                    server_send_simp(self.connection["connection_2_6"], membuf_fn)
                    server_send_simp(self.connection["connection_2_7"], membuf_fn)
                    # print("waiting...")
                    # time.sleep(15)
                    # pdb.set_trace()
                    time.sleep(7)
                    received_fn_agent1 = client_receive_simp(self.r_sock["recv_sock_one_two"])
                    # pdb.set_trace()
                    time.sleep(7)
                    received_fn_agent3 = client_receive_simp(self.connection["connection_2_3"])
                    time.sleep(7)
                    received_fn_agent4 = client_receive_simp(self.connection["connection_2_4"])
                    time.sleep(7)
                    received_fn_agent5 = client_receive_simp(self.connection["connection_2_5"])
                    time.sleep(7)
                    received_fn_agent6 = client_receive_simp(self.connection["connection_2_6"])
                    time.sleep(7)
                    received_fn_agent7 = client_receive_simp(self.connection["connection_2_7"])
                    time.sleep(7)
                    # pdb.set_trace()
                    if self.default_logic:
                        recv_membuf_fn_agent1 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 1, self.task_no) # + 1)
                        recv_membuf_fn_agent3 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 3, self.task_no)
                        recv_membuf_fn_agent4 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 4, self.task_no)
                        recv_membuf_fn_agent5 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 5, self.task_no)
                        recv_membuf_fn_agent6 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 6, self.task_no)
                        recv_membuf_fn_agent7 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 7, self.task_no)
                    else:
                        recv_membuf_fn_agent1 = received_fn_agent1
                        recv_membuf_fn_agent3 = received_fn_agent3
                        recv_membuf_fn_agent4 = received_fn_agent4
                        recv_membuf_fn_agent5 = received_fn_agent5
                        recv_membuf_fn_agent6=  received_fn_agent6
                        recv_membuf_fn_agent7 = received_fn_agent7
                        
                    recv_membuf_path_agent1 = os.path.join(self.membufdir, recv_membuf_fn_agent1) 
                    recv_membuf_path_agent3 = os.path.join(self.membufdir, recv_membuf_fn_agent3)
                    recv_membuf_path_agent4 = os.path.join(self.membufdir, recv_membuf_fn_agent4)
                    recv_membuf_path_agent5 = os.path.join(self.membufdir, recv_membuf_fn_agent5)
                    recv_membuf_path_agent6 = os.path.join(self.membufdir, recv_membuf_fn_agent6)
                    recv_membuf_path_agent7 = os.path.join(self.membufdir, recv_membuf_fn_agent7)
                    
                    
                    # print("Line 306")
                    server_send(None, self.connection["connection_2_3"], # send to agent 3
                                send_file_path=local_membuf_path)
                    # print("Line 309")
                    server_send(None, self.r_sock["recv_sock_one_two"], # send to agent 1
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_2_4"], # send to agent 4
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_2_5"], # send to agent 5
                                send_file_path=local_membuf_path)
                    
                    server_send(None, self.connection["connection_2_6"], # send to agent 6
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_2_7"], # send to agent 7
                                send_file_path=local_membuf_path)
                                
                    time.sleep(14)
                    # print("Line 312")
                    client_receive(self.r_sock["recv_sock_one_two"], recv_file_path=recv_membuf_path_agent1) # receive from agent 1
                    time.sleep(7)
                    # print("Line 314")
                    client_receive(self.connection["connection_2_3"], recv_file_path=recv_membuf_path_agent3) # receive from agent 3
                    # print("Line 316")
                    time.sleep(7)
                    client_receive(self.connection["connection_2_4"], recv_file_path=recv_membuf_path_agent4) # receive from agent 4
                    time.sleep(7)
                    client_receive(self.connection["connection_2_5"], recv_file_path=recv_membuf_path_agent5) # receive from agent 5
                    time.sleep(7)
                    client_receive(self.connection["connection_2_6"], recv_file_path=recv_membuf_path_agent6) # receive from agent 6
                    time.sleep(7)
                    client_receive(self.connection["connection_2_7"], recv_file_path=recv_membuf_path_agent7)   # receive from agent 7
                    time.sleep(1)

                if self.config.send_third:
                    # pdb.set_trace()
                    server_send_simp(self.connection["connection_3_4"], membuf_fn)
                    server_send_simp(self.connection["connection_3_5"], membuf_fn)
                    server_send_simp(self.connection["connection_3_6"], membuf_fn)
                    server_send_simp(self.connection["connection_3_7"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_one_three"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_two_three"], membuf_fn)

                    time.sleep(14)
                    received_fn_agent1 = client_receive_simp(self.r_sock["recv_sock_one_three"])
                    time.sleep(7)
                    # pdb.set_trace()
                    received_fn_agent2 = client_receive_simp(self.r_sock["recv_sock_two_three"])
                    time.sleep(7)
                    received_fn_agent4 = client_receive_simp(self.connection["connection_3_4"])
                    time.sleep(7)
                    received_fn_agent5 = client_receive_simp(self.connection["connection_3_5"])
                    time.sleep(7)
                    received_fn_agent6 = client_receive_simp(self.connection["connection_3_6"])
                    time.sleep(7)
                    received_fn_agent7 = client_receive_simp(self.connection["connection_3_7"])
                    time.sleep(7)
                    # pdb.set_trace()
                    if self.default_logic:
                        recv_membuf_fn_agent1 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 1, self.task_no) # + 1)
                        recv_membuf_fn_agent2 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 2, self.task_no)
                        recv_membuf_fn_agent4 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 4, self.task_no)
                        recv_membuf_fn_agent5 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 5, self.task_no)
                        recv_membuf_fn_agent6 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 6, self.task_no)
                        recv_membuf_fn_agent7 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 7, self.task_no)
                    else:
                        recv_membuf_fn_agent1 = received_fn_agent1
                        recv_membuf_fn_agent2 = received_fn_agent2
                        recv_membuf_fn_agent4 = received_fn_agent4
                        recv_membuf_fn_agent5 = received_fn_agent5
                        recv_membuf_fn_agent6 = received_fn_agent6
                        recv_membuf_fn_agent7 = received_fn_agent7
                    recv_membuf_path_agent1 = os.path.join(self.membufdir, recv_membuf_fn_agent1) 
                    recv_membuf_path_agent2 = os.path.join(self.membufdir, recv_membuf_fn_agent2)
                    recv_membuf_path_agent4 = os.path.join(self.membufdir, recv_membuf_fn_agent4)
                    recv_membuf_path_agent5 = os.path.join(self.membufdir, recv_membuf_fn_agent5)
                    recv_membuf_path_agent6 = os.path.join(self.membufdir, recv_membuf_fn_agent6)
                    recv_membuf_path_agent7 = os.path.join(self.membufdir, recv_membuf_fn_agent7)
                    # print("Line 346")
                    server_send(None, self.connection["connection_3_4"], # send to agent 4
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_3_5"], # send to agent 5
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_3_6"], # send to agent 6
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_3_7"], # send to agent 7
                                send_file_path=local_membuf_path)
                    # print("Line 349")
                    
                    server_send(None, self.r_sock["recv_sock_one_three"], # send to agent 1
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_two_three"], # send to agent 2
                                send_file_path=local_membuf_path)
                    time.sleep(14)
                    # print("Line 353")
                    client_receive(self.r_sock["recv_sock_two_three"], recv_file_path=recv_membuf_path_agent2) # receive from agent 2
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_one_three"], recv_file_path=recv_membuf_path_agent1) # receive from agent 1
                    # print("Line 356")
                    client_receive(self.connection["connection_3_7"], recv_file_path=recv_membuf_path_agent7) # receive from agent 7
                    time.sleep(7)
                    client_receive(self.connection["connection_3_6"], recv_file_path=recv_membuf_path_agent6) # receive from agent 6
                    time.sleep(7)
                    client_receive(self.connection["connection_3_5"], recv_file_path=recv_membuf_path_agent5) # receive from agent 5
                    time.sleep(7)
                    # print("Line 358")
                    client_receive(self.connection["connection_3_4"], recv_file_path=recv_membuf_path_agent4) # receive from agent 4
                    time.sleep(1)

                if self.config.send_fourth:
                    server_send_simp(self.r_sock["recv_sock_one_four"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_two_four"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_three_four"], membuf_fn)
                    server_send_simp(self.connection["connection_4_5"], membuf_fn)
                    server_send_simp(self.connection["connection_4_6"], membuf_fn)
                    server_send_simp(self.connection["connection_4_7"], membuf_fn)
                    time.sleep(7)
                    received_fn_agent1 = client_receive_simp(self.r_sock["recv_sock_one_four"])
                    time.sleep(7)
                    received_fn_agent2 = client_receive_simp(self.r_sock["recv_sock_two_four"])
                    time.sleep(7)
                    received_fn_agent3 = client_receive_simp(self.r_sock["recv_sock_three_four"])
                    time.sleep(7)
                    received_fn_agent5 = client_receive_simp(self.connection["connection_4_5"])
                    time.sleep(7)
                    received_fn_agent6 = client_receive_simp(self.connection["connection_4_6"])
                    time.sleep(7)
                    received_fn_agent7 = client_receive_simp(self.connection["connection_4_7"])
                    time.sleep(7)
                    if self.default_logic:
                        recv_membuf_fn_agent1 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 1, self.task_no) # + 1)
                        recv_membuf_fn_agent2 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 2, self.task_no)
                        recv_membuf_fn_agent3 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 3, self.task_no)
                        recv_membuf_fn_agent5 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 5, self.task_no)
                        recv_membuf_fn_agent6 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 6, self.task_no)
                        recv_membuf_fn_agent7 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 7, self.task_no)
                    else:
                        recv_membuf_fn_agent1 = received_fn_agent1
                        recv_membuf_fn_agent2 = received_fn_agent2
                        recv_membuf_fn_agent3 = received_fn_agent3
                        recv_membuf_fn_agent5 = received_fn_agent5
                        recv_membuf_fn_agent6 = received_fn_agent6
                        recv_membuf_fn_agent7 = received_fn_agent7
                        
                    recv_membuf_path_agent1 = os.path.join(self.membufdir, recv_membuf_fn_agent1)
                    recv_membuf_path_agent2 = os.path.join(self.membufdir, recv_membuf_fn_agent2)
                    recv_membuf_path_agent3 = os.path.join(self.membufdir, recv_membuf_fn_agent3)
                    recv_membuf_path_agent5 = os.path.join(self.membufdir, recv_membuf_fn_agent5)
                    recv_membuf_path_agent6 = os.path.join(self.membufdir, recv_membuf_fn_agent6)
                    recv_membuf_path_agent7 = os.path.join(self.membufdir, recv_membuf_fn_agent7)

                    # print("Line 375")
                    server_send(None, self.r_sock["recv_sock_one_four"], # send to agent 1
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_two_four"], # send to agent 2
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_three_four"], # send to agent 3
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_4_5"], # send to agent 5
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_4_6"], # send to agent 6
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_4_7"], # send to agent 7
                                send_file_path=local_membuf_path)
                    time.sleep(14)
                    # print("Line 383")
                    client_receive(self.r_sock["recv_sock_one_four"], recv_file_path=recv_membuf_path_agent1) # receive from agent 1
                    time.sleep(7)
                    # print("Line 386")
                    client_receive(self.r_sock["recv_sock_two_four"], recv_file_path=recv_membuf_path_agent2) # receive from agent 2
                    time.sleep(7)
                    # print("Line 389")
                    client_receive(self.r_sock["recv_sock_three_four"], recv_file_path=recv_membuf_path_agent3) # receive from agent 3
                    time.sleep(7)
                    client_receive(self.connection["connection_4_5"], recv_file_path=recv_membuf_path_agent5) # receive from agent 5
                    time.sleep(7)
                    client_receive(self.connection["connection_4_6"], recv_file_path=recv_membuf_path_agent6) # receive from agent 6
                    time.sleep(7)
                    client_receive(self.connection["connection_4_7"], recv_file_path=recv_membuf_path_agent7) # receive from agent 7
                    time.sleep(1)
                    
                    
                if self.config.send_fifth:
                    server_send_simp(self.r_sock["recv_sock_one_five"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_two_five"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_three_five"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_four_five"], membuf_fn)
                    server_send_simp(self.connection["connection_5_6"], membuf_fn)
                    server_send_simp(self.connection["connection_5_7"], membuf_fn)
                    time.sleep(7)
                    received_fn_agent1 = client_receive_simp(self.r_sock["recv_sock_one_five"])
                    time.sleep(7)
                    received_fn_agent2 = client_receive_simp(self.r_sock["recv_sock_two_five"])
                    time.sleep(7)
                    received_fn_agent3 = client_receive_simp(self.r_sock["recv_sock_three_five"])
                    time.sleep(7)
                    received_fn_agent4 = client_receive_simp(self.r_sock["recv_sock_four_five"])
                    time.sleep(7)
                    received_fn_agent6 = client_receive_simp(self.connection["connection_5_6"])
                    time.sleep(7)
                    received_fn_agent7 = client_receive_simp(self.connection["connection_5_7"])
                    time.sleep(7)
                    if self.default_logic:
                        recv_membuf_fn_agent1 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 1, self.task_no)
                        recv_membuf_fn_agent2 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 2, self.task_no)
                        recv_membuf_fn_agent3 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 3, self.task_no)
                        recv_membuf_fn_agent4 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 4, self.task_no)
                        recv_membuf_fn_agent6 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 6, self.task_no)
                        recv_membuf_fn_agent7 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 7, self.task_no)
                    else:
                        recv_membuf_fn_agent1 = received_fn_agent1
                        recv_membuf_fn_agent2 = received_fn_agent2
                        recv_membuf_fn_agent3 = received_fn_agent3
                        recv_membuf_fn_agent4 = received_fn_agent4
                        recv_membuf_fn_agent6 = received_fn_agent6
                        recv_membuf_fn_agent7 = received_fn_agent7
                    recv_membuf_path_agent1 = os.path.join(self.membufdir, recv_membuf_fn_agent1)
                    recv_membuf_path_agent2 = os.path.join(self.membufdir, recv_membuf_fn_agent2)
                    recv_membuf_path_agent3 = os.path.join(self.membufdir, recv_membuf_fn_agent3)
                    recv_membuf_path_agent4 = os.path.join(self.membufdir, recv_membuf_fn_agent4)
                    recv_membuf_path_agent6 = os.path.join(self.membufdir, recv_membuf_fn_agent6)
                    recv_membuf_path_agent7 = os.path.join(self.membufdir, recv_membuf_fn_agent7)
                    
                    server_send(None, self.r_sock["recv_sock_one_five"], # send to agent 1
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_two_five"], # send to agent 2
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_three_five"], # send to agent 3
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_four_five"], # send to agent 4
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_5_6"], # send to agent 6
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["connection_5_7"], # send to agent 7
                                send_file_path=local_membuf_path)
                    
                    time.sleep(14)
                    client_receive(self.r_sock["recv_sock_one_five"], recv_file_path=recv_membuf_path_agent1) # receive from agent 1
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_two_five"], recv_file_path=recv_membuf_path_agent2) # receive from agent 2
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_three_five"], recv_file_path=recv_membuf_path_agent3) # receive from agent 3
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_four_five"], recv_file_path=recv_membuf_path_agent4) # receive from agent 4
                    time.sleep(7)
                    client_receive(self.connection["connection_5_6"], recv_file_path=recv_membuf_path_agent6) # receive from agent 6
                    time.sleep(7)
                    client_receive(self.connection["connection_5_7"], recv_file_path=recv_membuf_path_agent7) # receive from agent 7
                    time.sleep(1)
                    
                if self.config.send_sixth:
                    server_send_simp(self.r_sock["recv_sock_one_six"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_two_six"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_three_six"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_four_six"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_five_six"], membuf_fn)
                    server_send_simp(self.connection["connection_6_7"], membuf_fn)
                    
                    time.sleep(7)
                    received_fn_agent1 = client_receive_simp(self.r_sock["recv_sock_one_six"])
                    time.sleep(7)
                    received_fn_agent2 = client_receive_simp(self.r_sock["recv_sock_two_six"])
                    time.sleep(7)
                    received_fn_agent3 = client_receive_simp(self.r_sock["recv_sock_three_six"])
                    time.sleep(7)
                    received_fn_agent4 = client_receive_simp(self.r_sock["recv_sock_four_six"])
                    time.sleep(7)
                    received_fn_agent5 = client_receive_simp(self.r_sock["recv_sock_five_six"])
                    time.sleep(7)
                    received_fn_agent7 = client_receive_simp(self.connection["connection_6_7"])
                    time.sleep(7)
                    if self.default_logic:
                        recv_membuf_fn_agent1 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 1, self.task_no)
                        recv_membuf_fn_agent2 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 2, self.task_no)
                        recv_membuf_fn_agent3 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 3, self.task_no)
                        recv_membuf_fn_agent4 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 4, self.task_no)
                        recv_membuf_fn_agent5 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 5, self.task_no)
                        recv_membuf_fn_agent7 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 7, self.task_no)
                    else:
                        recv_membuf_fn_agent1 = received_fn_agent1
                        recv_membuf_fn_agent2 = received_fn_agent2
                        recv_membuf_fn_agent3 = received_fn_agent3
                        recv_membuf_fn_agent4 = received_fn_agent4
                        recv_membuf_fn_agent5 = received_fn_agent5
                        recv_membuf_fn_agent7 = received_fn_agent7
                    recv_membuf_path_agent1 = os.path.join(self.membufdir, recv_membuf_fn_agent1)
                    recv_membuf_path_agent2 = os.path.join(self.membufdir, recv_membuf_fn_agent2)
                    recv_membuf_path_agent3 = os.path.join(self.membufdir, recv_membuf_fn_agent3)
                    recv_membuf_path_agent4 = os.path.join(self.membufdir, recv_membuf_fn_agent4)
                    recv_membuf_path_agent5 = os.path.join(self.membufdir, recv_membuf_fn_agent5)
                    recv_membuf_path_agent7 = os.path.join(self.membufdir, recv_membuf_fn_agent7)
                    
                    server_send(None, self.connection["connection_6_7"], # send to agent 7
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_five_six"], # send to agent 5
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_one_six"], # send to agent 1
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_two_six"], # send to agent 2
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_three_six"], # send to agent 3
                                send_file_path=local_membuf_path)
                    server_send(None, self.connection["recv_sock_four_six"], # send to agent 4
                                send_file_path=local_membuf_path)
                    
                    time.sleep(14)
                    client_receive(self.r_sock["recv_sock_one_six"], recv_file_path=recv_membuf_path_agent1) # receive from agent 1
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_two_six"], recv_file_path=recv_membuf_path_agent2) # receive from agent 2
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_three_six"], recv_file_path=recv_membuf_path_agent3) # receive from agent 3
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_four_six"], recv_file_path=recv_membuf_path_agent4) # receive from agent 4
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_five_six"], recv_file_path=recv_membuf_path_agent5) # receive from agent 5
                    time.sleep(7)
                    client_receive(self.connection["connection_6_7"], recv_file_path=recv_membuf_path_agent7) # receive from agent 7
                    time.sleep(1)
                    
                    
                if self.config.send_seventh:
                    server_send_simp(self.r_sock["recv_sock_one_seven"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_two_seven"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_three_seven"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_four_seven"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_five_seven"], membuf_fn)
                    server_send_simp(self.r_sock["recv_sock_six_seven"], membuf_fn)
                    
                    time.sleep(7)
                    received_fn_agent1 = client_receive_simp(self.r_sock["recv_sock_one_seven"])
                    time.sleep(7)
                    received_fn_agent2 = client_receive_simp(self.r_sock["recv_sock_two_seven"])
                    time.sleep(7)
                    received_fn_agent3 = client_receive_simp(self.r_sock["recv_sock_three_seven"])
                    time.sleep(7)
                    received_fn_agent4 = client_receive_simp(self.r_sock["recv_sock_four_seven"])
                    time.sleep(7)
                    received_fn_agent5 = client_receive_simp(self.r_sock["recv_sock_five_seven"])
                    time.sleep(7)
                    received_fn_agent6 = client_receive_simp(self.r_sock["recv_sock_six_seven"])
                    time.sleep(7)
                    if self.default_logic:
                        recv_membuf_fn_agent1 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 1, self.task_no)
                        recv_membuf_fn_agent2 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 2, self.task_no)
                        recv_membuf_fn_agent3 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 3, self.task_no)
                        recv_membuf_fn_agent4 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 4, self.task_no)
                        recv_membuf_fn_agent5 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 5, self.task_no)
                        recv_membuf_fn_agent6 = "membuf_{}_agent-{}-task-{}.pkl".format("BAR", 6, self.task_no)
                    else:
                        recv_membuf_fn_agent1 = received_fn_agent1
                        recv_membuf_fn_agent2 = received_fn_agent2
                        recv_membuf_fn_agent3 = received_fn_agent3
                        recv_membuf_fn_agent4 = received_fn_agent4
                        recv_membuf_fn_agent5 = received_fn_agent5
                        recv_membuf_fn_agent6 = received_fn_agent6
                        
                    recv_membuf_path_agent1 = os.path.join(self.membufdir, recv_membuf_fn_agent1)
                    recv_membuf_path_agent2 = os.path.join(self.membufdir, recv_membuf_fn_agent2)
                    recv_membuf_path_agent3 = os.path.join(self.membufdir, recv_membuf_fn_agent3)
                    recv_membuf_path_agent4 = os.path.join(self.membufdir, recv_membuf_fn_agent4)
                    recv_membuf_path_agent5 = os.path.join(self.membufdir, recv_membuf_fn_agent5)
                    recv_membuf_path_agent6 = os.path.join(self.membufdir, recv_membuf_fn_agent6)
                    
                    server_send(None, self.r_sock["recv_sock_one_seven"], # send to agent 1
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_two_seven"], # send to agent 2
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_three_seven"], # send to agent 3
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_four_seven"], # send to agent 4
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_five_seven"], # send to agent 5
                                send_file_path=local_membuf_path)
                    server_send(None, self.r_sock["recv_sock_six_seven"], # send to agent 6
                                send_file_path=local_membuf_path)
                    
                    time.sleep(14)
                    client_receive(self.r_sock["recv_sock_one_seven"], recv_file_path=recv_membuf_path_agent1) # receive from agent 1
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_two_seven"], recv_file_path=recv_membuf_path_agent2) # receive from agent 2
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_three_seven"], recv_file_path=recv_membuf_path_agent3) # receive from agent 3
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_four_seven"], recv_file_path=recv_membuf_path_agent4) # receive from agent 4
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_five_seven"], recv_file_path=recv_membuf_path_agent5) # receive from agent 5
                    time.sleep(7)
                    client_receive(self.r_sock["recv_sock_six_seven"], recv_file_path=recv_membuf_path_agent6) # receive from agent 6
                    time.sleep(1)
                    
                    
                    
                    
                    
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
    
    def learn_by_thres(self, agent, fr, apply_ewc=False, apply_lsc=False):
        if random.random() >= self.sample_thres:
            if apply_lsc:
                return agent.learning(fr, self.loss_fn, apply_ewc=apply_ewc, focus_curr=False, use_lsc=apply_lsc)
            else:
                return agent.learning(fr, self.loss_fn, apply_ewc=apply_ewc, focus_curr=True)
        else:
            return agent.learning(fr, self.loss_fn, apply_ewc=apply_ewc, focus_curr=False)
    
