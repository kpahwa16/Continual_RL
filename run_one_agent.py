import argparse
import os
import random
import torch
import gym
import time
import pdb
import pickle
import torch.nn as nn
from copy import deepcopy
from config import Config
from torch import autograd
from torch.optim import Adam
from trainer import Trainer
from tester import Tester
from buffer import ReplayBuffer
from common.wrappers import add_noop, add_frame_skip, add_random_action
from core.util import get_class_attr_val
from model import CnnDQNTypeTwo, DQN, TinyDQN
from agents import DQNAgentTypeThree
from socket_scripts import *
from gym_server_util import *
from task_similarity import compare_weight_similarity, find_replay_buffer, find_replay_buffer_nosim


def training(config, agents, default_logic=True, send_socket=None, 
             receive_socket=None, connection=None, atari_client=None):
    """
    This functions performs training process with inter-client sharing for all tasks in task sequences.
    
    Args:
        config: configuration
        agents: CANAL agent
        default_logic: set to True to run train_per_agent() for every task: 
                                   Share files with other agents and receive every other agent's buffer. 
                                   Reuse a buffer if it matches what this agent is about to learn.
                       set to False to run train_per_agent_logic2() for every task: 
                                   Train each task from scratch.
                                   Perform task similarity comparison with SimNets shared by other agents.
                                   Replay the closest candidates' buffers.
                       send_socket: sender socket API.
                       receive_socket: receiver socket API.
                       connection: obsolete.
                       atari_client: API of Atari environment server for querying training data.
    """
    for i in range(config.num_tasks):
        if default_logic:
            train_one_round(config, agents, i, train_func=train_per_agent, 
                            send_socket=send_socket, 
                            receive_socket=receive_socket, 
                            connection=connection,
                            atari_client=atari_client)
        else:
            train_one_round(config, agents, i, train_func=train_per_agent_logic2, 
                            send_socket=send_socket, 
                            receive_socket=receive_socket,
                            connection=connection,
                            atari_client=atari_client)
        

def train_one_round(config, agents, epoch_id, train_func=None, send_socket=None, 
                    receive_socket=None, connection=None, atari_client=None):
    """
    Defines training framework for one round. One learning round is composed learning a task 
    and sharing learnable information with other agents.
    """
    for j, agent in enumerate(agents):
        print("Training agent {} on task {}".format(config.agent_id, epoch_id + 1))
        round_starttime = time.time()
        train_func(config, agent, epoch_id, config.agent_id, send_socket=send_socket, 
                   receive_socket=receive_socket, connection=connection, atari_client=atari_client)
        round_endtime = time.time()
        print("Time for agent {} to complete round {}: {}".format(config.agent_id, epoch_id + 1, round_endtime - round_starttime))
    
def train_per_agent_logic2(config, agent, task_id, agent_id, send_socket=None, 
                           receive_socket=None, connection=None, atari_client=None):
    """
    Train each task from scratch.
    Perform task similarity comparison with SimNets shared by other agents.
    Replay the closest candidates' buffers.
    """
    atari_client.atari_create_env(agent.task_sequences[agent.task_no])
    time.sleep(1)
    trainer = Trainer(agent, 
                      atari_client, # train_env, 
                      config, 
                      sim_agent=None, 
                      agent_id=agent_id,
                      task_no=task_id,
                      default_logic=False, # set this to False to use second logic
                      connection=connection,
                      s_sock=send_socket,
                      r_sock=receive_socket,
                      atari_client=atari_client)
    print("Training task agent for task {}".format(agent.task_no + 1))
    
    if config.share_info and agent.task_no > 0:
        all_replay_fns = [fn for fn in os.listdir(trainer.membufdir) if fn.startswith("membuf_")]
        our_task = trainer.agent.task_sequences[trainer.agent.task_no].split("_")[0]
        replay_fns = find_replay_buffer_nosim(all_replay_fns, 
                                              agent_id=agent_id, 
                                              task_id=trainer.agent.task_no, 
                                              task_name=our_task, 
                                              exact_match=True)
        if replay_fns:
            print("Train task {} with shared experience".format(agent.task_no + 1))
            trainer.agent.clear_prev_buffer()
            if len(replay_fns) > 1:
                print("Line 89: Redundant buffers for the same task: {}".format(replay_fns))
                replay_fns = replay_fns[:1]
            for fn in replay_fns:
                trainer.agent.load_membuf(os.path.join(trainer.membufdir, fn), load_prev_buf=True)
            print("Loading complete")
            train_starttime = time.time()
            trainer.train(use_membuf=config.use_membuf, learn_new_env=False) # learn_new_env=False to use experience loaded to prev_buffer
            train_task_duration = time.time() - train_starttime
            print("Duration of training from shared - task {}: {}".format(agent.task_no + 1, train_task_duration))
        else:
            train_starttime = time.time()
            trainer.train(use_membuf=config.use_membuf, learn_new_env=True)
            train_task_duration = time.time() - train_starttime
            print("Duration of training from scratch - task {}: {}".format(agent.task_no + 1, train_task_duration))
    else:
        train_starttime = time.time()
        trainer.train(use_membuf=config.use_membuf, learn_new_env=True)
        train_task_duration = time.time() - train_starttime
        print("Duration of training from scratch - task {}: {}".format(agent.task_no + 1, train_task_duration))
    trainer.agent.save_model(trainer.outputdir, 'agent_{}_task_{}'.format(agent_id, trainer.agent.task_no))
    agent.task_no += 1


def train_per_agent(config, agent, task_id, agent_id, send_socket=None, 
                    receive_socket=None, connection=None, 
                    atari_client=None):
    """
    Share files with other agents and receive every other agent's buffer. 
    Reuse a buffer if it matches what this agent is about to learn.
    """
    sim_agent = DQNAgentTypeThree(config, use_simnet=True)
    #pdb.set_trace()
    atari_client.atari_create_env(agent.task_sequences[agent.task_no])
    trainer = Trainer(agent, 
                      atari_client, # train_env, 
                      config, 
                      sim_agent=sim_agent, 
                      sim_env=sim_env,
                      agent_id=agent_id,
                      task_no=task_id,
                      connection=connection,
                      s_sock=send_socket,
                      r_sock=receive_socket,
                      atari_client=atari_client)
    if config.use_simnet:
        print("Training simnet")
        trainer.train(use_membuf=config.use_membuf, use_simnet=config.use_simnet)
        print("Simnet trained")
    print("Training task agent for task {}".format(agent.task_no + 1))
    train_starttime = time.time()
    print("Line 159")
    trainer.train(use_membuf=config.use_membuf)
    train_task_duration = time.time() - train_starttime
    print("Duration of training task agent for task {}: {}".format(agent.task_no + 1, train_task_duration))
    trainer.agent.save_model(trainer.outputdir, 'train_self_agent_{}_task_{}'.format(agent_id, agent.task_no + 1))
    if config.share_info and trainer.agent.task_no > 0:
        print("Training on neighbors' information")
        if config.use_simnet:
            print("Using simNet for finding homogeneous tasks")
            target_weight = trainer.outputdir.rsplit('/', 1)[-1]
            closest_candidates = compare_weight_similarity(target_weight, 
                                                            trainer.membufdir, 
                                                            num_inputs=128, 
                                                            actions_dim=18, 
                                                            ignore_current_round=True,
                                                            task_no=agent.task_no,
                                                            n_closest=config.num_nearest_neighbors)
            replay_fns = find_replay_buffer(closest_candidates, trainer.membufdir)
        else: # retrieve all replay buffers
            all_replay_fns = [fn for fn in os.listdir(trainer.membufdir) if fn.startswith("membuf_")]
            our_task = trainer.agent.task_sequences[trainer.agent.task_no].split("_")[0]
          
            # turn exact_match on to find only identical matches of current tasks from other neighbors
            replay_fns = find_replay_buffer_nosim(all_replay_fns, 
                                                  agent_id=agent_id, 
                                                  task_id=trainer.agent.task_no, 
                                                  task_name=our_task, 
                                                  exact_match=False)
        
        trainer.agent.clear_prev_buffer()
        print("Loading {} buffers from closest tasks".format(len(replay_fns)))
        if replay_fns:
            for fn in replay_fns:
                trainer.agent.load_membuf(os.path.join(trainer.membufdir, fn), load_prev_buf=True)
            print("Loading complete")
            print("Start training from neighbors with replay")
            trainer.train(use_membuf=True, learn_new_env=False) # learn_new_env=False to use experience loaded to prev_buffer
            trainer.agent.save_model(trainer.outputdir, 'replay_agent_{}_task_{}'.format(agent_id, trainer.agent.task_no))
        print("Training complete")
    #pdb.set_trace()
    agent.task_no += 1


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env', default='Riverraid-ram-v0', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    parser.add_argument('--cl_retrain', dest='cl_retrain', action='store_true', help='cl_retrain model')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    parser.add_argument('--fisher_path', type=str, help='if test or retrain with EWC, import the fisher matrix')
    parser.add_argument('--learning_rate1', type=float, default=1e-5, help='learning rate of first task')
    parser.add_argument('--epsilon_decay_steps', type=int, default=30000, help='number of steps to decay exploration rate')
    parser.add_argument('--add_noop', dest='add_noop', action='store_true', help='apply NOOP when initializing environment')
    parser.add_argument('--use_frame_skip', dest='use_frame_skip', action='store_true', help='apply frame skipping when initializing environment')
    parser.add_argument('--num_frame_skip', type=int, default=4, help='number of frames to skip')
    parser.add_argument('--num_uniform_sampling', type=int, default=10000, help='number of uniform sampling of sequential memory chunks to perform when estimating fisher matrix')
    parser.add_argument('--agent_id', type=int, default=1, help='agent id for multi-agent setting')
    parser.add_argument('--task_no', type=int, default=0, help='which task in the sequence')
    parser.add_argument('--frames', type=int, default=100000, help='number of frames to run')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')

    # memory replay buffer
    parser.add_argument('--use_membuf', dest='use_membuf', action='store_true', help='load additional memory buffer')
    parser.add_argument('--membuf_loadpath', type=str, help='path to load memory replay buffer')
    parser.add_argument('--prev_membuf_loadpath', type=str, help='path to load the memory replay buffer from a task already learned from another agent')
    parser.add_argument('--membuf_parent_savedir', type=str, default='membuf', help='path to save memory replay buffers together')
    parser.add_argument('--membuf_savedir', type=str, default='foo_loc', help='path to save memory replay buffers together')
    parser.add_argument('--use_sample_thres', dest='use_sample_thres', action='store_true', help='apply sampling threshold to bias towards samples of current task during training')
    parser.add_argument('--sample_thres', type=float, default=0.9, help='threshold to sample only from current tasks, rather than the entire memory buffer')
    parser.add_argument('--learn_new_env', dest='learn_new_env', action='store_true', help='Turn on to use learn new samples from game environment')
    
    # low-switching cost (LSC) for memory replay buffer
    parser.add_argument('--use_lsc_membuf', dest='use_lsc_membuf', action='store_true', help='turn on to use LSC in memory buffer')
    parser.add_argument('--num_frames_save_buf', type=int, default=2000, help='number of frames to save shared memory buffer')
    parser.add_argument('--num_frames_load_buf', type=int, default=5000, help='number of frames to load shared memory buffer')
    
    # EWC
    parser.add_argument('--apply_ewc', dest='apply_ewc', action='store_true', help='run EWC and a second task')
    parser.add_argument('--lambda_value', type=float, default=100, help='learning rate of first task')
    parser.add_argument('--alpha', type=float, default=0.5, help='how much knowledge of previous fisher matrix to preserve')
    
    # communication protocol
    parser.add_argument('--simnet_weight_dir', type=str, default='simnet', help='location to save simnet parameter file')
    parser.add_argument('--apply_random_action', dest='apply_random_action', action='store_true', help='add random action wrapper to environment')
    parser.add_argument('--use_simnet', dest='use_simnet', action='store_true', help='set to True to use simNet to perform task similarity comparison')
    parser.add_argument('--test_simagent', dest='test_simagent', action='store_true', help='set to True to test sim_agent in test mode')
    parser.add_argument('--num_nearest_neighbors', type=int, default=1, help='which task in the sequence')
    parser.add_argument('--share_info', dest='share_info', action='store_true', help='set to True to allow sharing of information among agents')
    parser.add_argument('--port_one_two', type=int, default=1500, help='port number for connecting agent 1 and agent 2. Use if need be.')
    parser.add_argument('--port_one_three', type=int, default=1501, help='port number for connecting agent 1 and agent 3. Use if need be.')
    parser.add_argument('--port_one_four', type=int, default=1502, help='port number for connecting agent 1 and agent 4. Use if need be.')
    parser.add_argument('--port_one_five', type=int, default=1503, help='port number for connecting agent 1 and agent 5. Use if need be.')
    parser.add_argument('--port_two_three', type=int, default=1504, help='port number for connecting agent 2 and agent 3. Use if need be.')
    parser.add_argument('--port_two_four', type=int, default=1505, help='port number for connecting agent 2 and agent 4. Use if need be.')
    parser.add_argument('--port_two_five', type=int, default=1506, help='port number for connecting agent 2 and agent 5. Use if need be.')
    parser.add_argument('--port_three_four', type=int, default=1507, help='port number for connecting agent 3 and agent 4. Use if need be.')
    parser.add_argument('--port_three_five', type=int, default=1508, help='port number for connecting agent 3 and agent 5. Use if need be.')
    parser.add_argument('--port_four_five', type=int, default=1509, help='port number for connecting agent 3 and agent 1. Use if need be.')
    parser.add_argument('--host_one', type=str, default="localhost", help='IP of agent 1, used for responding to agent 1 from agent 2')
    parser.add_argument('--host_two', type=str, default="localhost", help='IP of agent 2, used for responding to agent 2 from agent 3')
    parser.add_argument('--host_three', type=str, default="localhost", help='IP of agent 3, used for responding to agent 3 from agent 4')
    parser.add_argument('--host_four', type=str, default="localhost", help='IP of agent 4, used for responding to agent 4 from agent 5')
    parser.add_argument('--host_five', type=str, default="localhost", help='IP of agent 4, used for responding to agent 5 from agent 1')
    parser.add_argument('--send_first', dest='send_first', action='store_true', help='set to True to send local membuf first then receive from others, set to false vice versa')
    parser.add_argument('--send_second', dest='send_second', action='store_true', help='set to True to send local membuf second then receive from others, set to false vice versa')
    parser.add_argument('--send_third', dest='send_third', action='store_true', help='set to True to send local membuf third then receive from others, set to false vice versa')
    parser.add_argument('--send_fourth', dest='send_fourth', action='store_true', help='set to True to send local membuf fourth then receive from others, set to false vice versa')
    parser.add_argument('--send_fifth', dest='send_fifth', action='store_true', help='set to True to send local membuf fifth then receive from others, set to false vice versa')
    parser.add_argument('--send_sixth', dest='send_sixth', action='store_true', help='set to True to send local membuf sixth then receive from others, set to false vice versa')
    parser.add_argument('--send_seventh', dest='send_seventh', action='store_true', help='set to True to send local membuf seventh then receive from others, set to false vice versa')
    parser.add_argument('--default_comm_logic', dest='default_comm_logic', action='store_true', help='set to True to use default training protocol in milestone 2')
    parser.add_argument('--open_atari_server', dest='open_atari_server', action='store_true', help='dummy argument, set to True to run remote Atari server')
    parser.add_argument('--atari_server_hostname', dest='atari_server_hostname', type=str, default="10.161.159.142", help='IP of Atari server')
    parser.add_argument('--atari_server_port', type=int, default=3333, help='port number to connect to remote Atari server')

    
    args = parser.parse_args()
    config = Config()
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.01
    config.eps_decay = args.epsilon_decay_steps # 30000
    config.frames = args.frames # 100000 # 300000 # 3000000 # 130000
    config.use_cuda = False # True
    config.learning_rate = args.learning_rate1 # 1e-5 works for RoadRunnerNoFrameskip-v4 # 2e-4 # 1e-4
    config.max_buff = 10000000
    config.update_tar_interval = 1000
    config.batch_size = args.batch_size # 128
    config.num_nearest_neighbors = args.num_nearest_neighbors
    config.num_tasks = 10 # 2 # 6
    config.print_interval = 10000
    config.log_interval = 10000
    config.checkpoint = True
    config.checkpoint_interval = 100000
    config.win_reward = 10000 # 7100 # 80 # 2000 # 700 # 100 # 250 # 1000 # RoadRunner-ram-v0
    config.win_break = True
    # config.env_two = args.env_two # "Boxing-ram-v0"
    config.apply_ewc_flag = True if args.apply_ewc else False
    config.lambda_value = args.lambda_value # 10000000000 # 100000000000000000000000 # 10000000000000000000000000000 # 200000
    config.continue_learning = False
    config.num_uniform_sampling = args.num_uniform_sampling # max(config.max_buff // config.batch_size, args.num_uniform_sampling) 
    config.add_noop = True if args.add_noop else False
    config.use_frame_skip = True if args.use_frame_skip else False
    config.num_frame_skip = args.num_frame_skip
    config.alpha = args.alpha
    config.use_membuf = True if args.use_membuf else False
    config.membuf_loadpath = args.membuf_loadpath
    config.prev_membuf_loadpath = args.prev_membuf_loadpath
    config.agent_id = args.agent_id
    config.membuf_parent_savedir = args.membuf_parent_savedir
    config.membuf_savedir = args.membuf_savedir
    config.task_no = args.task_no
    config.learn_new_env = True if args.learn_new_env else False
    config.apply_sample_thres = True if args.use_sample_thres else False
    config.sample_thres = args.sample_thres
    config.apply_lsc_membuf = True if args.use_lsc_membuf else False
    config.num_frames_save_buf = args.num_frames_save_buf
    config.num_frames_load_buf = args.num_frames_load_buf
    config.simnet_weight_dir = args.simnet_weight_dir
    config.add_random_action = True if args.apply_random_action else False
    config.test_simagent = True if args.test_simagent else False
    config.use_simnet = True if args.use_simnet else False
    config.share_info = True if args.share_info else False
    config.default_comm_logic = True if args.default_comm_logic else False
    config.send_first = True if args.send_first else False
    config.send_second = True if args.send_second else False
    config.send_third = True if args.send_third else False
    config.send_fourth = True if args.send_fourth else False
    config.send_fifth = True if args.send_fifth else False
    config.send_sixth = True if args.send_sixth else False
    config.send_seventh = True if args.send_seventh else False
    #
    config.port_one_two = args.port_one_two
    config.port_one_three = args.port_one_three
    config.port_one_four = args.port_one_four
    config.port_one_five = args.port_one_five
    config.port_one_six = args.port_one_six
    config.port_one_seven = args.port_one_seven
    #
    config.port_two_three = args.port_two_three
    config.port_two_four = args.port_two_four
    config.port_two_five = args.port_two_five
    config.port_two_six = args.port_two_six
    config.port_two_seven = args.port_two_seven
    #
    config.port_three_four = args.port_three_four
    config.port_three_five = args.port_three_five
    config.port_three_six = args.port_three_six
    config.port_three_seven = args.port_three_seven
    #
    config.port_four_five = args.port_four_five
    config.port_four_six = args.port_four_six
    config.port_four_seven = args.port_four_seven
    #
    config.port_five_six = args.port_five_six
    config.port_five_seven = args.port_five_seven
    #
    config.port_six_seven = args.port_six_seven
    
    
    config.host_one = args.host_one
    config.host_two = args.host_two
    config.host_three = args.host_three
    config.host_four = args.host_four
    config.host_five = args.host_five
    config.host_six = args.host_six
    config.host_seven = args.host_seven
    
    config.open_atari_server = True if args.open_atari_server else False
    config.atari_server_hostname = args.atari_server_hostname
    config.atari_server_port = args.atari_server_port
    
    # handle the atari env
    config.action_dim = 18 # orig_envs_one["train"].action_space.n # 18
    config.state_dim = 128 # orig_envs_one["train"].observation_space.shape[0] # 128
    
    if config.send_first:
        # task_sequences_one = ["RoadRunner-ram-v0_fs4", "Riverraid-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Krull-ram-v0_fs4", "Robotank-ram-v0_fs4", "Jamesbond-ram-v0_fs4"]
        task_sequences_one = ["RoadRunner-ram-v0_fs4", "Riverraid-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Krull-ram-v0_fs4", "Robotank-ram-v0_fs4", "Jamesbond-ram-v0_fs4", "Alien-ram-v0_fs4", "Kangaroo-ram-v0_fs4", "Frostbite-ram-v0_fs4", "BattleZone-ram-v0_fs4"]
    if config.send_second:
        # task_sequences_one = ["Riverraid-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Krull-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Jamesbond-ram-v0_fs4", "Robotank-ram-v0_fs4"]
        task_sequences_one = ["Krull-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Frostbite-ram-v0_fs4", "BattleZone-ram-v0_fs4", "Jamesbond-ram-v0_fs4", "Alien-ram-v0_fs4", "Kangaroo-ram-v0_fs4", "Robotank-ram-v0_fs4", "Riverraid-ram-v0_fs4", "RoadRunner-ram-v0_fs4"]
    if config.send_third:
        # task_sequences_one = ["Krull-ram-v0_fs4", "Riverraid-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Robotank-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Jamesbond-ram-v0_fs4"]
        task_sequences_one = ["Jamesbond-ram-v0_fs4", "Robotank-ram-v0_fs4", "Frostbite-ram-v0_fs4", "Krull-ram-v0_fs4", "Alien-ram-v0_fs4", "Kangaroo-ram-v0_fs4", "BattleZone-ram-v0_fs4", "Riverraid-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "StarGunner-ram-v0_fs4"]
    if config.send_fourth:
        task_sequences_one = ["Alien-ram-v0_fs4", "Kangaroo-ram-v0_fs4", "BattleZone-ram-v0_fs4", "Robotank-ram-v0_fs4", "Frostbite-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Krull-ram-v0_fs4", "Riverraid-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Jamesbond-ram-v0_fs4"]
    if config.send_fifth:
        task_sequences_one = ["Frostbite-ram-v0_fs4", "BattleZone-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Riverraid-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Krull-ram-v0_fs4", "Alien-ram-v0_fs4", "Kangaroo-ram-v0_fs4", "Jamesbond-ram-v0_fs4", "Robotank-ram-v0_fs4"]
    if config.send_sixth:
        task_sequences_one = ["BattleZone-ram-v0_fs4", "Jamesbond-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Krull-ram-v0_fs4", "Robotank-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Riverraid-ram-v0_fs4", "Kangaroo-ram-v0_fs4", "Alien-ram-v0_fs4", "Frostbite-ram-v0_fs4"]
    if config.send_seventh:
        task_sequences_one = ["StarGunner-ram-v0_fs4","Krull-ram-v0_fs4", "Alien-ram-v0_fs4", "BattleZone-ram-v0_fs4", "Robotank-ram-v0_fs4", "Frostbite-ram-v0_fs4",   "Kangaroo-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Jamesbond-ram-v0_fs4","Riverraid-ram-v0_fs4" ]
        
    agent_one = DQNAgentTypeThree(config, task_sequences=task_sequences_one)
    training_agents = [agent_one]

    atari_env_client = None
    send_sock_1_2 = send_sock_1_3 = send_sock_1_4 = send_sock_1_5 = send_sock_1_6 = send_sock_1_7 = send_sock_2_3 = send_sock_2_4 = send_sock_2_5 = send_sock_2_6= send_sock_2_7 = send_sock_3_4 = send_sock_3_5 = send_sock_3_6 = send_sock_3_7 = send_sock_4_5 = send_sock_4_6 = send_sock_4_7 = send_sock_5_6 = send_sock_5_7 = send_sock_6_7 =  None
    recv_sock_one_two = recv_sock_one_three = recv_sock_one_four = recv_sock_one_five = recv_sock_one_six = recv_sock_one_seven =  recv_sock_two_three = recv_sock_two_four = recv_sock_two_five = recv_sock_two_sock = recv_sock_two_seven =  recv_sock_three_four = recv_sock_three_five = recv_sock_three_six = recv_sock_three_seven =  recv_sock_four_five = recv_sock_four_six = recv_sock_four_seven = recv_sock_five_six = recv_sock_five_seven = recv_sock_six_seven = None
    connection_1_2 = connection_1_3 = connection_1_4 = connection_1_5 = connection_1_6 = connection_1_7 = connection_2_3 = connection_2_4 = connection_2_5 = connection_2_6 = connection_2_7 =  connection_3_4 = connection_3_5 = connection_3_6 = connection_3_7 =  connection_4_5 = connection_4_6 = connection_4_7 = connection_5_6 = connection_5_7 = connection_6_7 = None
    send_sock_dict = {}
    recv_sock_dict = {}
    connection_dict = {}
    start_time = time.time()
    print("Connecting to remote Atari remote server")
    if config.open_atari_server:
        atari_env_client = AtariEnvClient(PORT=config.atari_server_port, Atari_HOST=config.atari_server_hostname, agent_id=config.agent_id) # 3333)
        atari_env_client.atari_client_connect()
    print("Duration of establishing Atari remote server: {}".format(time.time() - start_time))
    if config.share_info:
        print("Establishing connections between server and clients for four-agent systems")
        # pdb.set_trace()
        if config.send_first:
            connection_1_2, send_sock_1_2 = server_connect(PORT=config.port_one_two,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            connection_1_3, send_sock_1_3 = server_connect(PORT=config.port_one_three,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            connection_1_4, send_sock_1_4 = server_connect(PORT=config.port_one_four,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            connection_1_5, send_sock_1_5 = server_connect(PORT=config.port_one_five,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
        
            connection_1_6, send_sock_1_6 = server_connect(PORT = config.port_one_six,
                                                           HOST = "0.0.0.0",
                                                           num_listeners = 6)
                                                           
            connection_1_7, send_sock_1_7 = server_connect(PORT = config.port_one_seven,
                                                           HOST = "0.0.0.0",
                                                           num_listeners = 6)
                                                           
            
            wait_execution(duration=14)

        if config.send_second:
            wait_execution(duration=7)
            print("Line 448")
            recv_sock_one_two = client_connect(PORT=config.port_one_two, HOST=config.host_one)
            print("Line 450")
            connection_2_3, send_sock_2_3 = server_connect(PORT=config.port_two_three,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            print("Line 454")
            connection_2_4, send_sock_2_4 = server_connect(PORT=config.port_two_four,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            print("Line 458")
            connection_2_5, send_sock_2_5 = server_connect(PORT=config.port_two_five,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            print("Line 462")
            connection_2_6, send_sock_2_6 = server_connect(PORT=config.port_two_six,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            connection_2_7, send_sock_2_7 = server_connect(PORT=config.port_two_seven,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)

        if config.send_third:
            wait_execution(duration=7)
            print("Line 487")
            recv_sock_one_three = client_connect(PORT=config.port_one_three, HOST=config.host_one)
            wait_execution(duration=7)
            print("Line 490")
            recv_sock_two_three = client_connect(PORT=config.port_two_three, HOST=config.host_two)
            print("Line 492")
            connection_3_4, send_sock_3_4 = server_connect(PORT=config.port_three_four,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            print("Line 496")
            connection_3_5, send_sock_3_5 = server_connect(PORT=config.port_three_five,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            print("Line 500")
            connection_3_6, send_sock_3_6 = server_connect(PORT=config.port_three_six,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            print("Line 500")
            connection_3_7, send_sock_3_7 = server_connect(PORT=config.port_three_seven,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            
        if config.send_fourth:
            wait_execution(duration=7)
            print("Line 524")
            recv_sock_one_four = client_connect(PORT=config.port_one_four, HOST=config.host_one)
            wait_execution(duration=7)
            print("Line 527")
            recv_sock_two_four = client_connect(PORT=config.port_two_four, HOST=config.host_two)
            wait_execution(duration=7)
            print("Line 530")
            recv_sock_three_four = client_connect(PORT=config.port_three_four, HOST=config.host_three)
            print("Line 532")
            connection_4_5, send_sock_4_5 = server_connect(PORT=config.port_four_five,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            connection_4_6, send_sock_4_6 = server_connect(PORT=config.port_four_six,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            connection_4_7, send_sock_4_7 = server_connect(PORT=config.port_four_seven,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)

        if config.send_fifth:
            wait_execution(duration=7)
            print("Line 553")
            recv_sock_one_five = client_connect(PORT=config.port_one_five, HOST=config.host_one)
            wait_execution(duration=7)
            print("Line 556")
            recv_sock_two_five = client_connect(PORT=config.port_two_five, HOST=config.host_two)
            wait_execution(duration=7)
            print("Line 559")
            recv_sock_three_five = client_connect(PORT=config.port_three_five, HOST=config.host_three)
            wait_execution(duration=7)
            print("Line 562")
            recv_sock_four_five = client_connect(PORT=config.port_four_five, HOST=config.host_four)
            wait_execution(duration=7)
            
            connection_5_6, send_sock_5_6 = server_connect(PORT=config.port_five_six,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
            connection_5_7, send_sock_5_7 = server_connect(PORT=config.port_five_seven,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
                                                           
        if config.send_sixth:
            wait_execution(duration=7)
            print("Line 553")
            recv_sock_one_six = client_connect(PORT=config.port_one_six, HOST=config.host_one)
            wait_execution(duration=7)
            print("Line 556")
            recv_sock_two_six = client_connect(PORT=config.port_two_six, HOST=config.host_two)
            wait_execution(duration=7)
            print("Line 559")
            recv_sock_three_six = client_connect(PORT=config.port_three_six, HOST=config.host_three)
            wait_execution(duration=7)
            print("Line 562")
            recv_sock_four_six = client_connect(PORT=config.port_four_six, HOST=config.host_four)
            wait_execution(duration=7)
            recv_sock_five_six= client_connect(PORT=config.port_five_six, HOST=config.host_five)
            wait_execution(duration=7)
            
            connection_6_7, send_sock_6_7 = server_connect(PORT=config.port_six_seven,
                                                           HOST="0.0.0.0",
                                                           num_listeners=6)
                                                           
        if config.send_seventh:
            wait_execution(duration=42)
            print("Line 553")
            recv_sock_one_seven = client_connect(PORT=config.port_one_seven, HOST=config.host_one)
            wait_execution(duration=7)
            print("Line 556")
            recv_sock_two_seven = client_connect(PORT=config.port_two_seven, HOST=config.host_two)
            wait_execution(duration=7)
            print("Line 559")
            recv_sock_three_seven = client_connect(PORT=config.port_three_seven, HOST=config.host_three)
            wait_execution(duration=7)
            print("Line 562")
            recv_sock_four_seven = client_connect(PORT=config.port_four_seven, HOST=config.host_four)
            wait_execution(duration=7)
            recv_sock_five_seven= client_connect(PORT=config.port_five_seven, HOST=config.host_five)
            wait_execution(duration=7)
            recv_sock_six_seven= client_connect(PORT=config.port_six_seven, HOST=config.host_six)
            wait_execution(duration=7)
            
            
 
        send_sock_dict = {
            "send_sock_1_2": send_sock_1_2, "send_sock_1_3": send_sock_1_3, "send_sock_1_4": send_sock_1_4, "send_sock_1_5": send_sock_1_5, "send_sock_1_6": send_sock_1_6 , "send_sock_1_7": send_sock_1_7 , "send_sock_2_3": send_sock_2_3,
            "send_sock_2_4": send_sock_2_4, "send_sock_2_5": send_sock_2_5, "send_sock_2_6": send_sock_2_6, "send_sock_2_7": send_sock_2_7 , "send_sock_3_4": send_sock_3_4, "send_sock_3_5": send_sock_3_5,"send_sock_3_6": send_sock_3_6, "send_sock_3_7": send_sock_3_7 , "send_sock_4_5": send_sock_4_5,
                        "send_sock_4_6": send_sock_4_6, "send_sock_4_7": send_sock_4_7, "send_sock_5_6": send_sock_5_6, "send_sock_5_7": send_sock_5_7,
                                                "send_sock_6_7": send_sock_6_7
        }
        recv_sock_dict = {
            "recv_sock_one_two": recv_sock_one_two, "recv_sock_one_three": recv_sock_one_three, "recv_sock_one_four": recv_sock_one_four, "recv_sock_one_five": recv_sock_one_five,  "recv_sock_one_six": recv_sock_one_six, "recv_sock_one_seven": recv_sock_one_seven,
            "recv_sock_two_three": recv_sock_two_three, "recv_sock_two_four": recv_sock_two_four, "recv_sock_two_five": recv_sock_two_five,
                        "recv_sock_two_six": recv_sock_two_six, "recv_sock_two_seven": recv_sock_two_seven,
            "recv_sock_three_four": recv_sock_three_four, "recv_sock_three_five": recv_sock_three_five,"recv_sock_three_six": recv_sock_three_six,
                        "recv_sock_three_seven": recv_sock_three_seven, "recv_sock_four_five": recv_sock_four_five,"recv_sock_four_six": recv_sock_four_six,
                                                "recv_sock_four_seven": recv_sock_four_seven,"recv_sock_five_six": recv_sock_five_six,"recv_sock_five_seven": recv_sock_five_seven, "recv_sock_six_seven":recv_sock_six_seven
        }
        connection_dict = {
            "connection_1_2": connection_1_2, "connection_1_3": connection_1_3, "connection_1_4": connection_1_4, "connection_1_5": connection_1_5,
            "connection_1_6": connection_1_6, "connection_1_7": connection_1_7,
            "connection_2_3": connection_2_3, "connection_2_4": connection_2_4, "connection_2_5": connection_2_5,"connection_2_6": connection_2_6,
            "connection_2_7": connection_2_7,
            "connection_3_4": connection_3_4,"connection_3_5": connection_3_5, "connection_3_6": connection_3_6, "connection_3_7": connection_3_7,
            "connection_4_5": connection_4_5,"connection_4_6": connection_4_6, "connection_4_7": connection_4_7,
            "connection_5_6": connection_5_6,"connection_5_7": connection_5_7,
            "connection_6_7": connection_6_7
        }

        print("Connections established")
        print("Duration of establishing connections: {}".format(time.time() - start_time))
    else:
        print("Train agents on {} tasks without communication".format(config.num_tasks))
    training_func_starttime = time.time()
    training(config, training_agents, send_socket=send_sock_dict, 
             receive_socket=recv_sock_dict, connection=connection_dict, 
             default_logic=config.default_comm_logic, 
             atari_client=atari_env_client)
    training_func_duration = time.time() - training_func_starttime
    print("Duration of training(): {}".format(training_func_duration))
    
    if config.share_info:
        disconnect_starttime = time.time()
        if config.send_first:
            server_disconnect(connection_1_2, send_sock_1_2)
            server_disconnect(connection_1_3, send_sock_1_3)
            server_disconnect(connection_1_4, send_sock_1_4)
            server_disconnect(connection_1_5, send_sock_1_5)
            server_disconnect(connection_1_6, send_sock_1_6)
            server_disconnect(connection_1_7, send_sock_1_7)
        if config.send_second:
            client_disconnect(recv_sock_one_two)
            server_disconnect(connection_2_3, send_sock_2_3)
            server_disconnect(connection_2_4, send_sock_2_4)
            server_disconnect(connection_2_5, send_sock_2_5)
            server_disconnect(connection_2_6, send_sock_2_6)
            server_disconnect(connection_2_7, send_sock_2_7)
        if config.send_third:
            client_disconnect(recv_sock_one_three)
            client_disconnect(recv_sock_two_three)
            server_disconnect(connection_3_4, send_sock_3_4)
            server_disconnect(connection_3_5, send_sock_3_5)
            server_disconnect(connection_3_6, send_sock_3_6)
            server_disconnect(connection_3_7, send_sock_3_7)
        if config.send_fourth:
            client_disconnect(recv_sock_one_four)
            client_disconnect(recv_sock_two_four)
            client_disconnect(recv_sock_three_four)
            server_disconnect(connection_4_5, send_sock_4_5)
            server_disconnect(connection_4_6, send_sock_4_6)
            server_disconnect(connection_4_7, send_sock_4_7)
        if config.send_fifth:
            client_disconnect(recv_sock_one_five)
            client_disconnect(recv_sock_two_five)
            client_disconnect(recv_sock_three_five)
            client_disconnect(recv_sock_four_five)
            server_disconnect(connection_5_6, send_sock_5_6)
            server_disconnect(connection_5_7, send_sock_5_7)
        if config.send_sixth:
            client_disconnect(recv_sock_one_six)
            client_disconnect(recv_sock_two_six)
            client_disconnect(recv_sock_three_six)
            client_disconnect(recv_sock_four_six)
            client_disconnect(recv_sock_five_six)
            server_disconnect(connection_6_7, send_sock_6_7)
        if config.send_seventh:
            client_disconnect(recv_sock_one_seven)
            client_disconnect(recv_sock_two_seven)
            client_disconnect(recv_sock_three_seven)
            client_disconnect(recv_sock_four_seven)
            client_disconnect(recv_sock_five_seven)
            client_disconnect(recv_sock_six_seven)
    
        print("Duration of closing connections: {}".format(time.time() - disconnect_starttime))
    total_time = time.time() - start_time

    # disconnect Atari server and client
    if config.open_atari_server:
        atari_env_client.atari_disconnect()
    print("Atari remote server and client closed")
    print("Duration of the entire learning process: {}".format(total_time))

