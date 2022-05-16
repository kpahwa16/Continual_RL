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
from task_similarity import compare_weight_similarity, find_replay_buffer, find_replay_buffer_nosim


def training(config, agents, default_logic=True, send_socket=None, receive_socket=None, connection=None):
    for i in range(config.num_tasks):
        if default_logic:
            train_one_round(config, agents, i, train_func=train_per_agent, 
                            send_socket=send_socket, 
                            receive_socket=receive_socket, 
                            connection=connection)
        else:
            # pdb.set_trace()
            train_one_round(config, agents, i, train_func=train_per_agent_logic2, 
                            send_socket=send_socket, 
                            receive_socket=receive_socket, 
                            connection=connection)
        

def train_one_round(config, agents, epoch_id, train_func=None, send_socket=None, 
                    receive_socket=None, connection=None):
    for j, agent in enumerate(agents):
        # train_per_agent(config, agent, epoch_id, j + 1)
        print("Training agent {} on task {}".format(config.agent_id, epoch_id + 1))
        round_starttime = time.time()
        train_func(config, agent, epoch_id, config.agent_id, send_socket=send_socket, 
                        receive_socket=receive_socket, connection=connection)
        round_endtime = time.time()
        print("Time for agent {} to complete round {}: {}".format(config.agent_id, epoch_id + 1, round_endtime - round_starttime))
    
def train_per_agent_logic2(config, agent, task_id, agent_id, send_socket=None, 
                           receive_socket=None, connection=None):
    # this logic assumes an agent knows task identities of shared tasks
    # hence, this logic doesn't use SimNet to compare task similarity
    # assert config.share_info and send_socket and receive_socket and connection, "need to turn on config.share_info"
    # pdb.set_trace()
    # first share the name of experience buffer, if matched with current task name, use it, otherwise, learn new task
    
    env_name, spec = agent.task_sequences[agent.task_no].split("_")
    envs = _get_task_group_envs(env_name, variant=spec)
    train_env, _ = envs["train"], envs["sim"]
    trainer = Trainer(agent, 
                      train_env, 
                      config, 
                      sim_agent=None, 
                      sim_env=None,
                      agent_id=agent_id,
                      task_no=task_id,
                      default_logic=False, # set this to False to use second logic
                      connection=connection,
                      s_sock=send_socket,
                      r_sock=receive_socket)
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
            # pdb.set_trace()
            if len(replay_fns) > 1:
                print("Line 89: Redundant buffers for the same task: {}".format(replay_fns))
                replay_fns = replay_fns[:1]
            for fn in replay_fns:
                # pdb.set_trace()
                trainer.agent.load_membuf(os.path.join(trainer.membufdir, fn), load_prev_buf=True)
            # pdb.set_trace()
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
        # pdb.set_trace()
        train_starttime = time.time()
        # trainer.agent.load_membuf("/content/CANAL/out/multi-agent-1/membuf_Riverraid-ram-v0_agent-1-task-0.pkl", load_prev_buf=True)
        # trainer.train(use_membuf=config.use_membuf, learn_new_env=False)
        trainer.train(use_membuf=config.use_membuf, learn_new_env=True)
        train_task_duration = time.time() - train_starttime
        print("Duration of training from scratch - task {}: {}".format(agent.task_no + 1, train_task_duration))
        # pdb.set_trace()
    trainer.agent.save_model(trainer.outputdir, 'agent_{}_task_{}'.format(agent_id, trainer.agent.task_no))
    agent.task_no += 1


def train_per_agent(config, agent, task_id, agent_id, send_socket=None, 
                    receive_socket=None, connection=None):
    sim_agent = DQNAgentTypeThree(config, use_simnet=True)
    env_name, spec = agent.task_sequences[agent.task_no].split("_")
    envs = _get_task_group_envs(env_name, variant=spec)
    train_env, sim_env = envs["train"], envs["sim"]
    trainer = Trainer(agent, 
                      train_env, 
                      config, 
                      sim_agent=sim_agent, 
                      sim_env=sim_env,
                      agent_id=agent_id,
                      task_no=task_id,
                      connection=connection,
                      s_sock=send_socket,
                      r_sock=receive_socket)
    if config.use_simnet:
        print("Training simnet")
        trainer.train(use_membuf=config.use_membuf, use_simnet=config.use_simnet)
        print("Simnet trained")
    print("Training task agent for task {}".format(agent.task_no + 1))
    # pdb.set_trace()
    train_starttime = time.time()
    trainer.train(use_membuf=config.use_membuf)
    train_task_duration = time.time() - train_starttime
    print("Duration of training task agent for task {}: {}".format(agent.task_no + 1, train_task_duration))
    # pdb.set_trace()
    trainer.agent.save_model(trainer.outputdir, 'train_self_agent_{}_task_{}'.format(agent_id, agent.task_no + 1))
    
    # trainer.agent.save_model(trainer.membufdir, 'train_self_agent_{}_task_{}'.format(agent_id, agent.task_no + 1))
    # learn from neighbors
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
            # pdb.set_trace()
        # make sure prev_buffer is clear
        # pdb.set_trace()
        trainer.agent.clear_prev_buffer()
        print("Loading {} buffers from closest tasks".format(len(replay_fns)))
        if replay_fns:
            for fn in replay_fns:
                # pdb.set_trace()
                trainer.agent.load_membuf(os.path.join(trainer.membufdir, fn), load_prev_buf=True)
                # pdb.set_trace()
            # pdb.set_trace()
            print("Loading complete")
            print("Start training from neighbors with replay")
            trainer.train(use_membuf=True, learn_new_env=False) # learn_new_env=False to use experience loaded to prev_buffer
            # pdb.set_trace()
            trainer.agent.save_model(trainer.outputdir, 'replay_agent_{}_task_{}'.format(agent_id, trainer.agent.task_no))
            # trainer.agent.save_model(trainer.membufdir, 'replay_agent_{}_task_{}'.format(agent_id, trainer.agent.task_no))
        print("Training complete")
    agent.task_no += 1
    

def _get_task_env(env_name, 
                  use_noop=False, 
                  use_frame_skip=False, 
                  num_frame_skip=4, 
                  use_random_action=False, 
                  epsilon=0.5):
    env = gym.make(env_name)
    env.seed(0)
    if use_noop:
        env = add_noop(env, noopmax=30)
    if use_frame_skip:
        env = add_frame_skip(env, skip=num_frame_skip)
    if use_random_action:
        env = add_random_action(env, epsilon=epsilon)
    return env


def _get_agent_envs(env_name,
                    add_noop=False, 
                    add_frame_skip=False, 
                    num_frame_skip=4, 
                    add_random_action=False, 
                    epsilon=0.5):
    env_one = _get_task_env(env_name,
                            use_noop=add_noop, 
                            use_frame_skip=add_frame_skip, 
                            num_frame_skip=num_frame_skip, 
                            use_random_action=add_random_action, 
                            epsilon=epsilon)
    sim_env_one = _get_task_env(env_name,
                                use_noop=add_noop, 
                                use_frame_skip=add_frame_skip, 
                                num_frame_skip=num_frame_skip, 
                                use_random_action=add_random_action, 
                                epsilon=epsilon)
    return {
        "train": env_one,
        "sim": sim_env_one,
    }


def _get_task_group_envs(env_name, variant="orig"):
    if variant == "orig": 
        return _get_agent_envs(env_name)
    else:
        return _get_agent_envs(env_name,
                               add_noop=True, 
                               add_frame_skip=True, 
                               num_frame_skip=4, 
                               add_random_action=False, 
                               epsilon=0.5)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env', default='Riverraid-ram-v0', type=str, help='gym environment')
    # parser.add_argument('--env_two', default='Riverraid-ram-v0', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    parser.add_argument('--cl_retrain', dest='cl_retrain', action='store_true', help='cl_retrain model')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    parser.add_argument('--fisher_path', type=str, help='if test or retrain with EWC, import the fisher matrix')
    parser.add_argument('--learning_rate1', type=float, default=1e-5, help='learning rate of first task')
    # parser.add_argument('--learning_rate2', type=float, default=1e-5, help='learning rate of first task')
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
    parser.add_argument('--port_one', type=int, default=1500, help='port number for send_first agent in two-agent system to send files to the other agent')
    parser.add_argument('--port_two', type=int, default=1501, help='port number for the other agent in two-agent system to send files to the send_first agent')
    parser.add_argument('--send_first', dest='send_first', action='store_true', help='set to True to send local membuf first then receive from others, set to false vice versa')
    parser.add_argument('--default_comm_logic', dest='default_comm_logic', action='store_true', help='set to True to use default training protocol in milestone 2')
    
    args = parser.parse_args()
    config = Config()
    # config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.01
    config.eps_decay = args.epsilon_decay_steps # 30000
    config.frames = args.frames # 100000 # 300000 # 3000000 # 130000
    config.use_cuda = True
    config.learning_rate = args.learning_rate1 # 1e-5 works for RoadRunnerNoFrameskip-v4 # 2e-4 # 1e-4
    # config.learning_rate_two = args.learning_rate2
    config.max_buff = 10000000
    config.update_tar_interval = 1000
    config.batch_size = args.batch_size # 128
    config.num_nearest_neighbors = args.num_nearest_neighbors
    config.num_tasks = 2 # 2 # 6
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
    config.port_one = args.port_one
    config.port_two = args.port_two
    
    # handle the atari env
    config.action_dim = 18 # orig_envs_one["train"].action_space.n # 18
    config.state_dim = 128 # orig_envs_one["train"].observation_space.shape[0] # 128
    
    # first group of tasks: two variants per game group
    # task_sequences_one = ["RoadRunner-ram-v0_orig", "RoadRunner-ram-v0_fs4", "Riverraid-ram-v0_orig", "Riverraid-ram-v0_fs4", "Jamesbond-ram-v0_orig", "Jamesbond-ram-v0_fs4"]
    # task_sequences_two = ["RoadRunner-ram-v0_fs4", "RoadRunner-ram-v0_orig", "Riverraid-ram-v0_fs4", "Riverraid-ram-v0_orig", "Jamesbond-ram-v0_fs4", "Jamesbond-ram-v0_orig"] 
    # task_sequences_three = ["Jamesbond-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Riverraid-ram-v0_fs4", "Jamesbond-ram-v0_orig", "RoadRunner-ram-v0_orig", "Riverraid-ram-v0_orig"] 
    # task_sequences_four = ["RoadRunner-ram-v0_fs4", "Jamesbond-ram-v0_orig", "Jamesbond-ram-v0_fs4", "Jamesbond-ram-v0_orig", "RoadRunner-ram-v0_fs4", "Riverraid-ram-v0_orig"] 
    
    # second group of tasks: one variant per game group
    # task_sequences_one = ["RoadRunner-ram-v0_fs4", "Riverraid-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Robotank-ram-v0_fs4", "Krull-ram-v0_fs4", "Jamesbond-ram-v0_fs4"]
    # task_sequences_two = ["Riverraid-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Krull-ram-v0_fs4", "Jamesbond-ram-v0_fs4", "Robotank-ram-v0_fs4", "StarGunner-ram-v0_fs4"] 
    # task_sequences_three = ["Jamesbond-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Robotank-ram-v0_fs4", "Riverraid-ram-v0_fs4", "Krull-ram-v0_fs4", "StarGunner-ram-v0_fs4"] 
    # task_sequences_four = ["StarGunner-ram-v0_fs4", "Jamesbond-ram-v0_fs4", "Krull-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Robotank-ram-v0_fs4", "Riverraid-ram-v0_fs4"]
    
    # agent_one = DQNAgentTypeThree(config, task_sequences=task_sequences_one)
    # agent_two = DQNAgentTypeThree(config, task_sequences=task_sequences_two)
    # agent_three = DQNAgentTypeThree(config, task_sequences=task_sequences_three)
    # agent_four = DQNAgentTypeThree(config, task_sequences=task_sequences_four)
    # training_agents = [agent_one, agent_two, agent_three, agent_four]
    
    # task_sequences_one = ["RoadRunner-ram-v0_fs4", "Riverraid-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Krull-ram-v0_fs4"]
    # task_sequences_two = ["StarGunner-ram-v0_fs4", "Riverraid-ram-v0_fs4", "Krull-ram-v0_fs4", "RoadRunner-ram-v0_fs4"]
    if config.send_first:
        task_sequences_one = ["RoadRunner-ram-v0_fs4", "Riverraid-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Krull-ram-v0_fs4", "Robotank-ram-v0_fs4", "Jamesbond-ram-v0_fs4"]
    else:
        task_sequences_one = ["Riverraid-ram-v0_fs4", "RoadRunner-ram-v0_fs4", "Krull-ram-v0_fs4", "StarGunner-ram-v0_fs4", "Jamesbond-ram-v0_fs4", "Robotank-ram-v0_fs4"]
    agent_one = DQNAgentTypeThree(config, task_sequences=task_sequences_one)
    # agent_two = DQNAgentTypeThree(config, task_sequences=task_sequences_two)
    # training_agents = [agent_one, agent_two]
    training_agents = [agent_one]

    send_sock = recv_sock = connection= None
    start_time = time.time()
    if config.share_info:
        print("Establishing connections between server and clients for two-agent systems")
        if config.send_first:
            connection, send_sock = server_connect(PORT=config.port_one, # 1240,
                                      num_listeners=2)
            wait_execution(duration=7)
            recv_sock = client_connect(PORT=config.port_two) # 1241)
        else:
            wait_execution(duration=2)
            recv_sock = client_connect(PORT=config.port_one) # 1240)
            connection, send_sock = server_connect(PORT=config.port_two, # 1241,
                                      num_listeners=2)
            # wait_execution(duration=10)
        print("Connections established")
        print("Duration of establishing connections: {}".format(time.time() - start_time))
    else:
        print("Train agents on {} tasks without communication".format(config.num_tasks))
    
    training_func_starttime = time.time()
    training(config, training_agents, send_socket=send_sock, 
             receive_socket=recv_sock, connection=connection, 
             default_logic=config.default_comm_logic)
    training_func_duration = time.time() - training_func_starttime
    print("Duration of training(): {}".format(training_func_duration))
    
    if config.share_info:
        disconnect_starttime = time.time()
        # wait_execution(duration=30) # wait for all agents to complete their training sequences
        if config.send_first:
            client_disconnect(recv_sock)
            server_disconnect(connection, send_sock)
        else:
            server_disconnect(connection, send_sock)
            client_disconnect(recv_sock)
        print("Duration of closing connections: {}".format(time.time() - disconnect_starttime))
    total_time = time.time() - start_time
    print("Duration of the entire learning process: {}".format(total_time))

