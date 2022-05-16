import gym
import sys
import time
import socket
import pdb
from socket_scripts import *
from common.wrappers import *
from socket import *
from threading import Thread


def _get_task_env(env_name, 
                  use_noop=False, 
                  use_frame_skip=False, 
                  num_frame_skip=4, 
                  use_random_action=False, 
                  epsilon=0.5):
    """
    This function creates an Gym environment on the server end.
    """
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
    """
    This function creates one environment for main LL model and one for SimNet.
    """
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
    """
    This function modifies configurations of Atari environment according to suffix of env_name.
    """
    if variant == "orig": 
        return _get_agent_envs(env_name)
    else:
        return _get_agent_envs(env_name,
                               add_noop=True, 
                               add_frame_skip=True, 
                               num_frame_skip=4, 
                               add_random_action=False, 
                               epsilon=0.5)


class AtariEnvServer:
    """
    This class defines Atari Environment Server and utilities 
    for exchanging commands and data with CANAL agents.
    """
    def __init__(self, PORT=3333, HOST="0.0.0.0"):
        self.PORT = PORT
        self.HOST = HOST
        # not server_connect!
        self.sock = server_init(PORT=self.PORT, 
                                HOST=self.HOST)
        self.connection = None
        self.env = {}
        self.sim_env = {}
        self.curr_env = None
    
    def server_connect_util(self):
        self.connection, address = self.sock.accept()
        return self.connection, address

    def get_command(self):
        agent_id, function_request, argument = receive_seq(self.connection) # input is a tuple of (func_name, args)
        if function_request == "create_env":
            self.create_env_logic(agent_id, argument)
        elif function_request == "select_env":
            self.select_env_logic(argument)
        elif function_request == "reset":
            self.reset_logic(agent_id)
        elif function_request == "step":
            self.step_logic(agent_id, argument)
        elif function_request == "env_id":
            self.env_name_logic(agent_id)
        elif function_request == "disconnect":
            server_disconnect(self.connection, self.sock)
            print("Atari Env Server closed")
            return False
        else:
            raise ValueError
        return True

    def create_env_logic(self, client_id, task_sequence_info):
        env_name, spec = task_sequence_info.split("_")
        envs = _get_task_group_envs(env_name, variant=spec)
        self.env[client_id], self.sim_env[client_id] = envs["train"], envs["sim"]
        print("Atari Env established")
    
    def select_env_logic(self, argument):
        if argument == "sim":
            self.curr_env = self.sim_env
        else:
            self.curr_env = self.env
    
    def reset_logic(self, client_id):
        state = self.curr_env[client_id].reset()
        send_seq(self.connection, state)
    
    def step_logic(self, client_id, action): # action is an integer
        # next_state, reward, done, _ = self.curr_env.step(action)
        result_tuple = self.curr_env[client_id].step(action)
        send_seq(self.connection, result_tuple)
    
    def env_name_logic(self, client_id):
        env_name = self.env[client_id].unwrapped.spec.id
        send_seq(self.connection, env_name)


class ClientHandler(Thread):
    def __init__(self, client, address, atari_server):
        Thread.__init__(self)
        self._client = client
        self._address = address
        self.atari_server = atari_server
        self.atari_server.connection = self._client

    def run(self):
        signal = True
        while signal:
            signal = self.atari_server.get_command()
        print("Server down")


if __name__ == "__main__":
    # To provide a CANAL agent with data, use one process to run a Atari environment server.
    port = int(sys.argv[1]) # 3333
    host = "0.0.0.0"
    atari_server = AtariEnvServer(PORT=port, HOST=host)
    while True:
        print("Waiting for connection...")
        client, address = atari_server.sock.accept() # .server_connect_util()
        print('...client connected from: {}'.format(address))
        handler = ClientHandler(client, address, atari_server)
        handler.start()
