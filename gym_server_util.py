import socket
import pickle
import time
import pdb
from socket_scripts import *


class AtariEnvClient:
    """
    This class defines API and commands to use by CANAL agent 
    to communicate data and requests with Atari server.
    """
    def __init__(self, PORT=3333, Atari_HOST="0.0.0.0", agent_id=-1):
        self.PORT = PORT
        self.HOST = Atari_HOST
        self.sock = None
        self.agent_id = agent_id

    def atari_client_connect(self):
        self.sock = client_connect(PORT=self.PORT, HOST=self.HOST)
        print("Atari Client connected")

    def atari_disconnect(self):
        send_seq(self.sock, [self.agent_id, "disconnect", ""])
        time.sleep(1)
        client_disconnect(self.sock)

    def atari_reset(self):
        send_seq(self.sock, [self.agent_id, "reset", ""])
        time.sleep(3)
        return receive_seq(self.sock) 

    def atari_step(self, action):
        send_seq(self.sock, [self.agent_id, "step", action])
        step_tuple = receive_seq(self.sock)
        time.sleep(0.8)
        return step_tuple

    def atari_get_env_id(self):
        send_seq(self.sock, [self.agent_id, "env_id", ""])
        env_id = receive_seq(self.sock)
        time.sleep(1)
        return env_id

    def atari_create_env(self, task_sequence_info):
        send_seq(self.sock, [self.agent_id, "create_env", task_sequence_info])
    
    def atari_select_env(self, argument):
        send_seq(self.sock, [self.agent_id, "select_env", argument])
    
