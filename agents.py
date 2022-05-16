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
from torch import autograd
from torch.optim import Adam
from tester import Tester
from buffer import ReplayBuffer
from common.wrappers import add_noop, add_frame_skip, add_random_action
from core.util import get_class_attr_val
from model import CnnDQNTypeTwo, DQN, TinyDQN


class DQNAgentTypeThree:
    def __init__(self, config, task_sequences=[], use_simnet=False):
        """
        This class defines functionalities and attributes of a CANAL agent.
        
        Args:
            config: object of configuration class
            task_sequences: list of Atari task names
            use_simnet: set to True to train SimNet, used in train_per_agent()
        """
        self.config = config
        self.is_training = True
        self.buffer = ReplayBuffer(self.config.max_buff)

        if use_simnet:
            self.model = TinyDQN(self.config.state_dim, self.config.action_dim)
        else:
            self.model = DQN(self.config.state_dim, self.config.action_dim)
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        if self.config.use_cuda:
            self.model = self.model.cuda()
            self.cuda()
        self.params = {n.replace('.', '__') : p for n, p in self.model.nn.named_parameters() if p.requires_grad}
        self.layer_names = [name.replace('.', '__') for name, param in self.model.nn.named_parameters() if param.requires_grad]
        self.debug_task2_loss = []
        self.debug_ewc_loss = []
        self.alpha = config.alpha
        self.task_sequences = task_sequences
        self.i = 1
        self.task_no = 0
        
    def act(self, state, epsilon=None, apply_ewc=False):
        """
        This function is used for training DQN. It generates information of next state.
        """
        if epsilon is None: epsilon = self.config.epsilon_min
        if random.random() > epsilon or not self.is_training:
            state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
            if self.config.use_cuda:
                state = state.cuda()
            q_value = self.model.forward(state)
            action = q_value.max(1)[1].item()
        else:
            action = random.randrange(self.config.action_dim)
        return action

    def learning(self, fr, loss_fn=None, apply_ewc=False, focus_curr=False, use_lsc=False):
        """
        This function is used for updating DQN model.
        """
        if use_lsc:
            s0, a, r, s1, done = self.buffer.sample(self.config.batch_size, use_lsc=use_lsc)
        else: 
            s0, a, r, s1, done = self.buffer.sample(self.config.batch_size, focus_curr=focus_curr)

        s0 = torch.tensor(s0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.long)
        r = torch.tensor(r, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        if self.config.use_cuda:
            s0 = s0.cuda()
            s1 = s1.cuda()
            a = a.cuda()
            r = r.cuda()
            done = done.cuda()
            q_values = self.model(s0).cuda()
            next_q_values = self.model(s1).cuda()
        else:
            q_values = self.model(s0)
            next_q_values = self.model(s1)
        next_q_value = next_q_values.max(1)[0]
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        task_loss = (q_value - expected_q_value.detach()).pow(2).mean()
        loss = None
        if apply_ewc:
            ewc_loss = self.model.get_ewc_terms(lambda_value=self.config.lambda_value, fr=fr)
            loss = task_loss + ewc_loss
            if fr % 20000 == 0:
                print("[Debug]: loss = task_loss ({}) + ewc_loss ({}) = {}".format(task_loss, ewc_loss, loss))
                print("[Debug]: Task 2 Loss: {}".format(self.debug_task2_loss[-10:]))
                print("[Debug]: EWC Loss: {}".format(self.debug_ewc_loss[-10:]))
                self.debug_task2_loss.append(float(task_loss))
                self.debug_ewc_loss.append(float(ewc_loss))
        else:
            loss = task_loss
        self.model_optim.zero_grad()
        loss.backward()
        self.model_optim.step()
        return loss.item()

    def cuda(self):
        """
        Util
        """
        self.model.cuda()
        
    def load_weights(self, model_path):
        """
        Util
        """
        model = torch.load(model_path)
        if 'model' in model:
            self.model.load_state_dict(model['model'])
        else:
            self.model.load_state_dict(model)
        
    def save_model(self, output, name=''):
        """
        Util
        """
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, name))

    def save_config(self, output):
        """
        Util
        """
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")
    
    def save_checkpoint(self, fr, output):
        """
        Util. Saves model after every fixed number of frames
        """
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))

    def load_checkpoint(self, model_path):
        """
        Util
        """
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        return fr
    
    def estimate_fisher_matrix(self, batch_size, loss_fn):
        """
        Util for training DQN with EWC.
        """
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.model.F_accum[n] = p.data.cuda() # .to(self.device)
        
        self.model.eval() # in case any normalization layer is not turned off
        buffer_size = self.buffer.size()
        for idx in range(self.config.num_uniform_sampling):
            batch = self.buffer.get_sequential_memory_two(batch_size, buffer_size)
            self.model.zero_grad()
            s0, a, r, s1, done = batch # self.buffer.sample(self.config.batch_size)

            s0 = torch.tensor(s0, dtype=torch.float)
            s1 = torch.tensor(s1, dtype=torch.float)
            a = torch.tensor(a, dtype=torch.long)
            r = torch.tensor(r, dtype=torch.float)
            done = torch.tensor(done, dtype=torch.float)

            if self.config.use_cuda:
                s0 = s0.cuda()
                s1 = s1.cuda()
                a = a.cuda()
                r = r.cuda()
                done = done.cuda()

            q_values = self.model(s0).cuda()
            next_q_values = self.model(s1).cuda()
            next_q_value = next_q_values.max(1)[0]
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
            expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
            loss = (q_value - expected_q_value.detach()).pow(2).mean()
            
            loss.backward() # get gradients
            for n, p in self.model.nn.named_parameters():
                key_name = n.replace('.', '__') 
                self.model.F_accum[key_name].data += p.grad.data ** 2
            
        self.model.train() # put the train mode back on
        for n, p in self.model.nn.named_parameters():
            key_name = n.replace('.', '__') 
            self.model.F_accum[key_name] /= self.config.num_uniform_sampling
        curr_task_fisher = {'{}_est_fisher'.format(key): self.model.F_accum[key] for key in self.layer_names}
        self.update_history_fisher(curr_task_fisher)
        self.model.save_parameters()
        return self.model.fisher

    def update_history_fisher(self, new_fisher):
        """
        Util for training DQN with EWC. Updates Fisher matrix after training a task.
        """
        for key in new_fisher:
            if key not in self.model.fisher:
                self.model.fisher[key] = new_fisher[key]
            else:
                self.model.fisher[key] = self.model.fisher[key] * self.alpha + new_fisher[key] * (1 - self.alpha)
                
    def initialize_fisher(self):
        """
        Util for training DQN with EWC.
        """
        for name, param in deepcopy(self.params).items():
            param.data.zero_()
            self.model.fisher['{}_est_fisher'.format(name)] = param.data.cuda()

    def save_fisher_matrix(self, fr, outputdir, out_name="fisher"):
        """
        Util for training DQN with EWC.
        """
        save_dict = {key: val for key, val in self.model.fisher.items()}
        save_dict['frames'] = fr
        torch.save(save_dict, "{}/{}.tar".format(outputdir, out_name))

    def initialize_fisher(self):
        """
        Util for training DQN with EWC.
        """
        for name, param in deepcopy(self.params).items():
            param.data.zero_()
            self.model.fisher['{}_est_fisher'.format(name)] = param.data.cuda()
        
    def clear_buffer(self):
        """
        Util.
        """
        self.buffer.buffer = []
        self.model.F_accum = {} # []
    
    def clear_prev_buffer(self):
        """
        Util.
        """
        del self.buffer.prev_buffer
        self.buffer.prev_buffer = []

    def load_matrices(self, model_path, fisher_path='', apply_ewc=True):
        """
        Util for training DQN with EWC.
        """
        # load weight of last task
        self.load_weights(model_path)
        
        # load fisher
        if apply_ewc:
            fisher_file = torch.load(fisher_path)
            self.model.fisher = {key: val for key, val in fisher_file.items() if key != 'frames'}
            self.model.save_parameters()
    
    def load_membuf(self, membuf_path, load_prev_buf=False):
        with open(membuf_path, 'rb') as f:
            membuf = pickle.load(f)
        if load_prev_buf:
            self.buffer.extend_prev_buffer(membuf)
        else:
            self.buffer.extend_buffer(membuf)
    
    def save_task_membuf(self, save_location, env_name, agent_id, task_no):
        """
        Util for training DQN.
        """
        # save memory buffer for this task only
        with open(os.path.join(save_location, "membuf_{}_agent-{}-task-{}.pkl".format(env_name, agent_id, task_no)), 'wb') as f:
            pickle.dump(self.buffer.prev_buffer, f)

    def init_simnet(self, weight_dir='simnet'):
        """
        Util for training SimNet.
        """
        if not os.path.exists(os.path.join(weight_dir, "simnet.pkl")):
            print("No existing simnet file found. Initialize a new one")
            os.makedirs(weight_dir, exist_ok=True)
            weight_path = os.path.join(weight_dir, "simnet.pkl")
            for _, param in self.model.nn.named_parameters():
                if param.requires_grad:
                    # pdb.set_trace()
                    nn.init.normal_(param)
                    # pdb.set_trace()
            torch.save(self.model.state_dict(), weight_path)
            print("Saved simnet parameters to: {}".format(weight_path))
        else:
            print("Simnet weight existing and loaded.")
            weight_path = os.path.join(weight_dir, "simnet.pkl")
            self.model.load_state_dict(torch.load(weight_path))
