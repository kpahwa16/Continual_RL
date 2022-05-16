import argparse
import os
import random
import torch
import time
import pdb
from copy import deepcopy
from torch import autograd
from torch.optim import Adam
from tester import Tester
from buffer import ReplayBuffer
from common.wrappers import make_atari, wrap_deepmind, wrap_pytorch
from config import Config
from core.util import get_class_attr_val
from model import CnnDQNTypeTwo
from trainer import Trainer


class CnnDDQNAgentTypeTwo:
    def __init__(self, config: Config):
        self.config = config
        self.is_training = True
        self.buffer = ReplayBuffer(self.config.max_buff)

        self.model = CnnDQNTypeTwo(self.config.state_shape, self.config.action_dim)
        self.target_model = CnnDQNTypeTwo(self.config.state_shape, self.config.action_dim)
        self.target_model.load_state_dict(self.model.state_dict())
        self.model_optim = Adam(self.model.parameters(), lr=self.config.learning_rate)

        if self.config.use_cuda:
            self.cuda()
        self.params = {n.replace('.', '__') : p for n, p in self.model.nn.named_parameters() if p.requires_grad}
        self.layer_names = [name.replace('.', '__') for name, param in self.model.nn.named_parameters() if param.requires_grad]
        self.debug_task2_loss = []
        self.debug_ewc_loss = []
        
    # def act(self, state, epsilon=None):
    def act(self, state, epsilon=None, apply_ewc=False, first_task=False): 
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

    def learning(self, fr, loss_fn=None, apply_ewc=False):
        s0, a, r, s1, done = self.buffer.sample(self.config.batch_size)

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
        # pdb.set_trace()
        next_q_values = self.model(s1).cuda()
        # pdb.set_trace()
        next_q_state_values = self.target_model(s1).cuda()
        # pdb.set_trace()
        q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
        expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
        # Notice that detach the expected_q_value
        task_loss = (q_value - expected_q_value.detach()).pow(2).mean()
        loss = None
        if apply_ewc:
            ewc_loss = self.model.get_ewc_terms(lambda_value=self.config.lambda_value, fr=fr)
            # if fr % 10000 == 0:
            #     pdb.set_trace()
            # debug_length = len(self.debug_task2_loss)
            # loss += ewc_loss
            loss = task_loss + ewc_loss
            # loss = task_loss * 0 + ewc_loss
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

        if fr % self.config.update_tar_interval == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        return loss.item()

    def estimate_fisher_matrix(self, batch_size, loss_fn):
        print("Checking if model and self.model are the same thing. Need to unify this variable!")
        self.target_model.load_state_dict(self.model.state_dict()) # possibly lead to computtional graph problem
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.model.F_accum[n] = p.data.cuda() # .to(self.device)
        print("see if model.F_accum and self.model.F_accum are the same")
        
        self.model.eval() # in case any normalization layer is not turned off
        self.target_model.eval() 
        trajectory = self.buffer.get_sequential_memory(batch_size)
        print("check if model.parameters() is equivalent to self.model.parameters()")
        # pdb.set_trace()
        for idx, batch in enumerate(trajectory):
            self.model.zero_grad()
            self.target_model.zero_grad()
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

            ###
            '''
            q_values = self.model(s0).cuda()
            pdb.set_trace()
            next_q_values = self.model(s1).cuda()
            pdb.set_trace()
            # next_q_state_values = self.target_model(s1).cuda()
            next_q_state_values = self.model(s1).cuda() # Check how to compute DDQM's loss in paper # self.target_model(s1).cuda()
            pdb.set_trace()
            
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
            next_q_value = next_q_state_values.gather(1, next_q_values.max(1)[1].unsqueeze(1)).squeeze(1)
            expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
            # Notice that detach the expected_q_value
            loss = (q_value - expected_q_value.detach()).pow(2).mean()
            '''
            # DQN's way of estimating fisher matrix
            q_values = self.model(s0).cuda()
            next_q_values = self.model(s1).cuda()
            next_q_value = next_q_values.max(1)[0]
            q_value = q_values.gather(1, a.unsqueeze(1)).squeeze(1)
            expected_q_value = r + self.config.gamma * next_q_value * (1 - done)
            loss = (q_value - expected_q_value.detach()).pow(2).mean()
            ###

            loss.backward() # get gradients
            for n, p in self.model.nn.named_parameters():
                key_name = n.replace('.', '__') 
                self.model.F_accum[key_name].data += p.grad.data ** 2
            
        self.model.train() # put the train mode back on
        self.target_model.train()
        for n, p in self.model.nn.named_parameters():
            key_name = n.replace('.', '__') 
            self.model.F_accum[key_name] /= self.buffer.size()
        # pdb.set_trace()

        # update fisher matrix and weight of last task
        self.model.fisher = {'{}_est_fisher'.format(key): self.model.F_accum[key] for key in self.layer_names}
        self.model.save_parameters()
        # pdb.set_trace()
        return self.model.fisher

    def cuda(self):
        self.model.cuda()
        self.target_model.cuda()

    def load_weights(self, model_path):
        model = torch.load(model_path)
        if 'model' in model:
            self.model.load_state_dict(model['model'])
        else:
            self.model.load_state_dict(model)
        
    def save_model(self, output, name=''):
        torch.save(self.model.state_dict(), '%s/model_%s.pkl' % (output, name))

    def save_config(self, output):
        with open(output + '/config.txt', 'w') as f:
            attr_val = get_class_attr_val(self.config)
            for k, v in attr_val.items():
                f.write(str(k) + " = " + str(v) + "\n")

    def save_fisher_matrix(self, fr, outputdir):
        save_dict = {key: val for key, val in self.model.fisher.items()}
        save_dict['frames'] = fr
        torch.save(save_dict, "{}/fisher.tar".format(outputdir))

    def save_checkpoint(self, fr, output):
        checkpath = output + '/checkpoint_model'
        os.makedirs(checkpath, exist_ok=True)
        torch.save({
            'frames': fr,
            'model': self.model.state_dict()
        }, '%s/checkpoint_fr_%d.tar'% (checkpath, fr))

    def load_checkpoint(self, model_path):
        checkpoint = torch.load(model_path)
        fr = checkpoint['frames']
        self.model.load_state_dict(checkpoint['model'])
        self.target_model.load_state_dict(checkpoint['model'])
        return fr
    
    def initialize_fisher_and_weight_buffer(self):
        for n, p in deepcopy(self.params).items():
            p.data.zero_()
            self.model.fisher[n] = p.data.cuda() # to(self.device)
        # for n, p in deepcopy(self.params).items():
        #     p.data.zero_()
        #     self.model.parameters_last_task[n] = p.data.cuda() # .to(self.device)

    def clear_buffer(self):
        self.buffer.buffer = []
        self.model.F_accum = {} # []
    
    def load_matrices(self, model_path, fisher_path):
        # load fisher
        fisher_file = torch.load(fisher_path)
        self.model.fisher = {key: val for key, val in fisher_file.items() if key != 'frames'}
        pdb.set_trace()
        # load weight of last task
        self.load_weights(model_path)
        self.model.save_parameters()
        pdb.set_trace()
        self.align_target_net_weight()
        pdb.set_trace()
    
    def align_target_net_weight(self):
        self.target_model.load_state_dict(self.model.state_dict())


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--train', dest='train', action='store_true', help='train model')
    parser.add_argument('--env', default='PongNoFrameskip-v4', type=str, help='gym environment')
    parser.add_argument('--env_two', default='PongNoFrameskip-v4', type=str, help='gym environment')
    parser.add_argument('--test', dest='test', action='store_true', help='test model')
    parser.add_argument('--retrain', dest='retrain', action='store_true', help='retrain model')
    parser.add_argument('--cl_retrain', dest='cl_retrain', action='store_true', help='cl_retrain model')
    parser.add_argument('--model_path', type=str, help='if test or retrain, import the model')
    parser.add_argument('--fisher_path', type=str, help='if test or retrain with EWC, import the fisher matrix')
    parser.add_argument('--lambda_value', type=float, default=10000000000, help='learning rate of first task')
    parser.add_argument('--learning_rate1', type=float, default=1e-5, help='learning rate of first task')
    parser.add_argument('--learning_rate2', type=float, default=1e-5, help='learning rate of first task')
    parser.add_argument('--apply_ewc', dest='apply_ewc', action='store_true', help='run EWC and a second task')
    args = parser.parse_args()
    # atari_ddqn.py --train --env PongNoFrameskip-v4

    config = Config()
    config.env = args.env
    config.gamma = 0.99
    config.epsilon = 1
    config.epsilon_min = 0.01
    config.eps_decay = 30000
    config.frames = 3000000 # 600000 # 2000000
    config.use_cuda = True
    config.learning_rate = args.learning_rate1 # 1e-5 works for RoadRunnerNoFrameskip-v4 # 2e-4 # 1e-4
    config.learning_rate_two = args.learning_rate2
    config.max_buff = 100000
    config.update_tar_interval = 1000
    config.batch_size = 32
    config.print_interval = 20000 # 5000
    config.log_interval = 5000
    config.checkpoint = True
    config.checkpoint_interval = 100000
    config.win_reward = 40 # 8 # 20 # 8  # 18  # PongNoFrameskip-v4
    config.win_break = True
    config.env_two = "BoxingNoFrameskip-v4"
    config.apply_ewc_flag = True if args.apply_ewc else False
    config.lambda_value = args.lambda_value # 10000000000 # 100000000000000000000000 # 10000000000000000000000000000 # 200000
    config.continue_learning = False

    # handle the atari env
    env = make_atari(config.env)
    env = wrap_deepmind(env)
    env = wrap_pytorch(env)
    env_two = make_atari(config.env_two)
    env_two = wrap_deepmind(env_two)
    env_two = wrap_pytorch(env_two)

    config.action_dim = env.action_space.n
    config.state_shape = env.observation_space.shape
    agent = CnnDDQNAgentTypeTwo(config)

    if args.train:
        agent.initialize_fisher_and_weight_buffer()
        trainer = Trainer(agent, env, config)
        print("First task doesn't apply EWC")
        print("Learning rate of task 1: {}".format(config.learning_rate))
        start_time = time.time()
        trainer.train(apply_ewc=False)
        task_one_endtime = time.time()
        print("Duration of training Task 1: {}".format(task_one_endtime - start_time))
        # pdb.set_trace()

        if config.continue_learning:
            print("Second task applies EWC: {}".format(config.env_two))    
            agent.clear_buffer()
            agent.align_target_net_weight()
            # config.win_reward = 18
            config.learning_rate = config.learning_rate_two
            print("Learning rate of task 2: {}".format(config.learning_rate))
            task_two_begintime = time.time()
            trainer_two = Trainer(agent, env_two, config)
            trainer_two.train(apply_ewc=config.apply_ewc_flag)
            task_two_endtime = time.time()
            print("Repeat. Duration of training Task 1: {}".format(task_one_endtime - start_time))
            print("Duration of training Task 2: {}".format(task_two_endtime - task_two_begintime))
            print("Duration of experiment: {}".format(task_two_endtime - start_time))

    elif args.test:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        tester = Tester(agent, env, args.model_path)
        tester.test() # (debug=True)

    elif args.cl_retrain:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        if args.fisher_path is None:
            print('please add the fisher path:', '--fisher_path xxxx')
            exit(0)
        print("Loading the task model and matrices from last task")
        # agent.load_weights(args.model_path)
        # agent.model.save_parameters()
        agent.load_matrices(args.model_path, args.fisher_path)
        trainer = Trainer(agent, env, config)
        start_time = time.time()
        print("Learning the next task with EWC: {}".format(config.env))
        trainer.train(apply_ewc=config.apply_ewc_flag)
        endtime = time.time()
        print("Duration of training the current task: {}".format(endtime - start_time))
        agent.clear_buffer()

    elif args.retrain:
        if args.model_path is None:
            print('please add the model path:', '--model_path xxxx')
            exit(0)
        fr = agent.load_checkpoint(args.model_path)
        trainer = Trainer(agent, env, config)
        start_time = time.time()
        trainer.train(fr)
        endtime = time.time()
        print("Duration of training the current task: {}".format(endtime - start_time))
        agent.clear_buffer()
