import torch
from torch import nn
from torch.autograd import Variable
import pdb


class DQN(nn.Module):
    """
    This class defines DQN model and related utilities.
    """
    def __init__(self, num_inputs, actions_dim):
        super(DQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, actions_dim)
        )
        self.params = {n.replace('.', '__') : p for n, p in self.nn.named_parameters() if p.requires_grad}
        self.F_accum = {}
        self.fisher = {}
        self.parameters_last_task = {}

    def forward(self, x):
        return self.nn(x)

    def save_parameters(self):
        for name, param in self.nn.named_parameters():
            if param.requires_grad:
                n = name.replace('.', '__')
                self.parameters_last_task['{}_est_mean'.format(n)] = param.data.clone()

    def get_ewc_terms(self, lambda_value=1000, fr=-1): # 0.90
        losses = 0
        for name, param in self.nn.named_parameters():
            n = name.replace('.', '__')
            mean = self.parameters_last_task.get('{}_est_mean'.format(n), None)
            fisher = self.fisher.get('{}_est_fisher'.format(n), None)
            ewc_loss = lambda_value * 0.5 * fisher * (param - mean) ** 2
            losses += ewc_loss.sum()
        return losses


class DQNTypeTwo(nn.Module):
    def __init__(self, num_inputs, actions_dim, num_base_filter=32): # 32):
        super(DQNTypeTwo, self).__init__()
        
        self.input_shape = num_inputs
        self.num_actions = actions_dim

        self.nn = nn.Sequential(
            nn.Conv2d(self.input_shape[0], num_base_filter, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(num_base_filter, num_base_filter * 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(num_base_filter * 2, num_base_filter * 2, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(49 * num_base_filter * 2, 512), # wait for it to make mistake
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        self.F_accum = []
        self.layer_names = [name for name, param in self.nn.named_parameters() if param.requires_grad]
        self.fisher = {}
        self.parameters_last_task = {}

    def forward(self, x):
        return self.nn(x)

    def save_parameters(self):
        for name, param in self.nn.named_parameters():
            if param.requires_grad:
                n = name.replace('.', '__')
                self.parameters_last_task['{}_est_mean'.format(n)] = param.data.clone()

    def get_ewc_terms(self, lambda_value=0.90, fr=-1):
        losses = []
        if fr % 10000 == 0:
            pdb.set_trace()
        for name, param in self.nn.named_parameters():
            n = name.replace('.', '__')
            mean = self.parameters_last_task.get('{}_est_mean'.format(n), None)
            fisher = self.fisher.get('{}_est_fisher'.format(n), None)
            mean = Variable(mean)
            fisher = Variable(fisher.data)
            losses.append((fisher * (param - mean) ** 2).sum() )
        if fr % 10000 == 0:
            pdb.set_trace()
        return lambda_value / 2 * sum(losses)


class CnnDQNTypeTwo(nn.Module):
    def __init__(self, inputs_shape, num_actions, num_intermediate_filters=32):
        super(CnnDQNTypeTwo, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions
        self.num_intermediate_filters = num_intermediate_filters

        self.nn = nn.Sequential(
            nn.Conv2d(self.inut_shape[0], self.num_intermediate_filters, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(self.num_intermediate_filters, self.num_intermediate_filters * 2, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(self.num_intermediate_filters * 2, self.num_intermediate_filters * 2, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(49 * self.num_intermediate_filters * 2, 512), # wait for it to make mistake
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )
        
        self.F_accum = {} # []
        self.params = {n.replace('.', '__') : p for n, p in self.nn.named_parameters() if p.requires_grad}
        self.fisher = {}
        self.parameters_last_task = {}
        
    def forward(self, x):
        return self.nn(x)

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)
    
    def save_parameters(self):
        for name, param in self.nn.named_parameters():
            if param.requires_grad:
                n = name.replace('.', '__')
                self.parameters_last_task['{}_est_mean'.format(n)] = param.data.clone()
    
    def get_ewc_terms(self, lambda_value=20.0, fr=-1): # 0.90
        losses = 0
        for name, param in self.nn.named_parameters():
            n = name.replace('.', '__')
            mean = self.parameters_last_task.get('{}_est_mean'.format(n), None)
            fisher = self.fisher.get('{}_est_fisher'.format(n), None)
            ewc_loss = lambda_value * 0.5 * fisher * (param - mean) ** 2
            losses += ewc_loss.sum()
        return losses


class CnnDQN(nn.Module):
    def __init__(self, inputs_shape, num_actions):
        super(CnnDQN, self).__init__()

        self.inut_shape = inputs_shape
        self.num_actions = num_actions

        self.features = nn.Sequential(
            nn.Conv2d(inputs_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(self.features_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def features_size(self):
        return self.features(torch.zeros(1, *self.inut_shape)).view(1, -1).size(1)


class TinyDQN(nn.Module):
    """
    This class defines SimNet.
    """
    def __init__(self, num_inputs, actions_dim):
        super(TinyDQN, self).__init__()

        self.nn = nn.Sequential(
            nn.Linear(num_inputs, 5),
            nn.ReLU(),
            nn.Linear(5, actions_dim)
        )
        self.params = {n.replace('.', '__') : p for n, p in self.nn.named_parameters() if p.requires_grad}
        self.F_accum = {}
        self.fisher = {}
        self.parameters_last_task = {}

    def forward(self, x):
        return self.nn(x)
