import os
import pdb
import torch
import scipy
import torch.nn as nn
from scipy.stats import wasserstein_distance
from model import TinyDQN

def get_weight_matrix(weight_path, num_inputs=128, actions_dim=18):
    model = TinyDQN(num_inputs, actions_dim)
    model.load_state_dict(torch.load(weight_path))
    return [param for _, param in model.nn.named_parameters()]

def get_model_output(model, state):
    with torch.no_grad():
        return model.forward(state)

def generate_outputs(model, other_model, states):
    with torch.no_grad():
        value = model.forward(state)
    with torch.no_grad():
        other_value = other_model(state)
    abs_diff = torch.abs(value - other_value) / value.shape[-1]
    if p == 1:
        return torch.sum(abs_diff)
    else:
        return torch.pow(abs_diff, p).sum()

def compare_outputs(model, other_model, states):
    model_outputs = [get_model_output(model, state) for state in states]
    other_model_outputs = [get_model_output(other_model, state) for state in states]
    mean_model_output = torch.zeros(18)
    mean_other_model_output = torch.zeros(18)
    for i in range(len(model_outputs)):
        mean_model_output += model_outputs[i]
        mean_other_model_output += other_model_outputs[i]
    mean_model_output /= mean_model_output.shape[-1]
    mean_other_model_output /= mean_other_model_output.shape[-1]
    return wasserstein_distance(mean_model_output, mean_other_model_output)


def get_weight_matrix(weight_path, num_inputs=128, actions_dim=18):
    model = TinyDQN(num_inputs, actions_dim)
    model.load_state_dict(torch.load(weight_path))
    return [param for _, param in model.nn.named_parameters()]


def compare_model_wasserstein(this_param_dir, 
                              param_dir, 
                              num_inputs=128, 
                              actions_dim=18, 
                              ignore_current_round=False,
                              task_no=0, 
                              env="RoadRunner-ram-v0", 
                              p=1):
    exp_dirs = [directory for directory in os.listdir(param_dir) if directory != this_param_dir and not directory.startswith(".ipynb")]
    if ignore_current_round:
        exp_dirs = [dir for dir in exp_dirs if int(dir.split("-")[-2]) < task_no]
    sim_model = "model_last_simnet_{}.pkl".format('-'.join(this_param_dir.split("-")[:3]))
    this_model = load_test_model(os.path.join(param_dir, this_param_dir, sim_model), num_inputs, actions_dim) 
    other_weight_paths = [] # [os.path.join(param_dir, dir, sim_model) for dir in exp_dirs]
    for dir in exp_dirs:
        env_name = '-'.join(dir.split("-")[:3])
        other_sim_model = "model_last_simnet_{}.pkl".format(env_name)
        other_weight_paths.append(os.path.join(param_dir, dir, other_sim_model))
    other_models = [load_test_model(path, num_inputs, actions_dim) for path in other_weight_paths]
    state_inputs = [torch.tensor(torch.randint(0, 256, (128, )), dtype=torch.float) for _ in range(1000)]
    result = [compare_outputs(this_model, other_models[x], state_inputs) for x in range(len(other_models))]
    
    return exp_dirs[int(torch.argmin(torch.tensor(result)))]


def compare_weight_similarity(target_param, shared_dir, num_inputs=128, 
                              actions_dim=18, ignore_current_round=False, 
                              task_no=0, n_closest=1):
    weights = [fn for fn in os.listdir(shared_dir) if fn.startswith("model_last_simnet") and fn != target_param and not fn.startswith(".ipynb")]
    if ignore_current_round:
        weights = [fn for fn in weights if int(fn.split("_")[-2]) != task_no]
    this_weight = get_weight_matrix(os.path.join(shared_dir, target_param), num_inputs, actions_dim) 
    other_weight_paths = [] # [os.path.join(shared_dir, dir, sim_model) for dir in exp_dirs]
    for fn in weights:
        other_weight_paths.append(os.path.join(shared_dir, fn))
    other_weights = [get_weight_matrix(path, num_inputs, actions_dim) for path in other_weight_paths]
    result = torch.tensor([get_layerwise_euclidean(this_weight, other_weights[x]) for x in range(len(other_weights))])
    _, rank_idx = torch.topk(result, n_closest, largest=False)
    return [weights[x] for x in rank_idx]


def get_ids(sim_fn):
    split_fn = sim_fn.split("_")
    agent_id, task_no = split_fn[-4], split_fn[-2]
    return agent_id, task_no

def find_replay_buffer(sim_files, shared_dir):
    file_ids = [get_ids(fn) for fn in sim_files]
    agent_ids, task_nos = zip(*file_ids)
    shared_membufs = [fn for fn in os.listdir(shared_dir) if fn.startswith("membuf_")]
    located_membufs = []
    for membuf in shared_membufs:
        split_fn = membuf[:-4].split("-")
        aid, tno = split_fn[-3], split_fn[-1]
        for agent_id, task_no in file_ids:
            pdb.set_trace()
            if agent_id != aid or task_no != tno:
                continue 
            located_membufs.append(membuf)
    return located_membufs

# find replay buffer when not using result of SimNet comparison
def find_replay_buffer_nosim(replay_list, agent_id=-1, task_id=-1, task_name="", exact_match=False):
    result = []
    for fn in replay_list:
        curr_agent_id = int(fn.split("-")[-3])
        if exact_match:
            curr_task_name = fn.split("_")[-2]
            if curr_agent_id != agent_id and curr_task_name == task_name:
                result.append(fn)
        else:
            curr_task_id = int(fn.split("-")[-1][:-4])
            if curr_agent_id != agent_id and curr_task_id == task_id - 1:
                result.append(fn)
    return result
    
