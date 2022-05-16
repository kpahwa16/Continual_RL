
class Config:
    env: str = None
    gamma: float = None
    learning_rate: float = None
    frames: int = None
    episodes: int = None
    max_buff: int = None
    batch_size: int = None

    epsilon: float = None
    eps_decay: float = None
    epsilon_min: float = None

    state_dim: int = None
    state_shape = None
    state_high = None
    state_low = None
    seed = None
    output = 'out'
    membuf_loadpath = 'temp'
    membuf_parent_savedir = 'membuf'
    membuf_savedir = 'temp'

    action_dim: int = None
    action_high = None
    action_low = None
    action_lim = None

    use_cuda: bool = None

    checkpoint: bool = False
    checkpoint_interval: int = None

    record: bool = False
    record_ep_interval: int = None

    log_interval: int = None
    print_interval: int = None

    update_tar_interval: int = None

    win_reward: float = None
    win_break: bool = None

    env_two: str = None
    
    apply_ewc_flag: bool = False
    lambda_value: int = None
    continue_learning: bool = True
    num_uniform_sampling: int = None
    add_noop: bool = False
    use_frame_skip: bool = False
    num_frame_skip: int = None
    alpha: float = None

    prev_membuf_loadpath: str = None
    agent_id: int = None
    membuf_parent_savedir: str = None
    membuf_savedir: str = None
    task_no: int = None
    learn_new_env: bool = None
    apply_sample_thres: bool = None
    sample_thres: bool = None
