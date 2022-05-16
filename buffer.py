import random
import numpy as np
import pickle
import time
import os
import pdb


class ReplayBuffer(object):
    """
    This class defines experience replay buffer and its utilities
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.prev_buffer = [] # keep a history of current task only
        self.shared_buffer = []
        self.curr_start_idx = 0
        self.p_idx = 0 # index for cyclically indexing elements in self.prev_buffer
        self.lsc_idx = 0

    def add(self, s0, a, r, s1, done):
        """
        Adds task tuples into buffer.
        Args:
            s0: curr state
            a: action state
            r: reward
            s1: next state
            done: done state
        """
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((s0[None, :], a, r, s1[None, :], done))

    def sample(self, batch_size, focus_curr=False, use_lsc=False):
        """
        Samples experience from buffer to train DQN.
        
        Args:
            batch_size: batch size of data to sample
            focus_curr: turn on to sample experience only from the current task
            use_lsc: turn on to use low-switching cost. No need to use at this point.
        """
        if focus_curr:
            s0, a, r, s1, done = zip(*random.sample(self.buffer[self.curr_start_idx :], batch_size))
        elif use_lsc:
            s0, a, r, s1, done = zip(*random.sample(self.shared_buffer, batch_size))
        else:
            s0, a, r, s1, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(s0), a, r, np.concatenate(s1), done

    def size(self):
        return len(self.buffer)

    def get_sequential_memory(self, batch_size):
        """
        Util. Replaying memory for EWC.
        """
        intervals = []
        num_batches = self.size() // batch_size
        for i in range(0, self.size(), batch_size):
            chunk = self.buffer[i : i + batch_size]
            s0, a, r, s1, done = zip(*chunk)
            intervals.append((np.concatenate(s0), a, r, np.concatenate(s1), done))
        return intervals
    
    def get_sequential_memory_two(self, batch_size, buffer_size):
        """
        Util.
        """
        i = random.randint(0, buffer_size - batch_size)
        chunk = self.buffer[i : i + batch_size]
        s0, a, r, s1, done = zip(*chunk)
        return np.concatenate(s0), a, r, np.concatenate(s1), done
    
    def extend_buffer(self, loaded_buffer):
        """
        Util.
        """
        self.buffer.extend(loaded_buffer)
    
    def extend_prev_buffer(self, loaded_buffer):
        """
        Util.
        """
        self.prev_buffer.extend(loaded_buffer)

    def clear_buffer(self):
        """
        Util.
        """
        self.buffer = []
        self.prev_buffer = []
    
    def update_prev_buffer(self):
        """
        Util.
        """
        self.prev_buffer = self.buffer[self.curr_start_idx :]
    
    def get_past_buffer_samples(self):
        """
        Util. Replaying memory frames.
        """
        if self.p_idx == len(self.prev_buffer):
            self.p_idx = 0
        output = self.prev_buffer[self.p_idx]
        self.p_idx += 1
        return output
    
    def set_curr_idx(self, orig_buf_size):
        """
        Util.
        """
        self.curr_start_idx = orig_buf_size

    def update_shared_buffer(self, num, file_path):
        """
        Util.
        """
        # read existing buffer
        start_time = time.time()
        print("self.lsc_idx = {}".format(self.lsc_idx))
        with open(file_path, "rb") as f:
            old_buffer = pickle.load(f)
        load_time = time.time()
        # write content to buffer
        content = self.buffer[self.lsc_idx : self.lsc_idx + num]
        old_buffer.extend(content)
        self.lsc_idx += num # len(content), since update self.lsc_idx by the number of elements actually written
        update_time = time.time()
        with open(file_path, "wb") as f:
            pickle.dump(old_buffer, f)
        end_time = time.time()
        print("self.lsc_idx = {}".format(self.lsc_idx))
        print("load: {}, update: {}, complete: {}".format(load_time - start_time, update_time - start_time, end_time - start_time))
    
    def load_shared_buffer(self, file_path):
        """
        Util.
        """
        start_time = time.time()
        with open(file_path, "rb") as f:
            self.shared_buffer = pickle.load(f)
        end_time = time.time()
        print("total load time: {}, size: {}".format(end_time - start_time, len(self.shared_buffer)))

    def find_learned_task_buffer(self, curr_task_name, membuf_dir):
        """
        Util.
        """
        memory_buffers = os.listdir(membuf_dir)
        for fn in memory_buffers:
            task_name
