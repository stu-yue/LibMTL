import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.utils import count_improvement

class _PerformanceMeter(object):
    def __init__(self, task_dict, multi_input, base_result=None):
        
        self.task_dict = task_dict
        self.multi_input = multi_input
        self.task_num = len(self.task_dict)
        self.task_name = list(self.task_dict.keys())
        
        self.weight = {task: self.task_dict[task]['weight'] for task in self.task_name}
        self.base_result = base_result
        self.best_result = {'improvement': -1e+2, 'epoch': 0, 'result': 0}
        
        self.losses = {task: self.task_dict[task]['loss_fn'] for task in self.task_name}
        self.metrics = {task: self.task_dict[task]['metrics_fn'] for task in self.task_name}
        
        self.results = {task:[] for task in self.task_name}
        self.loss_item = np.zeros(self.task_num)
        
        self.has_val = False
        self.improved = False
        
        self._init_display()
        
    def record_time(self, mode='begin'):
        if mode == 'begin':
            self.beg_time = time.time()
        elif mode == 'end':
            self.end_time = time.time()
        else:
            raise ValueError('No support time mode {}'.format(mode))
        
    def update(self, preds, gts, task_name=None):
        with torch.no_grad():
            if task_name is None:
                for tn, task in enumerate(self.task_name):
                    self.metrics[task].update_fun(preds[task], gts[task])
            else:
                self.metrics[task_name].update_fun(preds, gts)
        
    def get_score(self):
        with torch.no_grad():
            for tn, task in enumerate(self.task_name):
                self.results[task] = self.metrics[task].score_fun()
                self.loss_item[tn] = self.losses[task]._average_loss()
    
    def _init_display(self):
        print('='*40)
        init_str = 'LOG FORMAT | '
        for tn, task in enumerate(self.task_name):
            init_str += task+'_LOSS '
            for m in self.task_dict[task]['metrics']:
                init_str += m+' '
            init_str += '| '
        init_str += 'TIME'
        print(init_str)
    
    def display(self, mode, epoch):
        display_str = ''
        if epoch == 0 and self.base_result is None and mode==('val' if self.has_val else 'test'):
            self.base_result = self.results
        if mode == 'train':
            display_str += 'Epoch: {:04d} | '.format(epoch)
        if not self.has_val and mode == 'test':
            self._update_best_result(self.results, epoch)
        if self.has_val and mode != 'train':
            self._update_best_result_by_val(self.results, epoch, mode)
        if mode == 'train':
            p_mode = 'TRAIN'
        elif mode == 'val':
            p_mode = 'VAL'
        else:
            p_mode = 'TEST'
        display_str += '{}: '.format(p_mode)
        for tn, task in enumerate(self.task_name):
            display_str += '{:.4f} '.format(self.loss_item[tn])
            for i in range(len(self.results[task])):
                display_str += '{:.4f} '.format(self.results[task][i])
            display_str += '| '
        display_str += 'Time: {:.4f}'.format(self.end_time-self.beg_time)
        display_str += (' | ' if mode!='test' else '')
        print(display_str)
        
    def display_best_result(self):
        print('='*40)
        print('Best Result: Epoch {}, result {}'.format(self.best_result['epoch'], self.best_result['result']))
        print('='*40)
        
    def _update_best_result_by_val(self, new_result, epoch, mode):
        self.improved = False
        if mode == 'val':
            improvement = count_improvement(self.base_result, new_result, self.weight)
            self.improvement = improvement
            if improvement > self.best_result['improvement']:
                self.best_result['improvement'] = improvement
                self.best_result['epoch'] = epoch
                self.improved = True
        else:
            if epoch == self.best_result['epoch']:
                self.best_result['result'] = new_result
        
    def _update_best_result(self, new_result, epoch):
        improvement = count_improvement(self.base_result, new_result, self.weight)
        self.improvement = improvement
        self.improved = False
        if improvement > self.best_result['improvement']:
            self.best_result['improvement'] = improvement
            self.best_result['epoch'] = epoch
            self.best_result['result'] = new_result
            self.improved = True
        
    def reinit(self):
        for task in self.task_name:
            self.losses[task]._reinit()
            self.metrics[task].reinit()
        self.loss_item = np.zeros(self.task_num)
        self.results = {task:[] for task in self.task_name}
