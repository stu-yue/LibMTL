import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from utils import *
from create_dataset import GaitTackerDataset
from skgru import SelKerGru

from LibMTL import Trainer
from LibMTL.utils import set_random_seed, set_device
from LibMTL.config import LibMTL_args, prepare_args
from LibMTL.metrics import AccMetric
from LibMTL.loss import CELoss
import LibMTL.weighting as weighting_method
import LibMTL.architecture as architecture_method


def parse_args(parser):
    parser.add_argument('--multi', action='store_true', default=True, help='enable multi-task mode')
    parser.add_argument('--train_mode', default='trainval', type=str, help='trainval, train')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--data_root', default='/', type=str, help='dataset path')
    parser.add_argument('--epoch', default=100, type=int, help='epoch nums')
    parser.add_argument('--input_shape', default=(300, 9), nargs='+', help='')
    parser.add_argument('--pool_method', default='gap', type=str, help='')
    return parser.parse_args()


def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    
    # prepare dataloaders
    gait_train_set = GaitTackerDataset(data_root=params.data_root, mode="train", multi=True)
    gait_test_set = GaitTackerDataset(data_root=params.data_root, mode="test", multi=True)
    
    gait_train_loader = torch.utils.data.DataLoader(
        dataset=gait_train_set,
        batch_size=params.bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True)
    
    gait_test_loader = torch.utils.data.DataLoader(
        dataset=gait_test_set,
        batch_size=params.bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True)
    
    # define tasks
    task_dict = {'load': {'metrics':['Acc'], 
                              'metrics_fn': AccMetric(),
                              'loss_fn': CELoss(),
                              'weight': [1]}, 
                 'slope': {'metrics':['Acc'], 
                           'metrics_fn': AccMetric(),
                           'loss_fn': CELoss(),
                           'weight': [1]},
                 'speed': {'metrics':['Acc'], 
                            'metrics_fn': AccMetric(),
                            'loss_fn': CELoss(),
                            'weight': [1]}}

    # define encoder and decoders
    def encoder_class(kwargs=vars(params)):
        return SelKerGru(**kwargs)
    num_out_features = {"load": 2, "slope": 3, "speed": 3}
    decoders = nn.ModuleDict({task: nn.Linear(512, 
                                              num_out_features[task]) for task in list(task_dict.keys())})
    
    class GaitTrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, **kwargs):
            super(GaitTrainer, self).__init__(task_dict=task_dict, 
                                            weighting=weighting_method.__dict__[weighting], 
                                            architecture=architecture_method.__dict__[architecture], 
                                            encoder_class=encoder_class, 
                                            decoders=decoders,
                                            rep_grad=rep_grad,
                                            multi_input=multi_input,
                                            optim_param=optim_param,
                                            scheduler_param=scheduler_param,
                                            **kwargs)
        
        def process_preds(self, preds):
            return preds
        
    GaitModel = GaitTrainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=encoder_class, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          **kwargs)
    GaitModel.train(gait_train_loader, gait_test_loader, params.epoch)
    
if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)