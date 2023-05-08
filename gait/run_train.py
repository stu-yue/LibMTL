import torch, argparse
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import builtins
import sys
sys.path.append('../')
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
    parser.add_argument('--resume', default=False, type=bool, help='')
    parser.add_argument('--resume_path', default='', type=str, help='')
    parser.add_argument('--result_path', default='', type=str, help='')
    parser.add_argument('--milestones', default=(50,), nargs='+', help='for multi-step scheduler')
    parser.add_argument('--train_mode', default='train', type=str, help='trainval, train')
    parser.add_argument('--bs', default=128, type=int, help='batch size')
    parser.add_argument('--data_root', default='/', type=str, help='dataset path')
    parser.add_argument('--pool_method', default='gap', type=str, help='')
    return parser.parse_args()


def main(params):
    kwargs, optim_param, scheduler_param = prepare_args(params)
    
    # define tasks
    task_dict = {'load': {'metrics':['Acc'], 
                              'metrics_fn': AccMetric(),
                              'loss_fn': CELoss(),
                              'weight': [1]}, 
                #  'slope': {'metrics':['Acc'], 
                #            'metrics_fn': AccMetric(),
                #            'loss_fn': CELoss(),
                #            'weight': [1]},
                #  'speed': {'metrics':['Acc'], 
                #             'metrics_fn': AccMetric(),
                #             'loss_fn': CELoss(),
                #             'weight': [1]},
                 }  

    # define encoder and decoders
    def encoder_class(input_shape=params.img_size[1:], pool_method=params.pool_method):
        return SelKerGru(input_shape=input_shape, pool_method=pool_method)
    num_out_features = {"load": 2, "slope": 3, "speed": 3}
    decoders = nn.ModuleDict({task: nn.Linear(512, 
                                              num_out_features[task]) for task in list(task_dict.keys())})
    
    class GaitTrainer(Trainer):
        def __init__(self, task_dict, weighting, architecture, encoder_class, 
                     decoders, rep_grad, multi_input, optim_param, scheduler_param, params, **kwargs):
            self.params = params
            self.result_path, self.log_path, self.ckpt_path = self._init_files(params)
            self.logger = self._init_logger()
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
        
        def _init_files(self, params):
            if params.resume:
                result_path = params.resume_path
                ckpt_path = os.path.join(result_path, "ckpts")
                log_path = os.path.join(result_path, "logs")
            else:
                base_dir = "{}-{}-{}".format(params.arch, params.weighting, get_local_time())
                result_path = os.path.join(params.result_path, base_dir)
                ckpt_path = os.path.join(result_path, "ckpts")
                log_path = os.path.join(result_path, "logs")
                create_dirs([result_path, log_path, ckpt_path])
                with open(
                    os.path.join(result_path, "config.txt"), "w", encoding="utf-8"
                ) as fout:
                    fout.write('\n'.join([
                        "params." + str(k) + " = " + str(v) 
                            for k, v in vars(params).items()]))
            init_logger_config("info", log_path)
            return result_path, log_path, ckpt_path
        
        def _init_logger(self,):
            self.logger = getLogger(__name__)
            # hack print
            def use_logger(*msg, level="info"):
                try:
                    for m in msg:
                        getattr(self.logger, level)(m)
                except os.error:
                    raise ("logging have no {} level".format(level))
            builtins.print = use_logger
            return self.logger
        
    GaitModel = GaitTrainer(task_dict=task_dict, 
                          weighting=params.weighting, 
                          architecture=params.arch, 
                          encoder_class=encoder_class, 
                          decoders=decoders,
                          rep_grad=params.rep_grad,
                          multi_input=params.multi_input,
                          optim_param=optim_param,
                          scheduler_param=scheduler_param,
                          params=params,
                          **kwargs)
    
    # prepare dataloaders
    gait_train_set = GaitTackerDataset(data_root=params.data_root, mode="train", tasks=task_dict.keys())
    gait_val_set = GaitTackerDataset(data_root=params.data_root, mode="val", tasks=task_dict.keys())
    gait_test_set = GaitTackerDataset(data_root=params.data_root, mode="test", tasks=task_dict.keys())
    
    gait_train_loader = torch.utils.data.DataLoader(
        dataset=gait_train_set,
        batch_size=params.bs,
        shuffle=True,
        num_workers=2,
        pin_memory=True)
    gait_val_loader = torch.utils.data.DataLoader(
        dataset=gait_val_set,
        batch_size=params.bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True)
    gait_test_loader = torch.utils.data.DataLoader(
        dataset=gait_test_set,
        batch_size=params.bs,
        shuffle=False,
        num_workers=2,
        pin_memory=True)
    
    GaitModel.train(gait_train_loader, gait_test_loader, params.epoch, val_dataloaders=gait_val_loader)

if __name__ == "__main__":
    params = parse_args(LibMTL_args)
    # gait task
    params.data_root = "/home/wangyue/gaittracker-1.0.0"
    params.epoch        = 40
    params.bs           = 128
    params.pool_method  = "msd"
    # general
    params.resume       = False
    params.resume_path  = ""
    params.result_path  = "/home/wangyue/result_mtl"
    params.gpu_id       = 2
    params.seed         = 1
    params.weighting    = "GradNorm"
    params.arch         = "CGC"
    params.rep_grad     = False
    # optim
    params.optim        = "adam"
    params.lr           = 0.001
    params.momentum     = 0.9
    params.weight_decay = 0.0001
    # scheduler
    params.scheduler    = "multistep"
    params.milestones   =  [50,]
    params.gamma        = 0.1
    # args for weighting
    ## DWA
    params.T            = 2.0
    ## GradVac
    params.beta         = 0.5
    ## GradNorm
    params.alpha        = 1.5
    # args for architecture
    ## CGC
    params.img_size     = [2, 200, 12]
    params.num_experts  = [1, 1, 1, 1]
    
    # set device
    set_device(params.gpu_id)
    # set random seed
    set_random_seed(params.seed)
    main(params)