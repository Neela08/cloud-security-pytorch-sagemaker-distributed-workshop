import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from utils_basic import load_dataset_setting, train_model, eval_model, BackdoorDataset
import os
from datetime import datetime
import json
import argparse

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', type=str, default='gloo')
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--world-size', type=int, default=1)
    args = parser.parse_args()

    setup(args.rank, args.world_size)

    GPU = False
    SHADOW_PROP = 0.02
    TARGET_PROP = 0.5
    TARGET_NUM = 16
    np.random.seed(0)
    torch.manual_seed(0)
   
    BATCH_SIZE, N_EPOCH, trainset, testset, is_binary, need_pad, Model, troj_gen_func, random_troj_setting = load_dataset_setting("cifar10")
    tot_num = len(trainset)
    shadow_indices = np.random.choice(tot_num, int(tot_num*SHADOW_PROP))
    target_indices = np.random.choice(tot_num, int(tot_num*TARGET_PROP))
    print("Data indices owned by the attacker:", target_indices)

    SAVE_PREFIX = './shadow_model_ckpt/cifar10'
    if not os.path.isdir(SAVE_PREFIX):
        os.mkdir(SAVE_PREFIX)
    if not os.path.isdir(SAVE_PREFIX+'/models'):
        os.mkdir(SAVE_PREFIX+'/models')

    all_target_acc = []
    all_target_acc_mal = []

    for i in range(TARGET_NUM):
		model = Model(gpu=GPU)
		atk_setting = random_troj_setting('M')
		trainset_mal = BackdoorDataset(trainset, atk_setting, troj_gen_func, choice=target_indices, need_pad=need_pad)
		train_sampler = torch.utils.data.distributed.DistributedSampler(trainset_mal, num_replicas=args.world_size, rank=args.rank)
		trainloader = torch.utils.data.DataLoader(trainset_mal, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, sampler=train_sampler)
		testset_mal = BackdoorDataset(testset, atk_setting, troj_gen_func, mal_only=True)
		testloader_benign = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE)
		testloader_mal = torch.utils.data.DataLoader(testset_mal, batch_size=BATCH_SIZE)

		model = DDP(model, device_ids=[args.rank], output_device=args.rank)
		train_model(model, trainloader, epoch_num=int(N_EPOCH*SHADOW_PROP/TARGET_PROP), is_binary=is_binary, verbose=False)
		save_path = SAVE_PREFIX+'/models/target_troj%s_%d.model'%('M', i)
		if args.rank == 0:
			torch.save(model.module.state_dict(), save_path)
		acc = eval_model(model, testloader_benign, is_binary=is_binary, verbose=False)
		all_target_acc.append(acc)
		acc_mal = eval_model(model, testloader_mal, is_binary=is_binary, verbose=False)
		all_target_acc_mal.append(acc_mal)
    
	log = {'target_num':TARGET_NUM,
           'target_acc':sum(all_target_acc)/len(all_target_acc),
           'target_acc_mal':sum(all_target_acc_mal)/len(all_target_acc_mal)}
    log_path = SAVE_PREFIX+'/troj%s.log'%args.troj_type
    with open(log_path, "w") as outf:
        json.dump(log, outf)
    print ("Log file saved to %s"%log_path)