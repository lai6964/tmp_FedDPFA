import argparse
import copy
import logging
import os
import random
import time
import warnings

import numpy as np
import torch
from torch import nn

from flcore.servers.serverala import FedALA
from flcore.servers.serveras import FedAS
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverbn import FedBN
from flcore.servers.servercp import FedCP
from flcore.servers.serverdfpa import FedDFPA
from flcore.servers.serverfsa import FedFSA
from flcore.servers.serverlocal import Local
from flcore.servers.serverpac import FedPAC
from flcore.servers.serverpcl import FedPCL
from flcore.servers.serverper import FedPer
from flcore.servers.serverproto import FedProto
from flcore.servers.serverprox import FedProx
from flcore.servers.serverrep import FedRep
from flcore.servers.serverrod import FedROD
from flcore.servers.serverscaffold import SCAFFOLD
from flcore.trainmodel.models import BaseHeadSplit, BaseMineSplit, Decoder, FedAvgCNN
from flcore.trainmodel.mobilenet_v2 import mobilenet_v2
from flcore.trainmodel.resnet import resnet10, resnet18
from utils.mem_utils import MemReporter
from utils.result_utils import average_data


SERVER_CLASSES = {
    "FedDFPA": FedDFPA,
    "FedAvg": FedAvg,
    "FedProto": FedProto,
    "FedROD": FedROD,
    "FedPAC": FedPAC,
    "FedCP": FedCP,
    "FedALA": FedALA,
    "FedAS": FedAS,
    "FedPCL": FedPCL,
    "FedFSA": FedFSA,
    "Local": Local,
    "FedProx": FedProx,
    "SCAFFOLD": SCAFFOLD,
    "FedBN": FedBN,
    "FedPer": FedPer,
    "FedRep": FedRep,
}
SUPPORTED_ALGORITHMS = tuple(SERVER_CLASSES)
SUPPORTED_MODELS = ("CNN", "ResNet18", "ResNet10", "MobileNet")
HEAD_SPLIT_ALGORITHMS = frozenset({
    "FedAvg", "FedProto", "FedROD", "FedPAC", "FedCP",
    "FedAS", "FedFSA", "FedPer", "FedRep",
})

logging.getLogger().setLevel(logging.ERROR)
warnings.simplefilter("ignore")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ('yes', 'true', 't', 'y', '1'):
        return True
    if value in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def create_model(model_name, args):
    if model_name == "CNN":
        if "MNIST" in args.dataset:
            channels, flattened_dim = 1, 1024
        elif "Cifar10" in args.dataset:
            channels, flattened_dim = 3, 1600
        elif "Omniglot" in args.dataset:
            channels, flattened_dim = 1, 33856
        else:
            channels, flattened_dim = 3, 10816
        model = FedAvgCNN(
            in_features=channels,
            num_classes=args.num_classes,
            dim=flattened_dim,
            outdim=args.feature_dim,
        )
    elif model_name == "ResNet18":
        model = resnet18(num_classes=args.num_classes)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    elif model_name == "ResNet10":
        model = resnet10(num_classes=args.num_classes)
    elif model_name == "MobileNet":
        model = mobilenet_v2(pretrained=False, num_classes=args.num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(args.device)


def create_server(args, run_index):
    server_class = SERVER_CLASSES[args.algorithm]
    if args.algorithm == "FedDFPA":
        args.head = copy.deepcopy(args.model.fc)
        args.headL = copy.deepcopy(args.model.fc)
        feature_dim = args.model.fc.in_features
        args.model.fc = nn.Identity()
        args.decoder = Decoder(feature_dim, device=args.device)
        args.model = BaseMineSplit(
            args.model, args.head, args.headL, args.decoder,
        ).to(args.device)
    elif args.algorithm == "FedPCL":
        args.model.fc = nn.Identity()
    elif args.algorithm in HEAD_SPLIT_ALGORITHMS:
        args.head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, args.head)
    return server_class(args, run_index)


def run(args):
    time_list = []
    reporter = MemReporter()
    model_name = args.model
    for run_index in range(args.prev, args.times):
        print(f"\n============= Running time: {run_index}th =============")
        print("Creating server and clients ...")
        start = time.time()
        args.model = create_model(model_name, args)
        server = create_server(args, run_index)
        print(args.model)
        server.train()
        time_list.append(time.time() - start)
    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")
    average_data(
        dataset=args.dataset, algorithm=args.algorithm,
        goal=args.goal, times=args.times,
    )
    print("All done!")
    reporter.report()


def init_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-go', "--goal", type=str, default="test", 
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="Cifar10")
    parser.add_argument('-ncl', "--num_classes", type=int, default=10)
    parser.add_argument('-m', "--model", type=str, default="CNN",
                        choices=SUPPORTED_MODELS)
    parser.add_argument('-lbs', "--batch_size", type=int, default=64)
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.005,
                        help="Local learning rate")
    parser.add_argument('-ld', "--learning_rate_decay", type=str2bool, default=False)
    parser.add_argument('-ldg', "--learning_rate_decay_gamma", type=float, default=0.99)
    parser.add_argument('-gr', "--global_rounds", type=int, default=101)
    parser.add_argument('-tc', "--top_cnt", type=int, default=100, 
                        help="For auto_break")
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg",
                        choices=SUPPORTED_ALGORITHMS)
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0,
                        help="Ratio of clients per round")
    parser.add_argument('-rjr', "--random_join_ratio", type=str2bool, default=False,
                        help="Random ratio of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=20,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name", type=str, default='items')
    parser.add_argument('-ab', "--auto_break", type=str2bool, default=False)
    parser.add_argument('-dlg', "--dlg_eval", type=str2bool, default=False)
    parser.add_argument('-dlgg', "--dlg_gap", type=int, default=100)
    parser.add_argument('-bnpc', "--batch_num_per_client", type=int, default=2)
    parser.add_argument('-nnc', "--num_new_clients", type=int, default=0)
    parser.add_argument('-ften', "--fine_tuning_epoch_new", type=int, default=0)
    parser.add_argument('-fd', "--feature_dim", type=int, default=512)
    parser.add_argument('-fs', "--few_shot", type=int, default=0)
    parser.add_argument('-cdr', "--client_drop_rate", type=float, default=0.0,
                        help="Rate for clients that train but drop out")
    parser.add_argument('-tsr', "--train_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument('-ssr', "--send_slow_rate", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")
    parser.add_argument('-ts', "--time_select", type=str2bool, default=False,
                        help="Whether to group and select clients at each round according to time cost")
    parser.add_argument('-tth', "--time_threthold", type=float, default=1000000,
                        help="The threthold for droping slow clients")
    parser.add_argument('-lam', "--lamda", type=float, default=1.0,
                        help="Regularization weight")
    parser.add_argument('-mu', "--mu", type=float, default=0.0)
    parser.add_argument('-pls', "--plocal_epochs", type=int, default=1)
    parser.add_argument('-tau', "--tau", type=float, default=1.0)
    parser.add_argument('-slr', "--server_learning_rate", type=float, default=1.0)
    parser.add_argument('-et', "--eta", type=float, default=1.0)
    parser.add_argument('-s', "--rand_percent", type=int, default=80)
    parser.add_argument('-p', "--layer_idx", type=int, default=2,
                        help="More fine-graind than its original paper.")
    parser.add_argument("--using_reconstructloss", type=str2bool, default=False)
    parser.add_argument("--using_tripletloss", type=str2bool, default=True)
    parser.add_argument("--using_specialloss", type=str2bool, default=True)
    parser.add_argument("--using_glocla", type=str2bool, default=True)
    parser.add_argument("--using_aggregate", type=str2bool, default=False)
    return parser


def main():
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not available.\n")
        args.device = "cpu"
    init_seed(42)
    print("=" * 50)
    for name, value in vars(args).items():
        print(name, '=', value)
    print("=" * 50)
    run(args)


if __name__ == "__main__":
    main()
