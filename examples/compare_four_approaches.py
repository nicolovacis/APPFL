# examples/compare_four_approaches.py
# SASS VERSION: Pure stepsize adaptation based on Armijo condition

import os
import time
import argparse
import torch

import appfl.run_serial as rs
import appfl.run_serial_adapt as rs_adapt

from appfl.config import Config
from appfl.misc.utils import set_seed
from omegaconf import OmegaConf

from dataloader.mnist_dataloader import get_mnist
from dataloader.cifar10_dataloader import get_cifar10
from models.utils import get_model
from losses.utils import get_loss
from metric.utils import get_metric


def make_args(
    dataset: str,
    num_clients: int,
    server_epochs: int,
    local_epochs: int,
    server_lr: float = None,
    seed: int = 42,
):
    args = argparse.Namespace()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.dataset = dataset
    args.partition = "dirichlet_noiid"
    args.seed = seed

    args.num_clients = num_clients
    args.client_optimizer = "SGD"
    args.client_lr = 0.1
    args.num_local_steps = 100
    args.num_local_epochs = int(local_epochs)

    args.num_epochs = int(server_epochs)
    args.server_lr = float(server_lr) if server_lr is not None else None

    args.loss_fn = None
    args.loss_fn_name = None
    args.metric = "metric/acc.py"
    args.metric_name = None
    args.dr_metrics = None

    args.use_dp = False
    args.epsilon = 1.0
    args.clip_grad = False
    args.clip_value = 1.0
    args.clip_norm = 1

    if dataset == "MNIST":
        args.model = "CNN"
        args.num_channel = 1
        args.num_classes = 10
        args.num_pixel = 28
    elif dataset == "CIFAR10":
        args.model = "resnet18"
        args.num_channel = 3
        args.num_classes = 10
        args.num_pixel = 32
        args.train_data_batch_size = 128
        args.test_data_batch_size = 128
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    return args


def build_cfg(args, server_name: str, client_name: str, output_dir: str, 
              is_adapt: bool, gamma: float = 1.2, max_lr: float = 0.5):
    cfg = OmegaConf.structured(Config)
    cfg.device = args.device
    cfg.device_server = args.device

    cfg.reproduce = True
    if cfg.reproduce:
        set_seed(1)

    if hasattr(args, "train_data_batch_size"):
        cfg.train_data_batch_size = args.train_data_batch_size
        cfg.test_data_batch_size = args.test_data_batch_size
        cfg.train_data_shuffle = True

    cfg.num_clients = args.num_clients
    cfg.num_epochs = args.num_epochs
    cfg.fed.servername = server_name
    cfg.fed.clientname = client_name

    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    cfg.fed.args.num_local_steps = args.num_local_steps

    # SASS parameters for adaptive methods
    if is_adapt and args.server_lr is not None:
        cfg.fed.args.server_lr = args.server_lr
        cfg.fed.args.gamma = gamma
        cfg.fed.args.selection_rule = "per_client"
        cfg.fed.args.max_lr = max_lr
        cfg.fed.args.min_lr = 1e-8
    
    # Geometric Median parameters
    if "GM" in server_name:
        cfg.fed.args.gm_eps = 1e-6
        cfg.fed.args.gm_maxiter = 100
        cfg.fed.args.gm_ftol = 1e-10

    cfg.use_tensorboard = False
    cfg.save_model_state_dict = False
    cfg.output_dirname = output_dir

    return cfg


def load_data(args, cfg):
    start = time.time()
    if args.dataset == "MNIST":
        train_sets, test_set = get_mnist(
            None,
            num_clients=cfg.num_clients,
            partition=args.partition,
            visualization=False,
            output_dirname=cfg.output_dirname,
            seed=args.seed,
            alpha1=args.num_clients,
            dr_metrics=args.dr_metrics,
        )
    else:
        train_sets, test_set = get_cifar10(
            None,
            num_clients=cfg.num_clients,
            partition=args.partition,
            visualization=False,
            output_dirname=cfg.output_dirname,
            seed=args.seed,
            alpha1=args.num_clients,
            dr_metrics=args.dr_metrics,
        )
    print("Loading time:", time.time() - start)
    return train_sets, test_set


def run_one(
    dataset: str,
    server: str,
    client: str,
    is_adapt: bool,
    num_clients: int,
    server_epochs: int,
    local_epochs: int,
    server_lr: float = None,
    gamma: float = 1.2,
    max_lr: float = 0.5,
):
    args = make_args(
        dataset=dataset,
        num_clients=num_clients,
        server_epochs=server_epochs,
        local_epochs=local_epochs,
        server_lr=server_lr,
        seed=42,
    )

    if is_adapt:
        tag = f"E{local_epochs}_InitLR{server_lr}_Gamma{gamma}_MaxLR{max_lr}"
    else:
        tag = f"E{local_epochs}_SGD"
    outdir = os.path.join(".", "outputs_sass", dataset, server, tag)
    os.makedirs(outdir, exist_ok=True)

    cfg = build_cfg(args, server, client, outdir, is_adapt, gamma, max_lr)

    model = get_model(args)
    loss_fn = get_loss(args.loss_fn, args.loss_fn_name)
    metric = get_metric(args.metric, args.metric_name)

    train_sets, test_set = load_data(args, cfg)

    print("\n" + "=" * 110)
    if is_adapt:
        print(f"{dataset} | {server}+{client} | E={local_epochs} | "
              f"InitLR={server_lr} | γ={gamma} | MaxLR={max_lr}")
        print(f"SASS: SUCCESS → α×{gamma}, FAIL → α/{gamma}")
    else:
        print(f"{dataset} | {server}+{client} | E={local_epochs}")
    print("=" * 110)

    if is_adapt:
        rs_adapt.run_serial(cfg, model, loss_fn, train_sets, test_set, dataset, metric)
    else:
        rs.run_serial(cfg, model, loss_fn, train_sets, test_set, dataset, metric)

    return float(cfg.logginginfo.BestAccuracy), outdir


def main():
    datasets = ["MNIST"]
    
    approaches = [
        ("FedAvgSGD", "ClientOptim", False),
        ("FedAvgASS", "ClientAdaptOptim", True),
        ("FedGMSGD",  "ClientOptim", False),
        ("FedGMASS",  "ClientAdaptOptim", True),
    ]

    num_clients = 5
    server_epochs = 10
    local_epochs = 4
    
    # SASS hyperparameters
    # Key: gamma controls how aggressively we adjust stepsizes
    # - gamma=1.1: conservative (10% adjustment)
    # - gamma=1.2: moderate (20% adjustment)
    # - gamma=1.5: aggressive (50% adjustment)
    configs = [
        # (init_lr, gamma, max_lr)
        (0.01, 1.1, 0.5),   # Conservative
        (0.01, 1.2, 0.5),   # Moderate
        (0.01, 1.5, 1.0),   # Aggressive
    ]

    rows = []
    for dataset in datasets:
        for server, client, is_adapt in approaches:
            if is_adapt:
                for init_lr, gamma, max_lr in configs:
                    print(f"\n\n{'#'*110}")
                    print(f"# {dataset} | {server} | SASS")
                    print(f"# InitLR={init_lr}, γ={gamma}, MaxLR={max_lr}")
                    print(f"{'#'*110}\n")
                    
                    best, outdir = run_one(
                        dataset=dataset,
                        server=server,
                        client=client,
                        is_adapt=is_adapt,
                        num_clients=num_clients,
                        server_epochs=server_epochs,
                        local_epochs=local_epochs,
                        server_lr=init_lr,
                        gamma=gamma,
                        max_lr=max_lr,
                    )
                    rows.append((dataset, server, local_epochs, init_lr, gamma, max_lr, best, outdir))
            else:
                print(f"\n\n{'#'*110}")
                print(f"# {dataset} | {server} (baseline)")
                print(f"{'#'*110}\n")
                
                best, outdir = run_one(
                    dataset=dataset,
                    server=server,
                    client=client,
                    is_adapt=is_adapt,
                    num_clients=num_clients,
                    server_epochs=server_epochs,
                    local_epochs=local_epochs,
                )
                rows.append((dataset, server, local_epochs, None, None, None, best, outdir))

    print("\n" + "=" * 120)
    print("FINAL SUMMARY (SASS VERSION)")
    print("=" * 120)
    for row in sorted(rows, key=lambda x: (x[0], x[1], str(x[3]))):
        dataset, server, le, init_lr, gamma, max_lr, best, outdir = row
        if init_lr is not None:
            print(f"{dataset:6s} | {server:10s} | E={le} | InitLR={init_lr:.2f} | "
                  f"γ={gamma:.1f} | MaxLR={max_lr:.1f} | Best={best:.2f}%")
        else:
            print(f"{dataset:6s} | {server:10s} | E={le} | BASELINE | Best={best:.2f}%")
    print("=" * 120)


if __name__ == "__main__":
    main()