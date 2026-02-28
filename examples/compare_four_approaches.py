# examples/compare_four_approaches.py
# SASS VERSION: Pure stepsize adaptation based on Armijo condition

import os
import sys
import time
import argparse
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

import appfl.run_serial as rs
import appfl.run_serial_adapt as rs_adapt

from appfl.config import Config
from appfl.misc.utils import set_seed
from omegaconf import OmegaConf

from dataloader.mnist_dataloader import get_mnist
from dataloader.cifar10_dataloader import get_cifar10
from dataloader.cifar100_dataloader import get_cifar100
from models.utils import get_model
from losses.utils import get_loss
from metric.utils import get_metric

# All outputs go here (absolute path, outside examples/)
OUTPUT_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs_sass")


def analyze_data_distribution(train_sets, output_dir, num_classes=10):
    num_clients = len(train_sets)
    distribution = np.zeros((num_clients, num_classes), dtype=int)
    for cid, dataset in enumerate(train_sets):
        labels = dataset.data_label.numpy() if hasattr(dataset.data_label, 'numpy') else dataset.data_label
        unique, counts = np.unique(labels, return_counts=True)
        for label, count in zip(unique, counts):
            distribution[cid, int(label)] = count
    columns = [f'Class_{i}' for i in range(num_classes)]
    df = pd.DataFrame(distribution, columns=columns)
    df.index = [f'Client_{i}' for i in range(num_clients)]
    df['Total'] = df.sum(axis=1)
    totals = df.sum(axis=0); totals.name = 'Total'
    df = pd.concat([df, pd.DataFrame(totals).T])
    csv_path = os.path.join(output_dir, 'data_distribution.csv')
    df.to_csv(csv_path)
    print(f"\nData distribution saved to: {csv_path}")
    print(df.to_string())
    return df


def plot_learning_rates(output_dir, num_clients):
    plt.figure(figsize=(12, 6))
    found = False
    for cid in range(num_clients):
        log_file = os.path.join(output_dir, f'result_client_{cid}.txt')
        if not os.path.exists(log_file):
            continue
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
            if len(lines) < 2:
                continue
            header = lines[0].strip().split()
            lr_col = next((c for c in ('StepSize', 'LearningRate') if c in header), None)
            if lr_col is None:
                continue
            lr_idx = header.index(lr_col)
            round_idx = header.index('Round') if 'Round' in header else 0
            rounds, lrs = [], []
            for line in lines[1:]:
                parts = line.strip().split()
                if len(parts) > max(lr_idx, round_idx):
                    try:
                        rounds.append(float(parts[round_idx]))
                        lrs.append(float(parts[lr_idx]))
                    except (ValueError, IndexError):
                        continue
            if rounds:
                plt.plot(rounds, lrs, marker='o', label=f'Client {cid}', linewidth=2, markersize=4)
                found = True
        except Exception:
            continue
    if found:
        plt.xlabel('Round'); plt.ylabel('Learning Rate')
        plt.title('Learning Rate Evolution per Client', fontweight='bold')
        plt.legend(loc='best'); plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'learning_rate_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()


def make_args(dataset, num_clients, server_epochs, local_epochs, server_lr=None, seed=42):
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
    args.loss_fn = None; args.loss_fn_name = None
    args.metric = "metric/acc.py"; args.metric_name = None; args.dr_metrics = None
    args.use_dp = False; args.epsilon = 1.0
    args.clip_grad = False; args.clip_value = 1.0; args.clip_norm = 1

    if dataset == "MNIST":
        args.model = "CNN"; args.num_channel = 1; args.num_classes = 10; args.num_pixel = 28
    elif dataset == "CIFAR10":
        args.model = "resnet18"; args.num_channel = 3; args.num_classes = 10; args.num_pixel = 32
        args.train_data_batch_size = 128; args.test_data_batch_size = 128
    elif dataset == "CIFAR100":
        args.model = "resnet18"; args.num_channel = 3; args.num_classes = 100; args.num_pixel = 32
        args.train_data_batch_size = 128; args.test_data_batch_size = 128
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return args


def build_cfg(args, server_name, client_name, output_dir, is_adapt, gamma=1.2, max_lr=0.5):
    cfg = OmegaConf.structured(Config)
    cfg.device = args.device; cfg.device_server = args.device
    cfg.reproduce = True
    if cfg.reproduce:
        set_seed(1)
    if hasattr(args, "train_data_batch_size"):
        cfg.train_data_batch_size = args.train_data_batch_size
        cfg.test_data_batch_size = args.test_data_batch_size
        cfg.train_data_shuffle = True
    cfg.num_clients = args.num_clients; cfg.num_epochs = args.num_epochs
    cfg.fed.servername = server_name; cfg.fed.clientname = client_name
    cfg.fed.args.optim = args.client_optimizer
    cfg.fed.args.optim_args.lr = args.client_lr
    cfg.fed.args.num_local_epochs = args.num_local_epochs
    cfg.fed.args.num_local_steps = args.num_local_steps
    if is_adapt and args.server_lr is not None:
        cfg.fed.args.server_lr = args.server_lr
        cfg.fed.args.gamma = gamma
        cfg.fed.args.gamma_decr = 0.7
        cfg.fed.args.theta = 0.2
        cfg.fed.args.eps_f = 0.0
        cfg.fed.args.max_lr = max_lr
        cfg.fed.args.min_lr = 1e-8
    if "GM" in server_name:
        cfg.fed.args.gm_eps = 1e-6; cfg.fed.args.gm_maxiter = 100; cfg.fed.args.gm_ftol = 1e-10
    cfg.use_tensorboard = False; cfg.save_model_state_dict = False
    cfg.output_dirname = output_dir
    return cfg


def load_data(args, cfg):
    start = time.time()
    loader_fn = {"MNIST": get_mnist, "CIFAR10": get_cifar10, "CIFAR100": get_cifar100}[args.dataset]
    train_sets, test_set = loader_fn(
        None, num_clients=cfg.num_clients, partition=args.partition,
        visualization=False, output_dirname=cfg.output_dirname,
        seed=args.seed, alpha1=args.num_clients, dr_metrics=args.dr_metrics,
    )
    print("Loading time:", time.time() - start)
    return train_sets, test_set


def run_one(dataset, server, client, is_adapt, num_clients, server_epochs, local_epochs,
            server_lr=None, gamma=1.2, max_lr=0.5):
    args = make_args(dataset, num_clients, server_epochs, local_epochs, server_lr, seed=42)

    # Build serialized output folder name
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if is_adapt:
        folder = f"{dataset}_{server}_R{server_epochs}_E{local_epochs}_InitLR{server_lr}_Gamma{gamma}_MaxLR{max_lr}_{ts}"
    else:
        folder = f"{dataset}_{server}_R{server_epochs}_E{local_epochs}_SGD_{ts}"
    outdir = os.path.join(OUTPUT_ROOT, folder)
    os.makedirs(outdir, exist_ok=True)

    cfg = build_cfg(args, server, client, outdir, is_adapt, gamma, max_lr)
    model = get_model(args)
    loss_fn = get_loss(args.loss_fn, args.loss_fn_name)
    metric = get_metric(args.metric, args.metric_name)
    train_sets, test_set = load_data(args, cfg)

    print("\n" + "=" * 110)
    if is_adapt:
        print(f"{dataset} | {server}+{client} | R={server_epochs} E={local_epochs} | "
              f"InitLR={server_lr} | gamma={gamma} | MaxLR={max_lr}")
    else:
        print(f"{dataset} | {server}+{client} | R={server_epochs} E={local_epochs}")
    print("=" * 110)

    analyze_data_distribution(train_sets, outdir, num_classes=args.num_classes)

    if is_adapt:
        rs_adapt.run_serial(cfg, model, loss_fn, train_sets, test_set, dataset, metric)
    else:
        rs.run_serial(cfg, model, loss_fn, train_sets, test_set, dataset, metric)

    plot_learning_rates(outdir, num_clients)
    return float(cfg.logginginfo.BestAccuracy), outdir


def main():
    parser = argparse.ArgumentParser(description="Compare 4 FL approaches with SASS")
    parser.add_argument("--datasets", nargs="+", default=["MNIST"],
                        choices=["MNIST", "CIFAR10", "CIFAR100"])
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=4)
    cli = parser.parse_args()

    approaches = [
        ("FedAvgSGD", "ClientOptim", False),
        ("FedAvgASS", "ClientAdaptOptim", True),
        ("FedGMSGD",  "ClientOptim", False),
        ("FedGMASS",  "ClientAdaptOptim", True),
    ]

    configs = [
        (0.01, 1.1, 0.5),
        (0.01, 1.2, 0.5),
        (0.01, 1.5, 1.0),
    ]

    rows = []
    for dataset in cli.datasets:
        for server, client, is_adapt in approaches:
            if is_adapt:
                for init_lr, gamma, max_lr in configs:
                    print(f"\n\n{'#'*110}")
                    print(f"# {dataset} | {server} | SASS  InitLR={init_lr} gamma={gamma} MaxLR={max_lr}")
                    print(f"{'#'*110}\n")
                    best, outdir = run_one(
                        dataset, server, client, is_adapt, cli.num_clients,
                        cli.rounds, cli.local_epochs, init_lr, gamma, max_lr,
                    )
                    rows.append((dataset, server, cli.local_epochs, init_lr, gamma, max_lr, best, outdir))
            else:
                print(f"\n\n{'#'*110}")
                print(f"# {dataset} | {server} (baseline)")
                print(f"{'#'*110}\n")
                best, outdir = run_one(
                    dataset, server, client, is_adapt, cli.num_clients,
                    cli.rounds, cli.local_epochs,
                )
                rows.append((dataset, server, cli.local_epochs, None, None, None, best, outdir))

    print("\n" + "=" * 120)
    print("FINAL SUMMARY")
    print("=" * 120)
    for row in sorted(rows, key=lambda x: (x[0], x[1], str(x[3]))):
        dataset, server, le, init_lr, gamma, max_lr, best, outdir = row
        if init_lr is not None:
            print(f"{dataset:8s} | {server:10s} | E={le} | InitLR={init_lr:.2f} | "
                  f"gamma={gamma:.1f} | MaxLR={max_lr:.1f} | Best={best:.2f}%")
        else:
            print(f"{dataset:8s} | {server:10s} | E={le} | BASELINE | Best={best:.2f}%")
    print("=" * 120)


if __name__ == "__main__":
    main()
