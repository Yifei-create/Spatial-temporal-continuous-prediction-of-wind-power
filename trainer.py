import torch
import torch.nn.functional as func
import numpy as np
import os
import os.path as osp
import networkx as nx
from tqdm import tqdm
from torch import optim
from datetime import datetime
from torch_geometric.utils import to_dense_batch
from torch_geometric.loader import DataLoader

from data.dataset import SpatioTemporalDataset
from util.training_utils import cal_metric, masked_mae_np


def mkdirs(path):
    """Create directory"""
    if not os.path.exists(path):
        os.makedirs(path)


def _tensor_stat(logger, name, t):
    """Log basic stats for a tensor and return True if NaN/Inf exists."""
    nan = torch.isnan(t).any().item()
    inf = torch.isinf(t).any().item()
    td = t.detach()
    logger.info(
        f"[STAT] {name}: shape={tuple(td.shape)} nan={nan} inf={inf} "
        f"min={td.min().item():.6g} max={td.max().item():.6g} "
        f"mean={td.mean().item():.6g} std={td.std().item():.6g}"
    )
    return bool(nan or inf)


def load_best_model(args):
    """Load best model from previous period"""
    loss = []
    prev_period_dir = osp.join(args.model_path, args.logname + "-" + str(args.seed), str(args.period - 1))
    for filename in os.listdir(prev_period_dir):
        loss.append(filename[0:-4])
    loss = sorted(loss)
    load_path = osp.join(prev_period_dir, loss[0] + ".pkl")

    args.logger.info("[*] load from {}".format(load_path))
    ckpt = torch.load(load_path, map_location=args.device, weights_only=True)
    state_dict = ckpt["model_state_dict"]

    model = args.methods[args.method](args)

    if args.method in ['EAC', 'ScaleShift', 'VariationalScaleShift']:
        if args.period == args.begin_period:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx in range(args.period - args.begin_period):
                model.expand_adaptive_params(args.graph_size_list[idx])

    model.load_state_dict(state_dict)
    model = model.to(args.device)
    return model, loss[0]


def load_test_best_model(args):
    """Load best model for testing"""
    loss = []
    for filename in os.listdir(osp.join(args.model_path, args.logname + "-" + str(args.seed), str(args.period))):
        loss.append(filename[0:-4])
    loss = sorted(loss)
    load_path = osp.join(args.model_path, args.logname + "-" + str(args.seed), str(args.period), loss[0] + ".pkl")

    args.logger.info("[*] load from {}".format(load_path))
    ckpt = torch.load(load_path, map_location=args.device, weights_only=True)
    state_dict = ckpt["model_state_dict"]

    model = args.methods[args.method](args)

    if args.method in ['EAC', 'ScaleShift', 'VariationalScaleShift']:
        if args.period == args.begin_period:
            model.expand_adaptive_params(args.base_node_size)
        else:
            for idx in range(args.period - args.begin_period):
                model.expand_adaptive_params(args.graph_size_list[idx + 1])

    model.load_state_dict(state_dict)
    model = model.to(args.device)
    return model, loss[0]


def train(inputs, args):
    """Training function - ALL operations on GPU"""
    path = osp.join(args.path, str(args.period))
    mkdirs(path)

    # Verify GPU is being used
    args.logger.info(f"[*] Training on device: {args.device}")
    if torch.cuda.is_available():
        args.logger.info(f"[*] GPU: {torch.cuda.get_device_name(args.device)}")
        args.logger.info(f"[*] GPU Memory: {torch.cuda.get_device_properties(args.device).total_memory / 1024**3:.2f} GB")

    # Loss function
    if args.loss == "mse":
        lossfunc = func.mse_loss
    elif args.loss == "huber":
        lossfunc = func.smooth_l1_loss

    # Data loaders - with pin_memory for faster GPU transfer
    if args.strategy == 'incremental' and args.period > args.begin_period:
        # Incremental learning (subgraph)
        train_loader = DataLoader(
            SpatioTemporalDataset("", "", x=inputs["train_x"][:, :, args.subgraph.numpy()],
                                 y=inputs["train_y"][:, :, args.subgraph.numpy()],
                                 edge_index="", mode="subgraph"),
            batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4
        )
        val_loader = DataLoader(
            SpatioTemporalDataset("", "", x=inputs["val_x"][:, :, args.subgraph.numpy()],
                                 y=inputs["val_y"][:, :, args.subgraph.numpy()],
                                 edge_index="", mode="subgraph"),
            batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=4
        )

        # Build subgraph adjacency matrix
        graph = nx.Graph()
        graph.add_nodes_from(range(args.subgraph.size(0)))
        graph.add_edges_from(args.subgraph_edge_index.numpy().T)
        adj = nx.to_numpy_array(graph)
        adj = adj / (np.sum(adj, 1, keepdims=True) + 1e-6)
        vars(args)["sub_adj"] = torch.from_numpy(adj).to(torch.float).to(args.device)
    else:
        # Full graph training
        train_loader = DataLoader(SpatioTemporalDataset(inputs, "train"),
                                  batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=4)
        val_loader = DataLoader(SpatioTemporalDataset(inputs, "val"),
                               batch_size=args.batch_size, shuffle=False,
                               pin_memory=True, num_workers=4)
        vars(args)["sub_adj"] = vars(args)["adj"]

    test_loader = DataLoader(SpatioTemporalDataset(inputs, "test"),
                            batch_size=args.batch_size, shuffle=False,
                            pin_memory=True, num_workers=4)

    args.logger.info("[*] Period " + str(args.period) + " Dataset load!")

    # Model initialization
    if args.init and args.period > args.begin_period:
        model, _ = load_best_model(args)

        # Freeze backbone parameters
        if args.method in ['EAC', 'ScaleShift', 'VariationalScaleShift']:
            for name, param in model.named_parameters():
                if "gcn1" in name or "tcn1" in name or "gcn2" in name or "fc" in name:
                    param.requires_grad = False

        # Expand parameters
        if args.method in ['EAC', 'ScaleShift', 'VariationalScaleShift']:
            model.expand_adaptive_params(args.graph_size)
    else:
        model = args.methods[args.method](args).to(args.device)
        if args.method in ['EAC', 'ScaleShift', 'VariationalScaleShift']:
            model.expand_adaptive_params(args.graph_size)

    # Verify model is on GPU
    model_device = next(model.parameters()).device
    args.logger.info(f"[*] Model is on device: {model_device}")

    model.count_parameters()

    # Optimizer
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    args.logger.info("[*] Period " + str(args.period) + " Training start")
    lowest_validation_loss = 1e7
    counter = 0
    patience = 10
    model.train()
    use_time = []

    for epoch in range(args.epoch):
        start_time = datetime.now()

        # Training - ALL on GPU
        cn = 0
        training_loss = 0.0
        for batch_idx, data in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                args.logger.info("node number {}".format(data.x.shape))
                args.logger.info(f"[*] Data device before transfer: {data.x.device}")

            # Move data to GPU
            data = data.to(args.device, non_blocking=True)

            if epoch == 0 and batch_idx == 0:
                args.logger.info(f"[*] Data device after transfer: {data.x.device}")
                args.logger.info(f"[*] Adjacency matrix device: {args.sub_adj.device}")
                bad_x = _tensor_stat(args.logger, "train_batch.x", data.x)
                bad_y = _tensor_stat(args.logger, "train_batch.y", data.y)
                if bad_x or bad_y:
                    args.logger.info("[FATAL] NaN/Inf exists in input batch BEFORE forward. Problem is data/npz generation.")
                    return

            optimizer.zero_grad()

            out = model(data, args.sub_adj)

            # Handle variational model output
            if isinstance(out, tuple):
                pred, kl_term = out
            else:
                pred, kl_term = out, None

            if epoch == 0 and batch_idx == 0:
                bad_pred = _tensor_stat(args.logger, "pred", pred)
                if kl_term is not None:
                    args.logger.info(f"[STAT] kl_term: nan={torch.isnan(kl_term).any().item()} inf={torch.isinf(kl_term).any().item()} val={float(kl_term):.6g}")
                    if torch.isnan(kl_term).any().item() or torch.isinf(kl_term).any().item():
                        args.logger.info("[FATAL] NaN/Inf in KL term. Problem is variational scale-shift numerical stability.")
                        return
                if bad_pred:
                    args.logger.info("[FATAL] NaN/Inf appears AFTER forward. Problem is model forward / lr / normalization scale.")
                    return

            # Incremental learning mapping
            if args.strategy == "incremental" and args.period > args.begin_period:
                pred, _ = to_dense_batch(pred, batch=data.batch)
                data.y, _ = to_dense_batch(data.y, batch=data.batch)
                pred = pred[:, args.mapping, :]
                data.y = data.y[:, args.mapping, :]

            loss = lossfunc(data.y, pred, reduction="mean")

            if epoch == 0 and batch_idx == 0:
                if torch.isnan(loss).any().item() or torch.isinf(loss).any().item():
                    args.logger.info("[FATAL] Loss is NaN/Inf. Check pred/y magnitude and data validity.")
                    return

            # Add KL divergence loss
            if kl_term is not None:
                loss = loss + args.kl_weight * kl_term

            training_loss += float(loss)
            cn += 1

            loss.backward()
            optimizer.step()

        if epoch == 0:
            total_time = (datetime.now() - start_time).total_seconds()
        else:
            total_time += (datetime.now() - start_time).total_seconds()
        use_time.append((datetime.now() - start_time).total_seconds())
        training_loss = training_loss / cn

        # Validation
        validation_loss = 0.0
        cn = 0
        model.eval()
        with torch.no_grad():
            for batch_idx, data in enumerate(val_loader):
                data = data.to(args.device, non_blocking=True)
                out = model(data, args.sub_adj)

                if isinstance(out, tuple):
                    pred = out[0]
                else:
                    pred = out

                if args.strategy == "incremental" and args.period > args.begin_period:
                    pred, _ = to_dense_batch(pred, batch=data.batch)
                    data.y, _ = to_dense_batch(data.y, batch=data.batch)
                    pred = pred[:, args.mapping, :]
                    data.y = data.y[:, args.mapping, :]

                loss = masked_mae_np(data.y.cpu().data.numpy(), pred.cpu().data.numpy(), 0)
                validation_loss += float(loss)
                cn += 1
        validation_loss = float(validation_loss / cn)
        model.train()

        args.logger.info(f"epoch:{epoch}, training loss:{training_loss:.4f} validation loss:{validation_loss:.4f}")

        # Early Stopping
        if validation_loss <= lowest_validation_loss:
            counter = 0
            lowest_validation_loss = round(validation_loss, 4)
            torch.save({'model_state_dict': model.state_dict()},
                      osp.join(path, str(round(validation_loss, 4)) + ".pkl"))
        else:
            counter += 1
            if counter > patience:
                break

    best_model_path = osp.join(path, str(lowest_validation_loss) + ".pkl")
    best_model = model
    ckpt = torch.load(best_model_path, map_location=args.device, weights_only=True)
    best_model.load_state_dict(ckpt["model_state_dict"])
    best_model = best_model.to(args.device)

    # Test
    test_model(best_model, args, test_loader, True)

    # ===== CHANGED: do NOT overwrite metrics written by cal_metric() =====
    if args.period not in args.result:
        args.result[args.period] = {}
    args.result[args.period].update({
        "total_time": total_time,
        "average_time": sum(use_time)/len(use_time),
        "epoch_num": epoch + 1
    })
    # ================================================================

    args.logger.info("Finished optimization, total time:{:.2f} s, best model:{}".format(total_time, best_model_path))


def test_model(model, args, testset, pin_memory):
    model.eval()
    pred_ = []
    truth_ = []
    loss = 0.0

    with torch.no_grad():
        cn = 0
        for data in testset:
            data = data.to(args.device, non_blocking=pin_memory)
            out = model(data, args.adj)

            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out

            pred, _ = to_dense_batch(pred, batch=data.batch)
            data.y, _ = to_dense_batch(data.y, batch=data.batch)

            loss += func.mse_loss(data.y, pred, reduction="mean")

            pred_.append(pred.cpu().data.numpy())
            truth_.append(data.y.cpu().data.numpy())
            cn += 1

        loss = loss / cn
        args.logger.info("[*] loss:{:.4f}".format(loss))
        pred_ = np.concatenate(pred_, 0)
        truth_ = np.concatenate(truth_, 0)
        
        cal_metric(truth_, pred_, args)
        
        results_dir = osp.join(args.path, "results")
        mkdirs(results_dir)
        save_path = osp.join(results_dir, f"period_{args.period}_predictions.npz")
        np.savez(save_path, predictions=pred_, ground_truth=truth_)
        args.logger.info(f"[*] Saved predictions and ground truth to: {save_path}")
        args.logger.info(f"[*] Data shape - predictions: {pred_.shape}, ground_truth: {truth_.shape}")
