#!/usr/bin/env python3

import argparse, os, sys, time, shutil, tqdm
import warnings, json, gzip
import numpy as np
from sklearn.model_selection import GroupKFold

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Subset

import matplotlib.pyplot as plt

import epi_models
import epi_dataset
import misc_utils


import functools
from sklearn import metrics

import pandas as pd 
print = functools.partial(print, flush=True)

def make_directory(in_dir):
    if os.path.isfile(in_dir):
        warnings.warn("{} is a regular file".format(in_dir))
        return None
    outdir = in_dir.rstrip('/')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    return outdir

def model_summary(model):
    """
    model: pytorch model
    """
    import torch
    total_param = 0
    trainable_param = 0
    for i, p in enumerate(model.parameters()):
        num_p = torch.numel(p)
        if p.requires_grad:
            trainable_param += num_p
        total_param += num_p
    return {'total_param': total_param, 'trainable_param': trainable_param}




def predict(model: nn.Module, data_loader: DataLoader, device=torch.device('cuda')):
    model.eval()
    result, true_label, result_dist, true_dist = list(), list(), list(), list()
    for feats, dists, enh_idxs, prom_idxs, labels in data_loader:
        feats, dists, labels = feats.to(device), dists.to(device), labels.to(device)
        # enh_idxs, prom_idxs = feats.to(device), prom_idxs.to(device)
        pred, pred_dist, att = model(feats, return_att=True, enh_idx=enh_idxs, prom_idx=prom_idxs)
        del att
        pred = pred.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        pred_dist = pred_dist.detach().cpu().numpy()
        dists = dists.detach().cpu().numpy()
        result.append(pred)
        true_label.append(labels)
        result_dist.append(pred_dist)
        true_dist.append(dists)
    result = np.concatenate(result, axis=0)
    true_label = np.concatenate(true_label, axis=0)
    result_dist = np.concatenate(result_dist, axis=0)
    true_dist = np.concatenate(true_dist, axis=0)
    return (result.squeeze(), true_label.squeeze(), result_dist.squeeze(), true_dist.squeeze())


def train_transformer_model(
        model_class, model_params, 
        optimizer_class, optimizer_params, 
        dataset, groups, n_folds, 
        num_epoch, patience, batch_size, num_workers,
        outdir, checkpoint_prefix, device, 
        train_chroms, valid_chroms, test_chroms,
        use_scheduler=False,
        use_mse=True) -> nn.Module:
    bce_loss = nn.BCELoss() # binary cross entropy?
    mse_loss = nn.MSELoss() # 平均二乗誤差？
    # splitter = GroupKFold(n_splits=n_folds) # 染色体単位にする準備

    wait = 0
    best_epoch, best_val_auc, best_val_aupr = -999, -999, -999
    epoch_results = {"AUC": list(), "AUPR": list()}


    loss_dict = {"epochs": [], "fold": [], "train_loss": [], "valid_loss": []}
    learning_rates = [] 
    train_idx, valid_idx, test_idx = [], [], []
    for epoch_idx in range(num_epoch):

        epoch_results["AUC"] = 0
        epoch_results["AUPR"] = 0
        if epoch_idx == 0:
            print("Fold splits(validation): ")
            # for fold_idx, (train_idx, valid_idx) in enumerate(splitter.split(X=groups, groups=groups)): # 染色体単位で分けてくれる
            #     print("  - Fold{}: validation size:{}({}) training size:{}({})".format(fold_idx, len(valid_idx), misc_utils.count_unique_itmes(groups[valid_idx]), len(train_idx), misc_utils.count_unique_itmes(groups[train_idx])))
            for idx, chrom in enumerate(groups):
                if chrom in train_chroms:
                    train_idx.append(idx)
                elif chrom in valid_chroms:
                    valid_idx.append(idx)
                elif chrom in test_chroms:
                    test_idx.append(idx)
            print("  - validation size:{}({}) training size:{}({})".format(len(valid_idx), misc_utils.count_unique_itmes(groups[valid_idx]), len(train_idx), misc_utils.count_unique_itmes(groups[train_idx])))

                

        print("\nCV epoch: {}/{}\t({})".format(epoch_idx, num_epoch, time.asctime()))
        
        # ___holdout___

        with open(os.path.join(outdir, "log.txt"), "a") as f:
            print("  epochs{}: validation size:{}({}) training size:{}({})".format(epoch_idx, len(valid_idx), misc_utils.count_unique_itmes(groups[valid_idx]), len(train_idx), misc_utils.count_unique_itmes(groups[train_idx])), file=f)

        train_loader = DataLoader(Subset(dataset, indices=train_idx), shuffle=True, batch_size=batch_size, num_workers=num_workers)
        sample_idx = np.random.permutation(train_idx)[0:1024] # train から ランダムに1024個抽出
        sample_loader = DataLoader(Subset(dataset, indices=sample_idx), shuffle=False, batch_size=batch_size, num_workers=num_workers)
        valid_loader = DataLoader(Subset(dataset, indices=valid_idx), shuffle=False, batch_size=batch_size, num_workers=num_workers)
        checkpoint = "{}/{}.pt".format(outdir, checkpoint_prefix)
        if epoch_idx == 0:
            model = model_class(**model_params).to(device)
            optimizer = optimizer_class(model.parameters(), **optimizer_params)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2) # TODO 改善の余地あり
            if os.path.exists(checkpoint):
                os.remove(checkpoint)
        else:
            state_dict = torch.load(checkpoint)
            model.load_state_dict(state_dict["model_state_dict"])
            optimizer.load_state_dict(state_dict["optimizer_state_dict"])
            scheduler.load_state_dict(state_dict["scheduler_state_dict"])

        model.train()
        for feats, dists, enh_idxs, prom_idxs, labels in tqdm.tqdm(train_loader): # train...

            feats, dists, labels = feats.to(device), dists.to(device), labels.to(device) # それぞれbatch_size分のtensor
            if hasattr(model, "att_C"):
                pred, pred_dists, att = model(feats, return_att=True, enh_idx=enh_idxs, prom_idx=prom_idxs)
                # pred = model(feats, enh_idx=enh_idxs, prom_idx=prom_idxs)
                attT = att.transpose(1, 2)
                identity = torch.eye(att.size(1)).to(device)
                identity = Variable(identity.unsqueeze(0).expand(labels.size(0), att.size(1), att.size(1)))
                penal = model.l2_matrix_norm(torch.matmul(att, attT) - identity)

                # edit loss...
                loss = None
                if use_mse == True:
                    loss = bce_loss(pred, labels) + (model.att_C * penal / labels.size(0)).type(torch.cuda.FloatTensor) + mse_loss(dists, pred_dists)
                else:
                    loss = bce_loss(pred, labels) + (model.att_C * penal / labels.size(0)).type(torch.cuda.FloatTensor)

                del penal, identity
            else:
                pred = model(feats, enh_idx=enh_idxs, prom_idx=prom_idxs)
                loss = bce_loss(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step() # 重みの更新 validation は使っていない

            # if use_scheduler: # learning rateの更新（batch毎）
            #     with open(os.path.join(outdir, "learning_rate.txt"), "a") as f:
            #         print('epoch:{}, lr:{}'.format(epoch_idx, scheduler.get_last_lr()[0]), file=f)  
            #     scheduler.step()

            # batch処理終了

        # 次のepochのための準備
        learning_rates.append(optimizer.param_groups[0]["lr"])
        if use_scheduler:
            scheduler.step()

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            }, checkpoint)

        # ___以下追加（全てのepochsを保存）___
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict()
            }, f"{outdir}\\epoch{epoch_idx}.pt")
        # ______

        model.eval()
        train_loss, valid_loss = None, None
        train_pred, train_true, train_pred_dist, train_true_dist = predict(model, sample_loader)
        tra_AUC, tra_AUPR, tra_F1, tra_pre, tra_rec, tra_MCC = misc_utils.evaluator(train_true, train_pred, out_keys=["AUC", "AUPR", "F1", "precision", "recall", "MCC"])
        valid_pred, valid_true, valid_pred_dist, valid_true_dist = predict(model, valid_loader)
        val_AUC, val_AUPR, val_F1, val_pre, val_rec, val_MCC = misc_utils.evaluator(valid_true, valid_pred, out_keys=["AUC", "AUPR", "F1", "precision", "recall", "MCC"])
        if use_mse == True:
            train_loss = metrics.log_loss(train_true, train_pred) + metrics.mean_squared_error(train_true_dist, train_pred_dist)
            valid_loss = metrics.log_loss(valid_true, valid_pred) + metrics.mean_squared_error(valid_true_dist, valid_pred_dist)
        else:
            train_loss = metrics.log_loss(train_true, train_pred)
            valid_loss = metrics.log_loss(valid_true, valid_pred)

        log_tra_text = f"  - train...\nloss={train_loss:.4f}\tAUC={tra_AUC:.4f}\tAUPR={tra_AUPR:.4f}\tF1={tra_F1:.4f}\tpre={tra_pre:.4f}\trec={tra_rec:.4f}\tMCC={tra_MCC:.4f}\t"
        log_val_text = f"  - valid...\nloss={valid_loss:.4f}\tAUC={val_AUC:.4f}\tAUPR={val_AUPR:.4f}\tF1={val_F1:.4f}\tpre={val_pre:.4f}\trec={val_rec:.4f}\tMCC={val_MCC:.4f}\t"
        #print("  - Fold{}:train(AUC/AUPR)/vald(AUC/AUPR):\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t({})".format(fold_idx, tra_AUC, tra_AUPR, val_AUC, val_AUPR, time.asctime()))
        print(log_tra_text)
        print(log_val_text)


        with open(os.path.join(outdir, "log.txt"), "a") as f:
            print(f"___epochs{epoch_idx}___", file=f)
            print(log_tra_text, file=f)
            print(log_val_text, file=f)


        epoch_results["AUC"] = val_AUC
        epoch_results["AUPR"] = val_AUPR

        loss_dict["epochs"].append(epoch_idx)
        loss_dict["train_loss"].append(train_loss)
        loss_dict["valid_loss"].append(valid_loss)


        auc_mean, auc_std = epoch_results["AUC"], epoch_results["AUC"]
        aupr_mean, aupr_std = epoch_results["AUPR"], epoch_results["AUPR"]
        print("Epoch{:03d}(AUC/AUPR):\t{:.4f}({:.4f})\t{:.4f}({:.4f})".format(epoch_idx, auc_mean, auc_std, aupr_mean, aupr_std))



        # 良いepochを保存 (上回らないと保存されない)
        if auc_mean >= best_val_auc and aupr_mean >= best_val_aupr:
            wait = 0
            best_epoch, best_val_auc, best_val_aupr = epoch_idx, auc_mean, aupr_mean
            print("Best epoch {}\t({})".format(best_epoch, time.asctime()))
        else:
            wait += 1
            if wait >= patience:
                print("Early stopped ({})".format(time.asctime()))
                print("Best epoch/AUC/AUPR: {}\t{:.4f}\t{:.4f}".format(best_epoch, best_val_auc, best_val_aupr))
                break
            else:
                print("Wait{} ({})".format(wait, time.asctime()))

    # ___結果書き込み___

    with open(os.path.join(outdir, "result.txt"), "a") as f:
        print(f"best epochs:{best_epoch}, best AUC: {best_val_auc}, best AUPR: {best_val_aupr}", file=f)

    loss_df = pd.DataFrame.from_dict(loss_dict, orient='index').T
    loss_df.to_csv(os.path.join(outdir, "loss.csv"), index=False)

    plt.figure()
    plt.plot(learning_rates)
    plt.xlabel("epochs")
    plt.ylabel("learning rate")
    plt.savefig(os.path.join(outdir, "learning_rate.png"))

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-c', "--config", required=True, help="Configuration file for training the model")
    p.add_argument('-o', "--outdir", required=True, help="Output directory")
    p.add_argument('--gpu', default=-1, type=int, help="GPU ID, (-1 for CPU)")
    p.add_argument('--seed', type=int, default=2020, help="Random seed")

    # 以下追加
    p.add_argument('--use_mse', action="store_true")
    # ___
    return p


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    config = json.load(open(args.config))

    print(config["data_opts"])
    all_data = epi_dataset.EPIDataset(**config["data_opts"])

    config["model_opts"]["in_dim"] = all_data.feat_dim
    config["model_opts"]["seq_len"] = config["data_opts"]["seq_len"] // config["data_opts"]["bin_size"]

    print("##{}".format(time.asctime()))
    print("##command: {}".format(' '.join(sys.argv)))
    print("##config: {}".format(config))
    print("##sample size: {}".format(len(all_data)))
    torch.save(all_data.__getitem__(0), "tmp.pt")
    # print("##feature: {}".format(all_data.__getitem__(0)[0].size())) #squeeze(0).mean(dim=1)))
    # print("##feature: {}".format(all_data.__getitem__(0)[0].mean(dim=1)))
    # print("##feature: {}".format(all_data.__getitem__(0)[0][:, 2400:2600].mean(dim=0)))

    chroms = all_data.metainfo["chrom"]


    if args.gpu >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device('cpu')

    model_class = getattr(epi_models, config["model_opts"]["model"])
    model = model_class(**config["model_opts"])

    # print(model)
    # print(model_summary(model))
    del model
    optimizer_params = {'lr': config["train_opts"]["learning_rate"], 'weight_decay': 1e-8}

    # ___追加___
    if config["train_opts"]["use_scheduler"] == False:
        args.outdir += "_noScheduler"
    
    args.outdir += f"(lr={config['train_opts']['learning_rate']})"

    if args.use_mse == False:
        args.outdir += "_noMSE"
    # ___

    if not os.path.isdir(args.outdir):
        args.outdir = make_directory(args.outdir)

    train_transformer_model(
            model_class=model_class, 
            model_params=config["model_opts"],
            optimizer_class=torch.optim.Adam, 
            optimizer_params=optimizer_params,
            dataset=all_data,
            groups=all_data.metainfo["chrom"],
            n_folds=5,
            num_epoch=config["train_opts"]["num_epoch"], 
            patience=config["train_opts"]["patience"], 
            batch_size=config["train_opts"]["batch_size"], 
            num_workers=config["train_opts"]["num_workers"],
            outdir=args.outdir, 
            checkpoint_prefix="checkpoint",
            device=device,
            train_chroms=["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr10", "chr11", "chr12", "chr13", "chr14", "chr15"],
            valid_chroms=["chr16", "chr17", "chr18"],
            test_chroms=["chr19", "chr20", "chr21", "chr22", "chrX"],
            use_scheduler=config["train_opts"]["use_scheduler"],
            use_mse = args.use_mse
        )

