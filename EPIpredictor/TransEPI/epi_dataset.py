#!/usr/bin/env python3

import argparse, os, sys, time
import warnings, json, gzip
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from misc_utils import hg19_chromsize

import numpy as np

from typing import Dict, List, Union

from functools import partial


def custom_open(fn):
    if fn.endswith("gz"):
        return gzip.open(fn, 'rt')
    else:
        return open(fn, 'rt')


def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #p.add_argument()

    p.add_argument('--seed', type=int, default=2020)
    return p


class EPIDataset(Dataset):
    def __init__(self, 
            datasets: Union[str, List], 
            feats_config: Dict[str, str], 
            feats_order: List[str], 
            seq_len: int=2500000, 
            bin_size: int=500, 
            use_mark: bool=False,
            use_mask: bool=False,
            # mask_neighbor=False,
            # mask_window=False,
            sin_encoding=False,
            rand_shift=False,
            
            **kwargs):
        super(EPIDataset, self).__init__()

        if type(datasets) is str:
            self.datasets = [datasets]
        else:
            self.datasets = datasets

        self.bin_size = int(bin_size)
        if use_mask == False:
            self.seq_len = int(seq_len)
            assert self.seq_len % self.bin_size == 0, "{} / {}".format(self.seq_len, self.bin_size)
            self.num_bins = seq_len // bin_size

        self.feats_order = list(feats_order)
        self.num_feats = len(feats_order)
        self.feats_config = json.load(open(feats_config))
        if "_location" in self.feats_config:
            location =self.feats_config["_location"] 
            del self.feats_config["_location"]
            for cell, assays in self.feats_config.items():
                for a, fn in assays.items():
                    self.feats_config[cell][a] = os.path.join(location, fn)
        else:
            location = os.path.dirname(os.path.abspath(feats_config))
            for cell, assays in self.feats_config.items():
                for a, fn in assays.items():
                    self.feats_config[cell][a] = os.path.join(location, fn)


        self.feats = dict() # cell_name -> feature_name -> chrom > features (array)
        self.chrom_bins = {
                chrom: (length // bin_size) for chrom, length in hg19_chromsize.items()
                }

        self.samples = list()
        self.metainfo = { # 重要！！
                'label': list(), 
                'dist': list(), 
                'chrom': list(), 
                'cell': list(),
                'enh_name': list(),
                'prom_name': list(),
                'shift': list()
                }

        self.sin_encoding = sin_encoding
        self.use_mark = use_mark
        # self.mask_window = mask_window
        # self.mask_neighbor = mask_neighbor
        self.mask_window = use_mask
        self.mask_neighbor = use_mask
        self.rand_shift = rand_shift

        self.load_datasets()
        self.feat_dim = len(self.feats_order) + 1
        if self.use_mark:
            self.feat_dim += 1
        if self.sin_encoding:
            self.feat_dim += 1

    def load_datasets(self):
        for fn in self.datasets:
            with custom_open(fn) as infile:
                for l in infile:
                    fields = l.strip().split(',')[:10]
                    label, dist, chrom, enh_start, enh_end, enh_name, \
                            _, prom_start, prom_end, prom_name = fields[0:10]
                    if label == "label": # skip header
                        continue
                    knock_range = None
                    if len(fields) > 10:
                        assert len(fields) == 11
                        knock_range = list()
                        for knock in fields[10].split(';'):
                            knock_start, knock_end = knock.split('-')
                            knock_start, knock_end = int(knock_start), int(knock_end)
                            knock_range.append((knock_start, knock_end))

                    cell = enh_name.split('|')[0]
                    # append...
                    if cell == "HeLa-S3":
                        cell = "HeLa"
                    # ___

                    enh_coord = (int(enh_start) + int(enh_end)) // 2
                    p_start, p_end = prom_name.split('|')[1].split(':')[-1].split('-')
                    tss_coord = (int(p_start) + int(p_end)) // 2

                    enh_bin = enh_coord // self.bin_size
                    prom_bin = tss_coord // self.bin_size

                    if self.mask_window and self.mask_neighbor:
                        seq_begin, seq_end, start_bin, stop_bin = -1, -1, -1, -1
                    else:
                        seq_begin = (enh_coord + tss_coord) // 2 - self.seq_len // 2
                        seq_end = (enh_coord + tss_coord) // 2 + self.seq_len // 2

                        assert seq_begin <= enh_coord and seq_begin <= tss_coord, f"seq_begin:{seq_begin}, enh_coord:{enh_coord}, tss_coord:{tss_coord}"
                        assert enh_coord < seq_end and tss_coord < seq_end, f"seq_end:{seq_end}, enh_coord:{enh_coord}, tss_coord:{tss_coord}"
                        
                        # enh_bin = (enh_coord - seq_begin) // self.bin_size
                        # # enh_bin = enh_coord // self.bin_size
                        # prom_bin = (tss_coord - seq_begin) // self.bin_size
                        # # prom_bin = tss_coord // self.bin_size
                        start_bin, stop_bin = seq_begin // self.bin_size, seq_end // self.bin_size


                    left_pad_bin, right_pad_bin = 0, 0
                    if start_bin < 0:
                        left_pad_bin = abs(start_bin)
                        start_bin = 0
                    if stop_bin > self.chrom_bins[chrom]: 
                        right_pad_bin = stop_bin - self.chrom_bins[chrom] 
                        stop_bin = self.chrom_bins[chrom]

                    shift = 0
                    if self.rand_shift:
                        if left_pad_bin > 0:
                            shift = left_pad_bin
                            start_bin = -left_pad_bin
                            left_pad_bin = 0
                        elif right_pad_bin > 0:
                            shift = -right_pad_bin
                            stop_bin = self.chrom_bins[chrom] + right_pad_bin
                            right_pad_bin = 0
                        else:
                            min_range = min(min(enh_bin, prom_bin) - start_bin, stop_bin - max(enh_bin, prom_bin))
                            if min_range > (self.num_bins / 4):
                                shift = np.random.randint(-self.num_bins // 5, self.num_bins // 5)
                            if start_bin + shift <= 0 or stop_bin + shift >= self.chrom_bins[chrom]:
                                shift = 0

                    self.samples.append((
                        start_bin + shift, stop_bin + shift, 
                        left_pad_bin, right_pad_bin, 
                        enh_bin, prom_bin, 
                        cell, chrom, # np.log2(1 + 500000 / float(dist)),
                        int(label), knock_range
                    ))
                    # print(l.strip())
                    # print(self.samples[-1])
                    # print(enh_coord, enh_coord // self.bin_size, tss_coord, tss_coord // self.bin_size, seq_begin, seq_begin // self.bin_size, seq_end, seq_end // self.bin_size, start_bin, stop_bin, left_pad_bin, right_pad_bin)

                    self.metainfo['label'].append(int(label))
                    self.metainfo['dist'].append(float(dist))
                    self.metainfo['chrom'].append(chrom)
                    self.metainfo['cell'].append(cell)
                    self.metainfo['enh_name'].append(enh_name)
                    self.metainfo['prom_name'].append(prom_name)
                    self.metainfo['shift'].append(shift)

                    if cell not in self.feats:
                        self.feats[cell] = dict()
                        for feat in self.feats_order:
                            self.feats[cell][feat] = torch.load(self.feats_config[cell][feat])
        for k in self.metainfo:
            self.metainfo[k] = np.array(self.metainfo[k])

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        start_bin, stop_bin, left_pad, right_pad, enh_bin, prom_bin, cell, chrom, label, knock_range = self.samples[idx]
        enh_idx = enh_bin - start_bin + left_pad
        prom_idx = prom_bin - start_bin + left_pad

        # mask用に修正
        if self.mask_neighbor and self.mask_window:
            if prom_idx > enh_idx:
                enh_idx, prom_idx = 12, 37
            elif prom_idx < enh_idx:
                enh_idx, prom_idx = 37, 12
            else:
                enh_idx, prom_idx = 12, 37
        else:
            # 以下を追加 (何故かenh_idxとprom_idxがfeatsからはみ出すことがあるので、修正する)
            if enh_idx == self.seq_len // self.bin_size:
                print(f"enh idx: {enh_idx}, prom idx: {prom_idx}")
                enh_idx -= 1
            if prom_idx == self.seq_len // self.bin_size:
                print(f"enh idx: {enh_idx}, prom idx: {prom_idx}")
                prom_idx -= 1
            # _____


        # print(self.samples[idx], self.metainfo["shift"][idx])
        if self.mask_neighbor and self.mask_window:
            ar = torch.zeros((0, 50)) # enh+-10, prm +-10
        else:
            ar = torch.zeros((0, stop_bin - start_bin))
        # print(start_bin - left_pad, stop_bin + right_pad, enh_bin, prom_bin, enh_idx, prom_idx)
        for feat in self.feats_order:
            if self.mask_neighbor and self.mask_window:
                # enh_feats = self.feats[cell][feat][chrom][enh_bin-10:enh_bin+11].view(1, -1)
                enh_feats = self.feats[cell][feat][chrom][enh_bin-2:enh_bin+3].view(1, -1)
                # print(enh_feats.size())
                enh_feats = torch.cat((torch.zeros(1, 10), enh_feats), dim=1)
                enh_feats = torch.cat((enh_feats, torch.zeros(1, 10)), dim=1)
                # prom_feats = self.feats[cell][feat][chrom][prom_bin-10:prom_bin+11].view(1, -1)
                prom_feats = self.feats[cell][feat][chrom][prom_bin-2:prom_bin+3].view(1, -1)
                prom_feats = torch.cat((torch.zeros(1, 10), prom_feats), dim=1)
                prom_feats = torch.cat((prom_feats, torch.zeros(1, 10)), dim=1)
                assert enh_feats.size() == prom_feats.size()
                # print(enh_feats.size())
                # print(prom_feats.size())
                # print(enh_feats.size(), prom_feats.size())
                # exit()
                cat_feats = torch.cat((enh_feats, prom_feats), dim=1)
                ar = torch.cat((ar, cat_feats), dim=0)
            else:
                ar = torch.cat((ar, self.feats[cell][feat][chrom][start_bin:stop_bin].view(1, -1)), dim=0)

        if self.mask_neighbor and self.mask_window:
            tmp = ""
        else:
            ar = torch.cat((
                torch.zeros((self.num_feats, left_pad)),
                ar, 
                torch.zeros((self.num_feats, right_pad))
                ), dim=1)

        if knock_range is not None: # よくわかんない部分 無視でオッケーか？
            dim, length = ar.size()
            mask = [1 for _ in range(self.num_bins)]
            for knock_start, knock_end in knock_range:
                knock_start = knock_start // self.bin_size - start_bin + left_pad
                knock_end = knock_end // self.bin_size - start_bin + left_pad
                for pos in range(max(0, knock_start), min(knock_end + 1, self.num_bins)):
                    mask[pos] = 0
            mask = np.array(mask, dtype=np.float32).reshape(1, -1)
            mask = np.concatenate([mask for _ in range(dim)], axis=0)
            mask = torch.FloatTensor(mask)
            ar = ar * mask

        if self.mask_neighbor and self.mask_window:
            if prom_bin > enh_bin:
                enh_pos_enc = torch.arange(-12, 13, 1).view(1, -1)
                enh_pos_enc = torch.cat((enh_pos_enc, torch.arange(prom_bin-enh_bin-12, prom_bin-enh_bin+13, 1).view(1, -1)), dim=1)

                prom_pos_enc = torch.arange(enh_bin-prom_bin-12, enh_bin-prom_bin+13, 1).view(1, -1)
                prom_pos_enc = torch.cat((prom_pos_enc, torch.arange(-12, 13, 1).view(1, -1)), dim=1)

                pos_enc = torch.cat((enh_pos_enc, prom_pos_enc), dim=0)
            else:
                prom_pos_enc = torch.arange(-12, 13, 1).view(1, -1)
                prom_pos_enc = torch.cat((prom_pos_enc, torch.arange(enh_bin-prom_bin-12, enh_bin-prom_bin+13, 1).view(1, -1)), dim=1)

                enh_pos_enc = torch.arange(prom_bin-enh_bin-12, prom_bin-enh_bin+13, 1).view(1, -1)
                enh_pos_enc = torch.cat((enh_pos_enc, torch.arange(-12, 13, 1).view(1, -1)), dim=1)

                pos_enc = torch.cat((prom_pos_enc, enh_pos_enc), dim=0)
        else:
            pos_enc = torch.arange(self.num_bins).view(1, -1)
            pos_enc = torch.cat((pos_enc - min(enh_idx, prom_idx), max(enh_idx, prom_idx) - pos_enc), dim=0)

        if self.sin_encoding: # たぶん常にfalse
            pos_enc = torch.sin(pos_enc / 2 / self.num_bins * np.pi).view(2, -1)
        else:
            pos_enc = self.sym_log(pos_enc.min(dim=0)[0]).view(1, -1)
        ar = torch.cat((torch.as_tensor(pos_enc, dtype=torch.float), ar), dim=0)
        

        if self.use_mark:
            mark = [0 for i in range(self.num_bins)]
            mark[enh_idx] = 1
            mark[enh_idx - 1] = 1
            mark[enh_idx + 1] = 1
            mark[prom_idx] = 1
            mark[prom_idx - 1] = 1
            mark[prom_idx + 1] = 1
            ar = torch.cat((
                torch.as_tensor(mark, dtype=torch.float).view(1, -1),
                ar
            ), dim=0)




        return ar, torch.as_tensor([enh_idx], dtype=torch.float), torch.as_tensor([prom_idx], dtype=torch.float), torch.as_tensor([label], dtype=torch.float)

    def sym_log(self, ar):
        sign = torch.sign(ar)
        ar = sign * torch.log10(1 + torch.abs(ar))
        return ar


if __name__ == "__main__":
    p = get_args()
    args = p.parse_args()
    np.random.seed(args.seed)

    all_data = EPIDataset(
            datasets=["../data/BENGI/GM12878.HiC-Benchmark.v3.tsv"],
            feats_config="../data/genomic_features/CTCF_DNase_6histone.500.json",
            feats_order=["CTCF", "DNase", "H3K27ac", "H3K4me1", "H3K4me3"],
            seq_len=2500000,
            bin_size=500,
            mask_window=True,
            mask_neighbor=True,
            sin_encoding=True,
            rand_shift=True
        )

    for i in range(0, len(all_data), 411):
        np.savetxt(
                "data_{}".format(i),
                all_data.__getitem__(i)[0].T,
                fmt="%.4f",
                header="{}\t{}\t{}\n{}".format(all_data.metainfo["label"][i], all_data.metainfo["enh_name"][i], all_data.metainfo["prom_name"][i], all_data.samples[i])
            )


#     batch_size = 16
#     data_loader = DataLoader(all_data, batch_size=batch_size, shuffle=False, num_workers=8)
# 
#     # # import epi_models
#     # # model = epi_models.LstmAttModel(in_dim=6, 
#     # #         lstm_size=32, lstm_layer=2, lstm_dropout=0.2, 
#     # #         da=64, r=32,
#     # #         fc=[64, 32], fc_dropout=0.2)
#     # import epi_models
#     # model = epi_models.PerformerModel(
#     #         in_dim=6,
#     #         cnn_channels=[128],
#     #         cnn_sizes=[11],
#     #         cnn_pool=[5],
#     #         enc_layers=4,
#     #         num_heads=4,
#     #         d_inner=128,
#     #         fc=[32, 16],
#     #         fc_dropout=0.1
#     #     ).cuda()
#     for i, (feat, dist, label) in enumerate(data_loader):
#         print()
#         print(feat.size(), dist.size(), label.size())
#         # torch.save({'feat': feat, 'label': label}, "tmp.pt")
#         # feat = model(feat.cuda())
#         print(feat.size())
#         # for k in all_data.metainfo:
#         #     print(k, all_data.metainfo[k][i])
#         # if i > 200:
#         #     break
# 
