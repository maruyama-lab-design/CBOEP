import pandas as pd 
import numpy as np 
import os 
import argparse
import random
import math
import matplotlib.pyplot as plt

from tqdm import tqdm


def get_distance(enh, prm):
	# region name must be "GM12878|chr16:88874-88924"
	enh_start, enh_end = map(int, enh.split(":")[1].split("-"))
	prm_start, prm_end = map(int, prm.split(":")[1].split("-"))
	enh_pos = (enh_start + enh_end) / 2
	prm_pos = (prm_start + prm_end) / 2
	return abs(enh_pos - prm_pos)


def get_all_neg(pos_df, dmax):
	enhs, prms = set(pos_df["enhancer_name"].to_list()), set(pos_df["promoter_name"].to_list())
	all_neg = {}
	for enh in tqdm(enhs):
		for prm in prms:
			dist = get_distance(enh, prm)
			if dist > dmax:
				continue
			assert f"{enh}={prm}" not in all_neg
			all_neg[f"{enh}={prm}"] = 1
	
	for _, row in pos_df.iterrows():
		enh, prm = row["enhancer_name"], row["promoter_name"] 
		all_neg.pop(f"{enh}={prm}")
	
	return list(all_neg.keys())


def get_region_freq_dict(pos_df, now_neg):
	enh_freq, prm_freq = {}, {}
	for _, row in pos_df.iterrows():
		enh = row["enhancer_name"]
		prm = row["promoter_name"]

		if enh in enh_freq:
			enh_freq[enh]["+"] += 1
		else:
			enh_freq[enh] = {
				"+": 1,
				"-": 0
			}
			
		if prm in prm_freq:
			prm_freq[prm]["+"] += 1
		else:
			prm_freq[prm] = {
				"+": 1,
				"-": 0
			}
	for neg in now_neg.keys():
		enh, prm = neg.split("=")
		enh_freq[enh]["-"] += 1
		prm_freq[prm]["-"] += 1

	return enh_freq, prm_freq

def get_ClassBalance(pos_freq, neg_freq, alpha):
	cb = min(alpha*pos_freq, neg_freq) / max(alpha*pos_freq, neg_freq)
	return cb

def get_neg_df(now_neg):
	labels = [0] * len(now_neg)
	enhs = [x.split("=")[0] for x in now_neg.keys()]
	prms = [x.split("=")[1] for x in now_neg.keys()]
	enh_chorms = [x.split("|")[1].split(":")[0] for x in enhs]
	prm_chorms = [x.split("|")[1].split(":")[0] for x in prms]
	enh_starts = [int(x.split("|")[1].split(":")[1].split("-")[0]) for x in enhs]
	enh_ends = [int(x.split("|")[1].split(":")[1].split("-")[1]) for x in enhs]
	prm_starts = [int(x.split("|")[1].split(":")[1].split("-")[0]) for x in prms]
	prm_ends = [int(x.split("|")[1].split(":")[1].split("-")[1]) for x in prms]
	dists = [get_distance(enh, prm) for (enh, prm) in zip(enhs, prms)]
	neg_df = pd.DataFrame(
		columns=["label", "enhancer_distance_to_promoter",
		"enhancer_chrom", "enhancer_start", "enhancer_end", "enhancer_name",
		"promoter_chrom", "promoter_start", "promoter_end", "promoter_name"
		]
	)
	neg_df["label"], neg_df["enhancer_distance_to_promoter"] = labels, dists
	neg_df["enhancer_chrom"], neg_df["enhancer_start"], neg_df["enhancer_end"], neg_df["enhancer_name"] = enh_chorms, enh_starts, enh_ends, enhs
	neg_df["promoter_chrom"], neg_df["promoter_start"], neg_df["promoter_end"], neg_df["promoter_name"] = prm_chorms, prm_starts, prm_ends, prms
	return neg_df

def GibbsSampling_ratio(args):
	org_df = pd.read_csv(os.path.join(args.indir, f"{args.cell}.csv"))
	for chrom, org_df_by_chrom in org_df.groupby("enhancer_chrom"):
		print(f"_____{chrom}_____")
		pos_df = org_df_by_chrom[org_df_by_chrom["label"]==1]

		print("preprocess...")
		all_neg = get_all_neg(pos_df, args.dmax) # 全負例候補
		print("choose initial negative randomly...")
		np.random.shuffle(all_neg)
		# now_neg = dict(zip(all_neg[:len(pos_df)], [1]*len(pos_df))) # 選択されている負例
		now_neg = {}

		print("get initial freq^+/- of every enh/prm...")
		enh_freq, prm_freq = get_region_freq_dict(pos_df, now_neg)
		print("calcurate initial class balance")

		region_cnt, sum_cb = 0, 0
		for enh in list(enh_freq.keys()):
			sum_cb += get_ClassBalance(enh_freq[enh]["+"], enh_freq[enh]["-"], args.alpha)
			region_cnt += 1
		for prm in list(prm_freq.keys()):
			sum_cb += get_ClassBalance(prm_freq[prm]["+"], prm_freq[prm]["-"], args.alpha)
			region_cnt += 1

		all_cb = np.zeros((args.T+1))
		all_cb[0] = sum_cb
		for t in tqdm(range(args.T)):
			# choose one pair randomly
			cand_neg = random.choice(all_neg)
			cand_enh, cand_prm = cand_neg.split("=")
			if now_neg.get(cand_neg) != None: # remove if it is already negative
				now_neg.pop(cand_neg)
				sum_cb -= get_ClassBalance(enh_freq[cand_enh]["+"], enh_freq[cand_enh]["-"], args.alpha) + get_ClassBalance(prm_freq[cand_prm]["+"], prm_freq[cand_prm]["-"], args.alpha)
				enh_freq[cand_enh]["-"] -= 1
				prm_freq[cand_prm]["-"] -= 1
				sum_cb += get_ClassBalance(enh_freq[cand_enh]["+"], enh_freq[cand_enh]["-"], args.alpha) + get_ClassBalance(prm_freq[cand_prm]["+"], prm_freq[cand_prm]["-"], args.alpha)

			# score if it is not selected as negative
			score_0 = math.exp(get_ClassBalance(enh_freq[cand_enh]["+"], enh_freq[cand_enh]["-"], args.alpha) + get_ClassBalance(prm_freq[cand_prm]["+"], prm_freq[cand_prm]["-"], args.alpha))

			# score if it is selected as negative
			score_1 = math.exp(get_ClassBalance(enh_freq[cand_enh]["+"], enh_freq[cand_enh]["-"]+1, args.alpha) + get_ClassBalance(prm_freq[cand_prm]["+"], prm_freq[cand_prm]["-"]+1, args.alpha))
			
			# calcurate prob
			prob = score_1 / (score_0 + score_1)

			# pick as negative with prob
			if random.random() < prob:
				now_neg[cand_neg] = 1
				sum_cb -= get_ClassBalance(enh_freq[cand_enh]["+"], enh_freq[cand_enh]["-"], args.alpha) + get_ClassBalance(prm_freq[cand_prm]["+"], prm_freq[cand_prm]["-"], args.alpha)
				enh_freq[cand_enh]["-"] += 1
				prm_freq[cand_prm]["-"] += 1
				sum_cb += get_ClassBalance(enh_freq[cand_enh]["+"], enh_freq[cand_enh]["-"], args.alpha) + get_ClassBalance(prm_freq[cand_prm]["+"], prm_freq[cand_prm]["-"], args.alpha)
			
			# print(f"prob:{prob}")
			# print(f"now neg {len(now_neg)}")
			all_cb[t+1] = sum_cb

		plt.figure()
		plt.plot(range(args.T+1), all_cb/region_cnt)
		plt.ylim((0, 1))
		# plt.xlabel("t")
		# plt.ylabel("f($D^{-}$)")
		plt.title(f"{args.cell} {chrom}")
		plt.savefig(os.path.join(args.outdir, f"{args.cell}_{chrom}.png"))
		
		neg_df = get_neg_df(now_neg=now_neg)
		new_df = pd.concat([pos_df, neg_df], axis=0)
		new_df.to_csv(os.path.join(args.outdir, f"{args.cell}_{chrom}.csv"), index=False)


def GibbsSampling_mse(args):
	org_df = pd.read_csv(os.path.join(args.indir, f"{args.cell}.csv"))
	neg_df = pd.DataFrame()
	plt.figure()
	for chrom, org_df_by_chrom in org_df.groupby("enhancer_chrom"):
		print(f"_____{chrom}_____")
		pos_df = org_df_by_chrom[org_df_by_chrom["label"]==1]

		print("preprocess...")
		all_neg = get_all_neg(pos_df, args.dmax) # 全負例候補
		if len(all_neg) == 0:
			continue
		print("choose initial negative randomly...")
		np.random.shuffle(all_neg)
		now_neg = dict(zip(all_neg[:int(len(pos_df)*args.alpha)], [1]*int(len(pos_df)*args.alpha))) # 選択されている負例
		# now_neg = {}

		print("get initial freq^+/- of every enh/prm...")
		enh_freq, prm_freq = get_region_freq_dict(pos_df, now_neg)
		print("calcurate initial class balance")

		region_cnt, sum_cb = 0, 0
		for enh in list(enh_freq.keys()):
			sum_cb += (args.alpha * enh_freq[enh]["+"] - enh_freq[enh]["-"]) ** 2
			region_cnt += 1
		for prm in list(prm_freq.keys()):
			sum_cb += (args.alpha * prm_freq[prm]["+"] - prm_freq[prm]["-"]) ** 2
			region_cnt += 1

		all_cb = np.zeros((args.T+1))
		all_cb[0] = sum_cb
		for t in tqdm(range(args.T)):
			# 一つランダムに選ぶ
			cand_neg = random.choice(all_neg)
			cand_enh, cand_prm = cand_neg.split("=")
			if now_neg.get(cand_neg) != None: # すでに負例の場合は一旦取り除く
				now_neg.pop(cand_neg)
				sum_cb -= (args.alpha * enh_freq[cand_enh]["+"] - enh_freq[cand_enh]["-"]) ** 2
				sum_cb -= (args.alpha * prm_freq[cand_prm]["+"] - prm_freq[cand_prm]["-"]) ** 2
				enh_freq[cand_enh]["-"] -= 1
				prm_freq[cand_prm]["-"] -= 1
				sum_cb += (args.alpha * enh_freq[cand_enh]["+"] - enh_freq[cand_enh]["-"]) ** 2
				sum_cb += (args.alpha * prm_freq[cand_prm]["+"] - prm_freq[cand_prm]["-"]) ** 2

			# 負例としない場合のcb
			score_0 = math.exp(-((args.alpha * enh_freq[cand_enh]["+"] - enh_freq[cand_enh]["-"]) ** 2)) * math.exp(-((args.alpha * prm_freq[cand_prm]["+"] - prm_freq[cand_prm]["-"]) ** 2))

			# 負例とする場合のcb
			score_1 = math.exp(-((args.alpha * enh_freq[cand_enh]["+"] - (enh_freq[cand_enh]["-"]+1)) ** 2)) * math.exp(-((args.alpha * prm_freq[cand_prm]["+"] - (prm_freq[cand_prm]["-"]+1)) ** 2))
			
			# calcurate prob
			if score_0 + score_1 == 0:
				prob = 0
			else:
				prob = score_1 / (score_0 + score_1)

			# prob で 負例にする
			if random.random() < prob:
				now_neg[cand_neg] = 1
				sum_cb -= (args.alpha * enh_freq[cand_enh]["+"] - enh_freq[cand_enh]["-"]) ** 2
				sum_cb -= (args.alpha * prm_freq[cand_prm]["+"] - prm_freq[cand_prm]["-"]) ** 2
				enh_freq[cand_enh]["-"] += 1
				prm_freq[cand_prm]["-"] += 1
				sum_cb += (args.alpha * enh_freq[cand_enh]["+"] - enh_freq[cand_enh]["-"]) ** 2
				sum_cb += (args.alpha * prm_freq[cand_prm]["+"] - prm_freq[cand_prm]["-"]) ** 2
			
			# print(f"prob:{prob}")
			# print(f"now neg {len(now_neg)}")
			all_cb[t+1] = sum_cb

		plt.plot(range(args.T+1), all_cb/region_cnt, label=chrom)
		
		neg_df_by_chrom = get_neg_df(now_neg=now_neg)
		neg_df = pd.concat([neg_df, neg_df_by_chrom], axis=0)

	new_df = pd.concat([org_df[org_df["label"]==1], neg_df], axis=0)
	new_df.to_csv(os.path.join(args.outdir, f"{args.cell}.csv"), index=False)

	plt.xlabel("t")
	plt.ylabel("class imbalance")
	plt.legend(ncol=5, fontsize="small")
	plt.title(f"{args.cell}")
	plt.savefig(os.path.join(args.outdir, f"{args.cell}.png"))




def get_args():
	p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	p.add_argument("--indir", default="")
	p.add_argument("--outdir", default="")
	p.add_argument("-cell", type=str, default="GM12878")
	p.add_argument("-dmin", type=int, default=0)
	p.add_argument("-dmax", default=2500000)
	p.add_argument("-alpha", type=float, default=1.0)

	p.add_argument("--T", type=float, default=40000)
	
	return p


if __name__ == "__main__":
	p = get_args()
	args = p.parse_args()

	args.dmin, args.dmax = 10000, 2500000
	if args.dmax != "INF":
		args.dmax = int(args.dmax)


	cell_BG = ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]
	cell_TF = ["GM12878", "HeLa-S3", "HUVEC", "IMR90", "K562", "NHEK"]

	for cell in cell_TF:
		args.cell = cell
		args.indir = os.path.join(os.path.dirname(__file__), "TargetFinder_up_to_2500000")
		args.outdir = os.path.join(os.path.dirname(__file__), "TargetFinder_up_to_2500000", "MCMC", f"dmin_{args.dmin},dmax_{args.dmax},alpha_{args.alpha}")
		print(f"cell {args.cell}")
		print(f"dmax {args.dmax}")
		print(f"alpha {args.alpha}")
		os.makedirs(args.outdir, exist_ok=True)
		GibbsSampling_mse(args)