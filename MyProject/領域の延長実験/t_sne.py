import matplotlib.pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE
import argparse


def t_SNE(args, cell_line, X, Y):
	tsne = TSNE(n_components=2, random_state = 0, perplexity = 30, n_iter = 1000)
	X_reduced = tsne.fit_transform(X)

	figure = plt.figure()
	df = pd.DataFrame(dict(x=X_reduced[:, 0], y=X_reduced[:, 1], label=Y))
	groups = df.groupby("label")
	for name, group in groups:
		plt.plot(group.x, group.y, marker="o", linestyle="None", ms=4, label=name)
	plt.legend()
	# plt.show()
	figure.savefig(f"{args.my_data_folder_path}/figure/{args.output}.png")
