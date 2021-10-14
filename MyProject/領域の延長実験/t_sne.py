import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def t_SNE(args, cell_line, X, Y):
	tsne = TSNE(n_components=2, random_state = 0)
	X_reduced = tsne.fit_transform(X)

	figure = plt.figure()
	plt.figure(figsize=(13, 7))
	plt.scatter(X_reduced[:, 0], X_reduced[:, 1],
				c=Y, cmap='jet',
				s=15, alpha=0.5)
	plt.axis('off')
	plt.colorbar()
	# plt.show()
	figure.savefig(f"{args.my_data_folder_path}/figure/{args.output}.png")
