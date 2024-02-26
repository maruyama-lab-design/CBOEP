import result_analysis as ra
import os
import glob 


if __name__ == "__main__":

	prob_start = 0
	prob_end = 1.0
	n_bins = 20
	accum = False
	colors = [
		"#bf7fff", # 紫
		"#00a7db", # 少し濃い青
		"#ff7f7f", # 赤
		"#00bb85" # 少し濃い緑
	]
	markers = [
		"o",
		"x",
		"^",
		"v"
	]
	ylim = [-1, 1]
	custom_bin_list = [0.0, 0.05, 0.5, 0.95, 1.0]

	title = ""
	labels = [
		"unnormalized BENGI",
		"BENGI",
		"CBMF",
		"CBGS"
	]

	train_cell = "GM12878"

	indirs_TransEPI = [
		f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_BENGI_up_to_2500000_{train_cell}",
		f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_BENGI_up_to_2500000_region_restricted_{train_cell}",
		f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_dmax_2500000,p_100_{train_cell}",
		f"./TransEPI/2023-11-01_cv_TransEPI/wo-mask_wo-mse_w-weighted_bce_dmax_2500000,alpha_100_{train_cell}"
	]
	indirs_TargetFinder = [
		f"./TargetFinder/prediction_w-window/BENGI_up_to_2500000/train_{train_cell}",
		f"./TargetFinder/prediction_w-window/BENGI_up_to_2500000_region_restricted/train_{train_cell}",
		f"./TargetFinder/prediction_w-window/CBOEP(dmax_2500000,alpha_100)/train_{train_cell}",
		f"./TargetFinder/prediction_w-window/MCMC(dmax_2500000,alpha_100)/train_{train_cell}"
	]

	


	for test_cell in ["GM12878", "HeLa-S3", "HMEC", "IMR90", "K562", "NHEK"]:
		for fold in ["combined"]:

			ra.make_prob_line_graph(
				indirs=[
					f"./TransEPI/common_test/BENGI_up_to_2500000_region_restricted",
					f"./TransEPI/common_test/CBMF",
					f"./TransEPI/common_test/CBGS"
				],
				labels=[
					"BENGI", "CBMF", "CBGS"
				],
				colors=[
					colors[1], colors[2], colors[3]
				],
				markers=[
					markers[1], markers[2], markers[3]
				],
				prob_range=[prob_start, prob_end],
				n_bins=n_bins,
				custom_bin_list=custom_bin_list,
				accumulate=accum,
				ylim=ylim,
				title=test_cell,
				outfile= f"./line_graph[{prob_start},{prob_end}(accum={str(accum)},nbins={n_bins})]/{train_cell}-{test_cell}/{fold}/TransEPI_UNMASKED_all-neg(BGvsMFvsGS).png",
				folds=[fold],
				test_cell=test_cell
			)
			continue






			outfile_TransEPI = f"./line_graph[{prob_start},{prob_end}(accum={str(accum)},nbins={n_bins})]/{train_cell}-{test_cell}/{fold}/TransEPI-UNMASKED(BG(org)vsBG(rm)vsMFvsGS).png"
			outfile_TargetFinder = f"./line_graph[{prob_start},{prob_end}(accum={str(accum)},nbins={n_bins})]/{train_cell}-{test_cell}/{fold}/TargetFinder-UNMASKED(BG(org)vsBG(rm)vsMFvsGS).png"

			ra.make_prob_line_graph(
				indirs=indirs_TransEPI,
				labels=labels,
				colors=colors,
				markers=markers,
				prob_range=[prob_start, prob_end],
				n_bins=n_bins,
				custom_bin_list=custom_bin_list,
				accumulate=accum,
				ylim=ylim,
				title=title,
				outfile=outfile_TransEPI,
				folds=[fold],
				test_cell=test_cell
			)

			if test_cell == "HMEC":
				continue


			ra.make_prob_line_graph(
				indirs=indirs_TargetFinder,
				labels=labels,
				colors=colors,
				markers=markers,
				prob_range=[prob_start, prob_end],
				n_bins=n_bins,
				custom_bin_list=custom_bin_list,
				accumulate=accum,
				ylim=ylim,
				title=title,
				outfile=outfile_TargetFinder,
				folds=[fold],
				test_cell=test_cell

			)
			continue

			ra.make_prob_line_graph(
				indirs=[
					# f"./TransEPI/2023-11-01_cv_TransEPI/w-mask_wo-mse_w-weighted_bce_BENGI_up_to_2500000_region_restricted_{train_cell}",
					f"./TransEPI/2023-11-01_cv_TransEPI/w-mask_wo-mse_w-weighted_bce_dmax_2500000,p_100_{train_cell}",
					f"./TransEPI/2023-11-01_cv_TransEPI/w-mask_wo-mse_w-weighted_bce_dmax_2500000,alpha_100_{train_cell}"
				],
				labels=[
					# "BENGI-negative",
					"CBMF-negative",
					"CBGS-negative"
				],
				colors=colors,
				prob_range=[prob_start, prob_end],
				n_bins=n_bins,
				accumulate=accum,
				# title=f"TransEPI-MASKED test on {test_cell}",
				outfile=f"./line_graph[{prob_start},{prob_end}(accum={str(accum)},nbins={n_bins})]/{train_cell}-{test_cell}/{fold}/TransEPI-MASKED(MFvsGS).png",
				folds=[fold],
				test_cell=test_cell
			)
				