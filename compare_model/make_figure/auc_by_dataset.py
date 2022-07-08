from sklearn.metrics import roc_curve
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_recall_curve, auc
import matplotlib.pyplot as plt
import pandas as pd 
import os

def make_roc_curve():
    cell_line_list = ["GM12878", "K562", "HUVEC", "NHEK", "IMR90", "HeLa-S3"]
    # cell_line_list = ["GM12878"]
    dataset_list = ["new", "TargetFinder", "EP2vec"]
    model = "TargetFinder"


    for cell_line in cell_line_list:
        color = ["green", "blue", "orange"]
        label = ["D_mf", "D_tf", "D_ep"]

        fig, ax = plt.subplots()
        for i, dataset in enumerate(dataset_list):

            y_true = []
            y_pred = []

            resultDir = os.path.join(os.path.dirname(__file__), "..", "result", model, cell_line, dataset)
            if not os.path.exists(resultDir):
                print("not exist!!")
                continue
            files = os.listdir(resultDir)
            files_file = [f for f in files if os.path.isfile(os.path.join(resultDir, f))]
            if len(files_file) == 0:
                print("not exist!!")
                continue
            for file in files_file:
                if file.startswith("."):
                    continue
                result_csv = pd.read_csv(os.path.join(resultDir, file))
                y_true += result_csv["y_test"].tolist()
                y_pred += result_csv["y_pred"].tolist()


            # display = PrecisionRecallDisplay.from_predictions(y_true, y_pred, name="test")
            # _ = display.ax_.set_title("2-class Precision-Recall curve")
                
            precision, recall, threholds = precision_recall_curve(y_true, y_pred)
            auc_score = auc(recall, precision)
            # print(ytr, ypr)

            # print(len(ytr), len(ypr), len(threholds))
            df = pd.DataFrame(data={
                "precision":precision[1:],
                "recall":recall[1:],
                "threhold":threholds
            })
            df.to_csv(os.path.join(os.path.dirname(__file__), "fig", model, cell_line, f"{dataset}_pre-rec.csv"))

            ax.plot(recall, precision, marker='o', color=color[i], markersize=1, label=f"{label[i]} area={auc_score:.3f}")
            print(f"{dataset} plotted!!")

        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid()
        ax.legend()
        # plt.show()

        outputDir = os.path.join(os.path.dirname(__file__), "fig", model, cell_line)
        os.system(f"mkdir -p {outputDir}")
        plt.savefig(os.path.join(outputDir, "precision-recall curve"))


make_roc_curve()