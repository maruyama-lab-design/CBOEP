import pandas as pd
import heapq
import os

data_root = os.path.join(os.path.dirname(__file__), "data")

enh2prm = {} # key: chromosome -> enhancer_name -> promoter_name, value: 1(positive) or -1(negative)
prm2enh = {} # key: chromosome -> promoter_name -> enhancer_name, value: 1(positive) or -1(negative)

def extract_positive_pairs(cell_line):
    all_pairs_df = pd.read_csv(f"{data_root}/TargetFinder/{cell_line}_train.csv", usecols=["enhancer_chrom", "enhancer_name", "promoter_name", "label"])
    positive_pairs_df = all_pairs_df[all_pairs_df["label"] == 1]
    print(positive_pairs_df.head())
    for _, positive_pair in positive_pairs_df.iterrows():
        enh_name = positive_pair["enhancer_name"]
        prm_name = positive_pair["promoter_name"]
        chrom = positive_pair["enhancer_chrom"]

        # chromosome番号チェック
        if (prm_name.split(":")[0]).split("|")[1] != (enh_name.split(":")[0]).split("|")[1]:
            print("エラー!!")
            exit()
        
        if enh2prm.get(chrom) == None:
            enh2prm[chrom] = {}
        if prm2enh.get(chrom) == None:
            prm2enh[chrom] = {}

        if enh2prm[chrom].get(enh_name) == None:
            enh2prm[chrom][enh_name] = {}
        if prm2enh[chrom].get(prm_name) == None:
            prm2enh[chrom][prm_name] = {}

        enh2prm[chrom][enh_name][prm_name] = 1
        prm2enh[chrom][prm_name][enh_name] = 1



def make_negetive_pairs():

    for chrom in enh2prm.keys():
        print(f"{chrom}...")

        enh2pos_cnt_dic = {} # enhancer の名前 を key にして 正例が何本あるか
        prm2pos_cnt_dic = {} # promoter の名前 を key にして 正例が何本あるか
        for enh_name in enh2prm[chrom].keys():
            enh2pos_cnt_dic[enh_name] = len(enh2prm[chrom][enh_name])
        for prm_name in prm2enh[chrom].keys():
            prm2pos_cnt_dic[prm_name] = len(prm2enh[chrom][prm_name])

        # 辞書をリストに
        list_E = sorted(enh2pos_cnt_dic.items(), key = lambda x: x[1], reverse=True)
        list_P = sorted(prm2pos_cnt_dic.items(), key = lambda x: x[1], reverse=True)
        # print(list_E[:10])
        # print(list_P[:10])

        # 領域nameと正例ペア数を入れ替え & 正例ペア数に-1をかける(優先度付きキューは最小値の取り出しなので)
        list_E = [(-1 * cnt, name) for (name, cnt) in list_E]
        list_P = [(-1 * cnt, name) for (name, cnt) in list_P]

        # 優先度付きキューへ
        heapq.heapify(list_E)
        heapq.heapify(list_P)
        print(f"エンハンサー数 {len(list_E)}")
        print(f"プロモーター数 {len(list_P)}")

        # 収束(エンハンサーの優先度付きキューの中身がなくなる)まで
        while len(list_E) > 0:
            (enh_pos_cnt, enh_name) = heapq.heappop(list_E)
            enh_pos_cnt *= -1
            
            trash = [] # ゴミ箱
            while len(list_P) > 0:
                (prm_pos_cnt, prm_name) = heapq.heappop(list_P)
                prm_pos_cnt *= -1
                if enh2prm[chrom][enh_name].get(prm_name) == 1: # すでに正例ならゴミ箱へ
                    trash.append((prm_pos_cnt, prm_name))
                    continue
                else:
                    enh2prm[chrom][enh_name][prm_name] = -1 # 負例として登録！！
                    prm2enh[chrom][prm_name][enh_name] = -1 # 負例として登録！！
                    enh_pos_cnt -= 1
                    prm_pos_cnt -= 1

                    if enh_pos_cnt > 0: # キューへ戻す
                        heapq.heappush(list_E, (-1 * enh_pos_cnt, enh_name))
                    if prm_pos_cnt > 0: # キューへ戻す
                        heapq.heappush(list_P, (-1 * prm_pos_cnt, prm_name))     
                    break
            
            for (prm_pos_cnt, prm_name) in trash: # ゴミ箱へ入れたプロモーターを忘れずに戻す
                heapq.heappush(list_P, (-1 * prm_pos_cnt, prm_name))


        # どれくらいできているか確認...
        # ng_enh = 0
        # for enh_name in enh2prm[chrom].keys():
        #     score = 0
        #     for prm_name, value in enh2prm[chrom][enh_name].items():
        #         score += value
        #     if score != 0:
        #         ng_enh += 1
        # print(f"正負が一致していないエンハンサー数 {ng_enh}")

        # ng_prm = 0
        # for prm_name in prm2enh[chrom].keys():
        #     score = 0
        #     for enh_name, value in prm2enh[chrom][prm_name].items():
        #         score += value
        #     if score != 0:
        #         ng_prm += 1
        #         print(score)
        # print(f"正負が一致していないプロモーター数 {ng_prm}")


def make_my_train_csv(cell_line):
    chrom_list = []
    enh_name_list = []
    prm_name_list = []
    label_list = []
    for chrom in enh2prm.keys():
        for enh_name in enh2prm[chrom].keys():
            for prm_name, label in enh2prm[chrom][enh_name].items():
                chrom_list.append(chrom)
                enh_name_list.append(enh_name)
                prm_name_list.append(prm_name)
                if label == -1:
                    label = 0
                label_list.append(label)
    
    df = pd.DataFrame(list(zip(chrom_list, enh_name_list, prm_name_list, label_list)), columns = ["enhancer_chrom", "enhancer_name", "promoter_name", "label"])
    filename = os.path.join(data_root, "my", f"{cell_line}_train.csv")
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    cell_line_list = ["K562", "GM12878", "HUVEC", "HeLa-S3", "NHEK"]
    for cell_line in cell_line_list:
        enh2prm = {}
        prm2enh = {}
        extract_positive_pairs(cell_line)
        make_negetive_pairs()
        make_my_train_csv(cell_line)