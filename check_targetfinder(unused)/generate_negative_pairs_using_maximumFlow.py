import pandas as pd
import heapq
import os
import pulp

data_root = os.path.join(os.path.dirname(__file__), "data")

enh2prm = {} # key: chromosome -> enhancer_name -> promoter_name, value: 1(positive) or -1(negative)
prm2enh = {} # key: chromosome -> promoter_name -> enhancer_name, value: 1(positive) or -1(negative)

regionName2nodeIndex = {}
regionName2posCnt = {}

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
    

def draw_all_negative_edges():
    for chrom in enh2prm.keys():
        enh_name_list = enh2prm[chrom].keys()
        prm_name_list = prm2enh[chrom].keys()

        for enh_name in enh_name_list:
            for prm_name in prm_name_list:
                if enh2prm[chrom][enh_name].get(prm_name) == 1:
                    continue
                
                enh2prm[chrom][enh_name][prm_name] = -1
                prm2enh[chrom][prm_name][enh_name] = -1


def make_negativeGraph_for_maximumFlow(cell_line):
    edge = []

    nodeIndex = 0
    regionName2nodeIndex["source"] = nodeIndex
    nodeIndex += 1

    for chrom in enh2prm.keys():
        for enhName in enh2prm[chrom].keys():
            for prmName, label in enh2prm[chrom][enhName].items():
                # エンハンサー頂点番号割り当て
                if regionName2nodeIndex.get(enhName) == None:
                    regionName2nodeIndex[enhName] = nodeIndex
                    nodeIndex += 1
                # プロモーター頂点番号割り当て
                if regionName2nodeIndex.get(prmName) == None:
                    regionName2nodeIndex[prmName] = nodeIndex
                    nodeIndex += 1

                # 負例候補辺を貼っておく (cost = 1)
                if label == -1:
                    fr = regionName2nodeIndex[enhName]
                    to = regionName2nodeIndex[prmName]
                    edge.append((chrom, fr, to, 1,))

                # 各領域の正例edgeカウント
                if label == 1:
                    if regionName2posCnt.get(enhName) == None:
                        regionName2posCnt[enhName] = 0
                    if regionName2posCnt.get(prmName) == None:
                        regionName2posCnt[prmName] = 0
                    regionName2posCnt[enhName] += 1
                    regionName2posCnt[prmName] += 1
    
    regionName2nodeIndex["sink"] = nodeIndex

    # sourse -> 各エンハンサー の edge を 貼る
    # cost は 各エンハンサー の 正例数
    for chrom in enh2prm.keys():
        for enhName in enh2prm[chrom].keys():
            cost = regionName2posCnt[enhName]
            fr = regionName2nodeIndex["source"]
            to = regionName2nodeIndex[enhName]
            edge.append((chrom, fr, to, cost))

    # 各プロモーター -> sink の edge を 貼る
    # cost は 各プロモーター の 正例数
    for chrom in prm2enh.keys():
        for prmName in prm2enh[chrom].keys():
            cost = regionName2posCnt[prmName]
            fr = regionName2nodeIndex[prmName]
            to = regionName2nodeIndex["sink"]
            edge.append((chrom, fr, to, cost))


    # graph 情報を csv へ
    chroms, i, j, cost = [], [], [], []
    for (chrom, _i, _j, _cost) in edge:
        chroms.append(chrom)
        i.append(_i)
        j.append(_j)
        cost.append(_cost)
    negativeGraph_for_maximumFlow = pd.DataFrame(
        {
            "chrom": chroms,
            "i" : i,
            "j" : j,
            "cost" : cost
        },
        index=None
    )

    filename = os.path.join(data_root, "negativeGraph_for_maximumFlow", f"{cell_line}_negativeG.csv")
    negativeGraph_for_maximumFlow.to_csv(filename, index=False)

def my_maximumFlow(cell_line):
    filename = os.path.join(data_root, "negativeGraph_for_maximumFlow", f"{cell_line}_negativeG.csv")
    df = pd.read_csv(filename)
    source = 0
    sink = df["j"].max()

    i_list = df["i"].tolist()
    j_list = df["j"].tolist()
    cost_list = df["cost"].tolist()

    z = pulp.LpVariable("z", lowBound=0) # このzはsourceから流出するflow量であり，最大化する目的関数でもある.
    problem = pulp.LpProblem("最大流問題", pulp.LpMaximize)
    problem += z # 目的関数を設定
    # 大量の変数を作る
    df["Var"] = [pulp.LpVariable(f"x{i_list[index]}_{j_list[index]}", lowBound=0, upBound=cost_list[index],cat=pulp.LpInteger) for index in df.index]

    # 全頂点に関する制約条件を追加していく（保存則）
    for node in set(i_list)|set(j_list):
        if node == source:
            sumFlowFromSource = pulp.lpSum(list(df[df["i"] == node]["Var"]))
            problem += sumFlowFromSource == z
        elif node == sink:
            sumFlowToSink = pulp.lpSum(list(df[df["j"] == node]["Var"]))
            problem += sumFlowToSink == z
        else:
            sumFlowFromNode = pulp.lpSum(list(df[df["i"] == node]["Var"]))
            sumFlowToNode = pulp.lpSum(list(df[df["j"] == node]["Var"]))
            problem += sumFlowFromNode - sumFlowToNode == 0

    # 解を求める
    result = problem.solve()
    # LpStatusが`optimal`なら最適解が得られた事になる
    print(pulp.LpStatus[result])
    # 目的関数の値
    print(pulp.value(problem.objective))
    # 'Var'変数の結果の値をまとめて'Val'列にコピーしている
    df['Val'] = df.Var.apply(pulp.value)
    # 結果表示
    print(df)
    df.to_csv(filename, index=False)


def maximumFlow(cell_line):
    filename = os.path.join(data_root, "negativeGraph_for_maximumFlow", f"{cell_line}_negativeG.csv")
    df = pd.read_csv(filename)

    s = 0  # 始点
    t = df["j"].max()  # 終点

    # CSVの各行Lに対して、変数xi_jを生成し、
    # 新しい列'Var'として追加している。
    df['Var'] = [pulp.LpVariable(f'x{df.i[L]}_{df.j[L]}', 0, df.c[L])
            for L in df.index]
    z = pulp.LpVariable('z')

    p = pulp.LpProblem('最大流問題', sense=pulp.LpMaximize)
    p += z, '目的関数'  # zの実体は以下の制約条件内で定義する
    # set(df.i)はi列の一覧、set(df.j)はj列の一覧。|でつないだら、その和集合
    for n in set(df.i)|set(df.j):
        if n == s:
            p += pulp.lpSum(df.Var[df.i==n]) - pulp.lpSum(df.Var[df.j==n]) == z, f'始点{n}'
        elif n == t:
            p += pulp.lpSum(df.Var[df.i==n]) - pulp.lpSum(df.Var[df.j==n]) == -z, f'終点{n}'
        else:
            p += pulp.lpSum(df.Var[df.i==n]) - pulp.lpSum(df.Var[df.j==n]) == 0, f'点{n}'

    # 制約条件をすべて登録したので解を求める
    result = p.solve()
    # LpStatusが`optimal`なら最適解が得られた事になる
    print(pulp.LpStatus[result])
    # 目的関数の値
    print(pulp.value(p.objective))
    # 'Var'変数の結果の値をまとめて'Val'列にコピーしている
    df['Val'] = df.Var.apply(pulp.value)
    # 結果表示
    print(df)

    df.to_csv(filename, index=False)


def make_negetive_pairs():
    nodeIndex2regionName = {nodeIndex: regionName for regionName, nodeIndex in regionName2nodeIndex.items()}

    filename = os.path.join(data_root, "negativeGraph_for_maximumFlow", f"{cell_line}_negativeG.csv")
    df = pd.read_csv(filename)
    for _, row_data in df.iterrows():
        if row_data["Val"] != 1:
            continue

        chrom = row_data["chrom"]
        fr = nodeIndex2regionName[row_data["i"]]
        to = nodeIndex2regionName[row_data["j"]]

        if fr == "source" or fr == "sink" or to == "source" or to == "sink":
            continue
        
        enh2prm[chrom][fr][to] = 0


def make_my_train_csv(cell_line):
    chrom_list = []
    enh_name_list = []
    prm_name_list = []
    label_list = []
    for chrom in enh2prm.keys():
        for enh_name in enh2prm[chrom].keys():
            for prm_name, label in enh2prm[chrom][enh_name].items():
                # label = (1: positive, 0: negative)
                if label == -1: # 辺が無い時
                    continue
                chrom_list.append(chrom)
                enh_name_list.append(enh_name)
                prm_name_list.append(prm_name)
                label_list.append(label)
    
    df = pd.DataFrame(list(zip(chrom_list, enh_name_list, prm_name_list, label_list)), columns = ["enhancer_chrom", "enhancer_name", "promoter_name", "label"])
    filename = os.path.join(data_root, "my3", f"{cell_line}_train.csv")
    df.to_csv(filename, index=False)


if __name__ == "__main__":
    cell_line_list = ["K562", "GM12878", "HUVEC", "HeLa-S3", "NHEK", "IMR90"]
    for cell_line in cell_line_list:
        enh2prm = {}
        prm2enh = {}
        regionName2nodeIndex = {}
        regionName2posCnt = {}
        extract_positive_pairs(cell_line)
        draw_all_negative_edges()
        make_negativeGraph_for_maximumFlow(cell_line)
        my_maximumFlow(cell_line)
        make_negetive_pairs()
        make_my_train_csv(cell_line)