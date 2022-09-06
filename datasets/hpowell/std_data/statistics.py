import pickle

data = pickle.load(open("BCRE-KG-data.pkl", mode="rb"))

new_data = {"Token": [], "CUI": [], "STY": [],
            r"Token$\rightarrow$CUI": [], r"CUI$\rightarrow$CUI": [], r"CUI$\rightarrow$STY": [],
            r"STY$\rightarrow$STY": []}

for line in data:
    node_num = line["kg-data"]["nodes"].size(0)
    token_num = line["kg-data"]["num_recognized_tokens"]
    cui_num = line["kg-data"]["CUI_length"]
    sty_num = line["kg-data"]["STY_length"]
    edge_attrs = line["kg-data"]["edge_attr"]
    token_cui_num, cui_cui_num, cui_sty_num, sty_sty_num = 0, 0, 0, 0
    for items in edge_attrs:
        if items == ["token2CUI"]:
            token_cui_num += 1
        elif items == ["CUI2STY"]:
            cui_sty_num += 1
        for item in items:
            if item.islower():
                sty_sty_num += 1
            else:
                cui_cui_num += 1
    new_data["Token"].append(token_num)
    new_data["CUI"].append(cui_num)
    new_data["STY"].append(sty_num)

    new_data[r"Token$\rightarrow$CUI"].append(token_cui_num)
    new_data[r"CUI$\rightarrow$CUI"].append(cui_cui_num)
    new_data[r"CUI$\rightarrow$STY"].append(cui_sty_num)
    new_data[r"STY$\rightarrow$STY"].append(sty_sty_num)

data = {}
for key, value in new_data.items():
    data[key] = sum(value)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd_dict1 = {"type": [], "num": []}
pd_dict2 = {"type": [], "num": []}

for key, values in new_data.items():
    for value in values:
        if key in ["Token", "CUI", "STY"]:
            pd_dict1["type"].append(key)
            pd_dict1["num"].append(value)
        else:
            pd_dict2["type"].append(key)
            pd_dict2["num"].append(value)

from matplotlib import rcParams

config = {
    "font.family": 'Times New Roman',  # 设置字体类型
    "font.size": 18,
    "mathtext.fontset": 'stix',
}

font_size = 18

rcParams.update(config)
node_pd1 = pd.DataFrame(pd_dict1)
node_pd2 = pd.DataFrame(pd_dict2)
fig = plt.figure(figsize=(16, 6))
plt.subplot(121)
sns.boxplot(x="type", y="num", data=node_pd1)
plt.ylabel("Node Number of Each Sentence", fontsize=font_size)
plt.xlabel("Node Type", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.subplot(122)
sns.boxplot(x="type", y="num", data=node_pd2)
plt.ylabel("Edge Number of Each Sentence", fontsize=font_size)
plt.xlabel("Edge Type", fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.subplots_adjust(wspace=0.1)
fig.tight_layout()
plt.show()

print()
