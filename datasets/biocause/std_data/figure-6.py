layer_number = [1,2,3,4,5,6,7,8]
hidden_dim = [0,100,200,300,400,500,600,700]

layer_bert1 = [0.5852,0.5886,0.5771,0.5882,0.5978, 0.5928, 0.5906, 0.5577]
layer_bert2 = [0.5989,0.6110,0.6006,0.5908,0.6270, 0.6028, 0.6194, 0.5995]
layer_bert3 = [0.6064,0.6110,0.6006,0.6203,0.6197, 0.6215, 0.6077, 0.5983]
layer_bert4 = [0.5989,0.6089,0.6006,0.5908,0.6270, 0.6028, 0.6178, 0.5995]


hidden_bert1 = [0.5847,0.6053,0.5756,0.5978,0.5756, 0.5819, 0.6006, 0.5915]
hidden_bert2 = [0.5968,0.6070,0.6141,0.6270,0.5803, 0.5862, 0.6113, 0.5899]
hidden_bert3 = [0.6175,0.6028,0.6167,0.6197,0.5918, 0.6011, 0.6094, 0.6150]
hidden_bert4 = [0.6017,0.6017,0.6192,0.6270,0.5771, 0.5876, 0.6095, 0.5892]

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.DataFrame({"layer_number": layer_number, "hidden_dim": hidden_dim,
                     "layer_bert1": layer_bert1, "layer_bert2": layer_bert2, "layer_bert3": layer_bert3, "layer_bert4": layer_bert4,
                     "hidden_bert1": hidden_bert1, "hidden_bert2": hidden_bert2, "hidden_bert3": hidden_bert3, "hidden_bert4": hidden_bert4})

data.to_excel("data.xlsx")
