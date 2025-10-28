import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as patches

df = pd.read_csv("/srv12t/educampos/ATV3/dataset/final_df.csv")
df2 = pd.read_csv("/srv12t/educampos/ATV3/dataset/test_df.csv")

df = pd.concat([df,df2],axis=0)
df.reset_index(inplace=True)
pca = PCA(n_components=2)

principalComponents = pca.fit_transform(df[[f"emb_dim{i}" for i in range(512)]])
pca_df = pd.DataFrame(principalComponents)
pca_df = pd.concat([pca_df,df["dir"]],axis=1)
pca_df.columns = ["dm_1","dm_2","dir"]

list = {pca_df["dir"].unique()[0] : "red",
pca_df["dir"].unique()[1] : "blue",
pca_df["dir"].unique()[2] : "green",
pca_df["dir"].unique()[3] : "white",
pca_df["dir"].unique()[4] : "black",
pca_df["dir"].unique()[5] : "brown",
pca_df["dir"].unique()[6] : "orange",
pca_df["dir"].unique()[7] : "purple",
pca_df["dir"].unique()[8] : "cyan"}


print(pca_df["dir"].value_counts())

fig = px.scatter(pca_df, x=pca_df["dm_1"], y=pca_df["dm_2"], 
                 color=pca_df["dir"],color_discrete_map=list,
                 title = "PCA Images Embbeds")
fig.show()