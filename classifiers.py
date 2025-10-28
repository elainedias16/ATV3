import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

import numpy as np
import matplotlib.pyplot as plt
df1 = pd.read_csv("ATV3/dataset/final_df.csv")
df2 = pd.read_csv("ATV3/dataset/test_df.csv")

df = pd.concat([df1,df2],axis=0)
df.reset_index(inplace=True,drop=True)
cms = []
mean_f1,mean_accr,mean_precision =[],[],[]



kf = StratifiedKFold(n_splits=5)
X, y = df[[f"emb_dim{i}" for i in range(512)]], df["dir"]
for train_idx, test_idx in kf.split(X, y):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X.iloc[train_idx], y.iloc[train_idx])      # use .iloc for row indexing
    y_pred = knn.predict(X.iloc[test_idx])
    mean_f1.append(f1_score(y.iloc[test_idx], y_pred,average="weighted"))
    mean_accr.append(accuracy_score(y.iloc[test_idx], y_pred))
    mean_precision.append(precision_score(y.iloc[test_idx], y_pred,average="weighted"))
    cm = confusion_matrix(y.iloc[test_idx], y_pred,normalize="true")
    cms.append(cm)

print(sum(mean_f1)/len(mean_f1)) 
print(sum(mean_accr)/len(mean_accr))
print(sum(mean_precision)/len(mean_precision))
mean_cm = np.mean(cms, axis=0) * 100

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(mean_cm, interpolation='nearest', cmap='Blues')
ax.figure.colorbar(im, ax=ax, label="Percentage (%)")
ax.set(
    xticks=np.arange(mean_cm.shape[1]),
    yticks=np.arange(mean_cm.shape[0]),
    xlabel='Predicted label',
    ylabel='True label',
    title='Mean Normalized Confusion Matrix (5-fold CV)'
)

# Annotate values with two decimal precision
for i in range(mean_cm.shape[0]):
    for j in range(mean_cm.shape[1]):
        ax.text(j, i, f"{mean_cm[i, j]:.2f}%", ha="center", va="center", color="black")

plt.tight_layout()
plt.savefig("/srv12t/educampos/ATV3/fig.png")
plt.show()