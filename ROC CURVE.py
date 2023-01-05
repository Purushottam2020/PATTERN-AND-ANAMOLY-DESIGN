import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = {'actuals':[1,0,1,1,1,1,0,1,1,1],
      'predictions':[1,0,1,1,1,1,1,1,1,0]}
df = pd.DataFrame(df)
print(df)
cf_matrix=confusion_matrix(df['actuals'],df['predictions'])
print(cf_matrix)
sns.heatmap(cf_matrix,annot=True,cmap='Reds')
plt.ylabel('actuals')
plt.xlabel('predictions')
plt.show()
sns.heatmap(cf_matrix/np.sum(cf_matrix),annot=True,fmt='.2%',cmap='Reds')
plt.ylabel('actuals')
plt.xlabel('predictions')
plt.show()
from sklearn.metrics import accuracy_score
score = accuracy_score(df['actuals'],df['predictions'])
print(score)
