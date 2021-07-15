import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
import pickle
warnings.filterwarnings("ignore")

df = pd.DataFrame(pd.read_csv(r"parkinsons.csv"))
x=np.array(df.drop('status',axis=1))
y=np.array(df['status'])
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
pca.fit(x)
x_pca = pca.transform(x)
x_train , x_test , y_train , y_test = train_test_split(x_pca,y,test_size=0.2,random_state=7)
scaler=StandardScaler()
scaler.fit(x_train)
X_train=scaler.transform(x_train)
X_test=scaler.transform(x_test)

from xgboost import XGBClassifier
model1=XGBClassifier()
model1.fit(X_train,y_train)
y_pred=model1.predict(X_test)


pickle.dump(model1,open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
