"""Titanic ödevi: veri sızıntısı olmadan GridSearchCV(5 kat) ve test ROC."""
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from pathlib import Path

ROOT=Path(__file__).resolve().parent
csv=ROOT/'Titanic-Dataset.csv'
if not csv.exists(): csv=Path('/mnt/data/Titanic-Dataset.csv')
df=pd.read_csv(csv)
# PassengerId: kimlik; Name/Ticket: yüksek kardinaliteli metin; Cabin: büyük eksiklik (%77)
X=df.drop(columns=['Survived','PassengerId','Name','Ticket','Cabin'])
y=df['Survived']
num=['Pclass','Age','SibSp','Parch','Fare']; cat=['Sex','Embarked']
pre=ColumnTransformer([
 ('numeric',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),
 ('categoric',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(handle_unknown='ignore'))]),cat)
])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
models={
 'Logistic Regression':(LogisticRegression(max_iter=1000,random_state=42), {'model__C':[.1,1,10]}),
 'KNN':(KNeighborsClassifier(),{'model__n_neighbors':[3,5,9]}),
 'SVM':(SVC(probability=True,random_state=42),{'model__C':[.1,1,10], 'model__kernel':['linear','rbf']}),
 'CART':(DecisionTreeClassifier(random_state=42),{'model__max_depth':[3,5,None], 'model__min_samples_leaf':[1,5]}),
 'Random Forest':(RandomForestClassifier(random_state=42,n_jobs=1),{'model__n_estimators':[100], 'model__max_depth':[4,8,None]}),
 'AdaBoost':(AdaBoostClassifier(random_state=42),{'model__n_estimators':[50,100], 'model__learning_rate':[.5,1.]}),
 'XGBoost':(XGBClassifier(random_state=42,eval_metric='logloss',n_jobs=1),{'model__n_estimators':[100], 'model__max_depth':[2,4], 'model__learning_rate':[.05,.1]})
}
cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
rows=[]; best={}
for name,(est,params) in models.items():
 print('Fitting:',name,flush=True)
 search=GridSearchCV(Pipeline([('preprocess',pre),('model',est)]),params,cv=cv,scoring='roc_auc',n_jobs=1,refit=True)
 search.fit(X_train,y_train); best[name]=search.best_estimator_
 pred=search.predict(X_test); proba=search.predict_proba(X_test)[:,1]
 rows.append({'Model':name,'CV_AUC':round(search.best_score_,4),'Test_Accuracy':round(accuracy_score(y_test,pred),4),'Precision':round(precision_score(y_test,pred,zero_division=0),4),'Recall':round(recall_score(y_test,pred,zero_division=0),4),'F1':round(f1_score(y_test,pred),4),'Test_AUC':round(roc_auc_score(y_test,proba),4),'Best_Params':str(search.best_params_)})
 fig,ax=plt.subplots(figsize=(6,5)); RocCurveDisplay.from_predictions(y_test,proba,ax=ax,name=name);ax.plot([0,1],[0,1],'k--',alpha=.4);fig.tight_layout();fig.savefig(ROOT/('ROC_'+name.replace(' ','_')+'.png'),dpi=130);plt.close(fig)
res=pd.DataFrame(rows).sort_values('Test_AUC',ascending=False)
res.to_csv(ROOT/'Titanic_Model_Sonuclari.csv',index=False)
fig,ax=plt.subplots(figsize=(8,6))
for name,model in best.items():RocCurveDisplay.from_predictions(y_test,model.predict_proba(X_test)[:,1],ax=ax,name=name)
ax.plot([0,1],[0,1],'k--',alpha=.4);fig.tight_layout();fig.savefig(ROOT/'ROC_Tum_Modeller.png',dpi=160);plt.close(fig)
print(res.to_string(index=False))
