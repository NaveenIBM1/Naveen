import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df=pd.read_csv("kidney_disease.csv")
print("The dataset shape is {}".format(df.shape))

# remove "id" feature 
df.drop('id',axis=1,inplace=True)

#Cleaning the Data values¶

#in our dataset some features ['pcv','wc','rc','dm','cad','classification'] contains some special character.so replace them with appropriate values.

# cleaning 'PCV'
df['pcv']=df['pcv'].apply(lambda x:x if type(x)==type(3.5) else x.replace('\t43','43').replace('\t?','Nan'))

# cleaning "WC"
df['wc']=df['wc'].apply(lambda x:x if type(x)==type(3.5) else x.replace('\t?','Nan').replace('\t6200','6200').replace('\t8400','8400'))

# cleaning "RC"
df['rc']=df['rc'].apply(lambda x:x if type(x)==type(3.5) else x.replace('\t?','Nan'))

# cleaning "dm"
df['dm']=df['dm'].apply(lambda x:x if type(x)==type(3.5) else x.replace('\tno','no').replace('\tyes','yes').replace(' yes','yes'))

# cleaning "CAD"
df['cad']=df['cad'].apply(lambda x:x if type(x)==type(3.5) else x.replace('\tno','no'))

# cleaning "Classification"
df['classification']=df['classification'].apply(lambda x:x if type(x)==type(3.5) else x.replace('ckd\t','ckd'))

#Note: Some features are mistyped as "object".so convert them into "float" type

mistyped=[['pcv','rc','wc']]
for i in mistyped:
    df[i]=df[i].astype('float')
#  define categoricsl features
cat_cols=list(df.select_dtypes('object'))
cat_cols

# define numeric features
num_cols=list(df.select_dtypes(['int64','float64']))
num_cols
# Checking missing/Nan values
df.isnull().sum().sort_values(ascending=False)

# Let's impute Nan Values with median in numeric features
for col in num_cols:
    df[col]=df[col].fillna(df[col].median())
# let's impute categorical features with most frequent value
df['rbc'].fillna('normal',inplace=True)
df['pc'].fillna('normal',inplace=True)
df['pcc'].fillna('notpresent',inplace=True)
df['ba'].fillna('notpresent',inplace=True)
df['htn'].fillna('no',inplace=True)
df['dm'].fillna('no',inplace=True)
df['cad'].fillna('no',inplace=True)
df['appet'].fillna('good',inplace=True)
df['pe'].fillna('no',inplace=True)
df['ane'].fillna('no',inplace=True)
df.isna().sum().sort_values(ascending=False)


# Encode classification
df['classification']=df['classification'].map({'ckd':1,'notckd':0})
attr_count=df['classification'].value_counts()
attr_label=df['classification'].value_counts().index

### plot
##fig,ax=plt.subplots(figsize=(14,6))
##ax.pie(attr_count,explode=(0.1,0),labels=attr_label,autopct='%.2f%%',startangle=90)
##ax.set_title("Classification ",fontsize=15)
##plt.show()
##
##fig,ax=plt.subplots(figsize=(7,70),ncols=1,nrows=14)
##
##i=0
##for col in num_cols:
##    sns.kdeplot(x=df[col],fill=True,alpha=1,ax=ax[i])
##    ax[i].set_xlabel(' ')
##    ax[i].set_ylabel(' ')
##    ax[i].set_title(col,fontsize=21)
##    i=i+1
##plt.show()
##
##
### check skewness of the distribution 
##skew=[]
##for col in num_cols:
##    skew.append(round(df[col].skew(),3))
##num_dist=pd.DataFrame({'features':num_cols,'skewness':skew})
##num_dist
##
##plt.figure(figsize=(16,8))
##plt.title('Correlation between All Numerical Features',size=15)

### create mask
##mask=np.triu(np.ones_like(df.corr()))
##
### create colormap
##colormap=sns.color_palette('Blues')
### plot heatmap
##sns.heatmap(df.corr(),annot=True,cmap=colormap,mask=mask)
##plt.show()
##
##df.drop('pcv',axis=1,inplace=True)
##num_cols.remove('pcv')
##
##tg_num_corr=[]
##
##for col in num_cols:
##    tg_num_corr.append(df[col].corr(df['classification']))
##    
### create as DataFrame
##tg_num_df=pd.DataFrame({'numerical_predictor':num_cols,'correlation_w_target':tg_num_corr})
##
### sort the DataFrmae by the absolute vaue of their correlation coefficient,descending
##tg_num_df=tg_num_df.sort_values(by='correlation_w_target',ascending=False).reset_index(drop=True)
##
##tg_num_df
##
##
### display as figure
##plt.figure(figsize=(7,5))
##sns.barplot(x=tg_num_df['correlation_w_target'],y=tg_num_df['numerical_predictor'],color='#a2c9f4')
##plt.xlabel('Correlation Coefficient')
##plt.title('Numerical-Target Relationship',fontsize=12)
##plt.show()
##

df['rbc']=df['rbc'].map({'normal':0,'abnormal':1})
df['pc']=df['pc'].map({'normal':0,'abnormal':1})
df['pcc']=df['pcc'].map({'notpresent':0,'present':1})
df['ba']=df['ba'].map({'notpresent':0,'present':1})
df['htn']=df['htn'].map({'no':0,'yes':1})
df['dm']=df['dm'].map({'no':0,'yes':1})
df['cad']=df['cad'].map({'no':0,'yes':1})
df['pe']=df['pe'].map({'no':0,'yes':1})
df['ane']=df['ane'].map({'no':0,'yes':1})
df['appet']=df['appet'].map({'good':0,'poor':1})

# scaling with MinMaxScaler
from sklearn.preprocessing import StandardScaler,MinMaxScaler           
mm_scaler=MinMaxScaler()
df[num_cols]=mm_scaler.fit_transform(df[num_cols])

from sklearn.model_selection import train_test_split
x=df.drop('classification',axis=1)
y=df['classification']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
print("X_train size {} , X_test size {}".format(X_train.shape,X_test.shape))

# Using GridSearchCV we find the best algorithm to this problem
from sklearn.model_selection import ShuffleSplit,GridSearchCV,StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# Crete a function to find the best algo. for this problem
def find_best_model(x,y):
    models={'Logistic_regression':{'model':LogisticRegression(solver='liblinear',penalty='l2',multi_class='auto'),'parameter':{'C':[1,4,8]}},
           'decision_tree':{'model':DecisionTreeClassifier(splitter='best'),'parameter':{'criterion':['gini','entropy'],'max_depth':[5,7,13,15]}},
           'svm':{'model':SVC(gamma='auto'),'parameter':{'kernel':['sigmoid','linear'],'C':[1,5,10,15]}},
           'random_forest':{'model':RandomForestClassifier(criterion='gini'),'parameter':{'max_depth':[5,10,15],'n_estimators':[1,3,5]}}}
    scores=[]
    cv_shuffle=StratifiedKFold(n_splits=10)
    
    for model_name,model_params in models.items():
        gs=GridSearchCV(model_params['model'],model_params['parameter'],cv=cv_shuffle,return_train_score=False)
        gs.fit(x,y)
        scores.append({'model':model_name,'best_parameters':gs.best_params_,'score':gs.best_score_})
    return pd.DataFrame(scores,columns=['model','best_parameters','score'])


find_best_model(X_train,y_train)
# Using cross_val_score for gaining average accuracy
from sklearn.model_selection import cross_val_score
score=cross_val_score(RandomForestClassifier(max_depth=15,n_estimators=5),X_train,y_train,cv=10)
print("Average Accuracy Score {}".format(score.mean()))

# Creating Random Forest model
rf=RandomForestClassifier(max_depth=5,n_estimators=5)
rf.fit(X_train,y_train)
RandomForestClassifier(max_depth=5, n_estimators=5)

# Creating a confusion matrix
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
y_pred=rf.predict(X_test)
cm=confusion_matrix(y_pred,y_test)
cm


##
### Plotting the confusion matrix
##plt.figure(figsize=(10,7))
##p = sns.heatmap(cm, annot=True, cmap="Blues", fmt='g')
##plt.title('Confusion matrix for RandomForest Model - Test Set')
##plt.xlabel('Predicted Values')
##plt.ylabel('Actual Values')
##plt.show()
##

# Accuracy score
score=round(accuracy_score(y_test,y_pred),3)
print("Accuracy on the Test set: {}".format(score))

# Classification report 
print(classification_report(y_test,y_pred))

# Creating a confusion matrix for training set
y_train_pred=rf.predict(X_train)
cm=confusion_matrix(y_train,y_train_pred)
cm


# Accuracy score
score=round(accuracy_score(y_train,y_train_pred),3)
print("Accuracy on training set: {}".format(score))

print(classification_report(y_train,y_train_pred))

# Top 10 Features
feature_scores=pd.DataFrame(rf.feature_importances_,columns=['Score'],index=X_train.columns).sort_values(by='Score',ascending=False)
top10_feature = feature_scores.nlargest(n=10, columns=['Score'])

plt.figure(figsize=(14,6))
g = sns.barplot(x=top10_feature.index, y=top10_feature['Score'])
p = plt.title('Top 10 Features with Random Forest')
p = plt.xlabel('Feature name')
p = plt.ylabel('Random Forest score')
p = g.set_xticklabels(g.get_xticklabels(), horizontalalignment='right')



X_train=X_train[['hemo','rc','sg','al','sc','htn','sod','bp','wc','age']]
X_test=X_test[['hemo','rc','sg','al','sc','htn','sod','bp','wc','age']]
rf.fit(X_train,y_train)
def predict(hemo,rc,sg,al,sc,htn,sod,bp,wc,age):
    hemo=float(hemo)
    rc=float(rc)
    sg=float(sg)
    sc=float(sc)
    htn=int(htn)
    sod=float(sod)
    bp=float(bp)
    wc=float(wc)
    age=int(age)
    
    x=[[hemo,rc,sg,al,sc,htn,sod,bp,wc,age]]
    return rf.predict(x)
# Prediction 1
# input parameter : Hemoglobin(hemo), Red Blood Cells(rc), Specific Gravity(sg), Albumin(al), Searum Creatinite(sc), 
# Hypertension(htn), Sodium(sod), Blood Pressure(bp), White Blood Cells(wc), Age
prediction = predict(67.4,7.2,0.99,4,17.0,1,160.6,87,22089,36)[0]
if prediction:
  print('Oops! You have Chronic Kidney Disease.')
else:
  print("Great! You don't have Chronic Kidney Disease.")


# Prediction 2
# input parameter : Hemoglobin(hemo), Red Blood Cells(rc), Specific Gravity(sg), Albumin(al), Searum Creatinite(sc), 
# Hypertension(htn), Sodium(sod), Blood Pressure(bp), White Blood Cells(wc), Age
prediction = predict(27.4,4.2,0.19,1,7.0,0,90.6,37,30949,26)[0]
if prediction:
  print('Oops! You have Chronic Kidney Disease.')
else:
  print("Great! You don't have Chronic Kidney Disease.")


# Prediction 3
# input parameter : Hemoglobin(hemo), Red Blood Cells(rc), Specific Gravity(sg), Albumin(al), Searum Creatinite(sc), 
# Hypertension(htn), Sodium(sod), Blood Pressure(bp), White Blood Cells(wc), Age
#prediction = predict(17.4,2.2,0.89,0,12.0,0,50.6,87,949,19)[0]
prediction = predict(17.7,5.5,1.02,0,1.2,0,142,80,4300,23)[0]
if prediction:
  print('Oops! You have Chronic Kidney Disease.')
else:
  print("Great! You don't have Chronic Kidney Disease.")

