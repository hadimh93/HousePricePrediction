#imort package
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from sklearn.linear_model import LogisticRegression,LinearRegression,SGDClassifier
from sklearn.preprocessing import scale
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn import preprocessing
from sklearn import utils
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import cross_val_score
import warnings

#read data
train = pd.read_csv('D:/Data/Data Mining/Data Minhig Class (Zerehsaz)/Data-Project/Data-Project01/Project/train_data.csv')
test = pd.read_csv('D:/Data/Data Mining/Data Minhig Class (Zerehsaz)/Data-Project/Data-Project01/Project/test_data.csv')

test['SalePrice']=0

#histogram and normal probability plot
sns.distplot(train['SalePrice'] , fit=norm)
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#histogram and normal probability plot after log transform
train["SalePrice"] = np.log(train["SalePrice"])
sns.distplot(train['SalePrice'] , fit=norm);
(mu, sigma) = norm.fit(train['SalePrice'])
print( '\n mu = {:.2f} and sigma = {:.2f}\n'.format(mu, sigma))
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)],loc='best')
plt.ylabel('Frequency')
plt.title('SalePrice distribution')
fig = plt.figure()
res = stats.probplot(train['SalePrice'], plot=plt)
plt.show()

#concat
all_data= pd.concat([train, test], axis=0).reset_index()
all_data=all_data.drop(['index'], axis=1)

#plot missing data before preproceesing
all_data_na = (all_data.isnull().sum() / len(all_data)) * 100
all_data_na = all_data_na.drop(all_data_na[all_data_na == 0].index).sort_values(ascending=False)[:80]
missing_data = pd.DataFrame({'Missing Ratio' :all_data_na})
missing_data.head(80)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_data_na.index, y=all_data_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

#PreProcess
df=all_data
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1890,1909), 1890, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1910,1919), 1910, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1920,1929), 1920, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1930,1939),1930, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1940,1949), 1940, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1950,1959), 1950, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1960,1969), 1960, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1970,1979), 1970, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1980,1984), 1980, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1985,1989), 1985, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1990,1994), 1990, df['GarageYrBlt'])
df['GarageYrBlt'] = np.where(df['GarageYrBlt'].between(1995,1999), 1995, df['GarageYrBlt'])

indexs=df.columns.get_loc('BsmtQual')
for i in range(len( df.loc[:,'Condition1'])):
    if df.isnull().ix[i,29]:#basement
        df.iloc[i,29]='nobasement'
        df.iloc[i,30]='nobasement'
        df.iloc[i,31]='nobasement'
        df.iloc[i,32]='nobasement'
        df.iloc[i,34]='nobasement'
for i in range(len( df.loc[:,'Condition1'])):
    if df.isnull().ix[i,57]:#Garage
        df.iloc[i,57]='nogarage'
        df.iloc[i,58]='nogarage'
        df.iloc[i,59]='nogarage'
        df.iloc[i,62]='nogarage'
        df.iloc[i,63]='nogarage'               
for i in range(len( df.loc[:,'Condition1'])):
    if df.isnull().ix[i,5]:#alley
        df.iloc[i,5]='noalley'
for i in range(len( df.loc[:,'Condition1'])):
    if df.isnull().ix[i,71]:#poolqc
        df.iloc[i,71]='nopool'
for i in range(len( df.loc[:,'Condition1'])):
    if df.isnull().ix[i,72]:#fence
        df.iloc[i,72]='nofence'   
for i in range(len( df.loc[:,'Condition1'])):
    if df.isnull().ix[i,73]:#miscfeature
        df.iloc[i,73]='nomiscfeature'   
for i in range(len( df.loc[:,'Condition1'])):
    if df.isnull().ix[i,56]:#fireplacequality
        df.iloc[i,56]='nofireplace'           
        
for i in range(len( df.loc[:,'Condition1'])):#condition
    if  df.iloc[i,13]== df.iloc[i,12]:
         df.iloc[i,13]='nocondition'
for i in range(len( df.loc[:,'Condition1'])):#exterior
    if  df.iloc[i,22]== df.iloc[i,23]:
         df.iloc[i,23]='noexterior'       

df['renovation']=np.zeros(len( df.loc[:,'Condition1']))#add_column_renovation
 
for i in range(len( df.loc[:,'Condition1'])):#renovation(1,0)
    if  df.iloc[i,18]!= df.iloc[i,19]:
         df.iloc[i,80]=1
df['diff_yearbuilt']=np.zeros(len( df.loc[:,'Condition1']))#add_diff_yearbuilt
df['diff_remodadd']=np.zeros(len( df.loc[:,'Condition1']))#add_diff_remodadd

for i in range(len( df.loc[:,'Condition1'])):
    df.iloc[i,81]=2019-df.iloc[i,18]
    df.iloc[i,82]=2019-df.iloc[i,19]
df.to_csv('D:/Arshad UT/Term03/RA/RA-Eco-UT/Emad Akrami/df.csv')

#plot missing data after preproceesing
df_na = (df.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)[:80]
missing_data = pd.DataFrame({'Missing Ratio' :df_na})
missing_data.head(80)
f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=df_na.index, y=df_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)

all_data=df

#drop NA
#all_data_dropna1=all_data
all_data_dropna1=all_data.dropna().reset_index() #18% missing value
z=all_data_dropna1.iloc[1076:,0]
#.loc[price_dropna.iloc[:,0]!=0,:]
all_data_dropna1=all_data_dropna1.drop(['index'], axis=1)

#read impute Data
all_data_impute=pd.read_csv('D:/Data/Data Mining/Data Minhig Class (Zerehsaz)/Data-Project/Data-Project01/Project/dfn.csv')
all_data_impute.rename({'Saleprice': 'SalePrice','X1stFlrSF': '1stFlrSF','X2ndFlrSF': '2ndFlrSF','X3SsnPorch': '3SsnPorch'}, axis=1, inplace=True)

#Seprate Price, YearBuilt & YearRemodAdd DATA
price_dropna= pd.DataFrame(all_data_dropna1.SalePrice, columns = ['SalePrice'])
price_impute= pd.DataFrame(all_data_impute.SalePrice, columns = ['SalePrice'])

all_data_dropna=all_data_dropna1.drop(['SalePrice','YearBuilt','YearRemodAdd'], axis=1)
all_data_impute=all_data_impute.drop(['SalePrice','YearBuilt','YearRemodAdd'], axis=1)

all_data_dropna['renovation']=all_data_dropna['renovation'].astype(object)
all_data_impute['renovation']=all_data_impute['renovation'].astype(object)

#backward-stepwise
dfnn=pd.read_csv('D:/Data/Data Mining/Data Minhig Class (Zerehsaz)/Data-Project/Data-Project01/Project/dfnn.csv')
all_data_dropna_stepwise1=all_data_dropna.loc[:,dfnn.columns]
all_data_impute_stepwise1=all_data_impute.loc[:,dfnn.columns]

#get dummy
all_data_dropna=pd.get_dummies(all_data_dropna)
all_data_impute=pd.get_dummies(all_data_impute)

all_data_dropna_stepwise=pd.get_dummies(all_data_dropna_stepwise1)
all_data_impute_stepwise=pd.get_dummies(all_data_impute_stepwise1)
#all_data_impute_stepwise.to_csv('D:/Arshad UT/Term03/RA/RA-Eco-UT/Emad Akrami/all_data_impute_stepwise.csv')
all_data_dropna_stepwise.to_csv('D:/Arshad UT/Term03/RA/RA-Eco-UT/Emad Akrami/all_data_dropna_stepwise.csv')

##splite Data
#X_train_dropna, X_test_dropna, y_train_dropna, y_test_dropna = train_test_split(all_data_dropna.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
#X_train_impute, X_test_impute, y_train_impute, y_test_impute = train_test_split(all_data_impute.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
#X_train_dropna_stepwise, X_test_dropna_stepwise, y_train_dropna_stepwise, y_test_dropna_stepwise = train_test_split(all_data_dropna_stepwise.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
#X_train_impute_stepwise, X_test_impute_stepwise, y_train_impute_stepwise, y_test_impute_stepwise = train_test_split(all_data_impute_stepwise.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)

'''
#PCA for Scaled Data
p=PCA() #n_components=17
p.fit(X_train.iloc[:,:36])
W=p.components_.T
y=p.fit_transform(X_train.iloc[:,:36])
yhat=X_train.iloc[:,:35].dot(W)
plt.figure(1)
plt.scatter(yhat.iloc[:,0],yhat.iloc[:,1],c="red",marker='o',alpha=0.5)
plt.xlabel('PC Scores 1')
plt.ylabel('PC Scores 2')

#ta 17 taye aval 81%
cat=pd.DataFrame(X_train.iloc[:,35:]).reset_index()
X_train = pd.concat([pd.DataFrame(y),cat], axis=1)
test=p.fit_transform(X_test.iloc[:,:35])
cat=pd.DataFrame(X_test.iloc[:,35:]).reset_index()
X_test = pd.concat([pd.DataFrame(test),cat], axis=1)

X_train=X_train.drop(['index'], axis=1)
X_test=X_test.drop(['index'], axis=1)

'''
iteration=1000
#Model1: LinearRegression
def model1(xtrain,xtest,ytrain,ytest):
    model = LinearRegression()
    model.fit(xtrain,ytrain)
    y_hat_train=model.predict(xtrain)
    y_hat_test=model.predict(xtest)
    r2_score_train=r2_score(ytrain, y_hat_train)
    r2_score_test=r2_score(ytest, y_hat_test)
    MSE=mean_squared_error(ytest, y_hat_test)
    MAE=mean_absolute_error(ytest, y_hat_test)
    return r2_score_train,r2_score_test, MSE, MAE
# dropna
r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_dropna, X_test_dropna, y_train_dropna, y_test_dropna = train_test_split(all_data_dropna.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
    output=model1(X_train_dropna, X_test_dropna, y_train_dropna, y_test_dropna)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_dropna1,r2_score_test_dropna1,MSE_dropna1, MAE_dropna1=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: dropna','\n','r2_score train for LinearRegression:',r2_score_train_dropna1,'\n','r2_score test for LinearRegression:',r2_score_test_dropna1,'\n','MSE for LinearRegression  :', MSE_dropna1,'\n','MAE for LinearRegression  :', MAE_dropna1,'\n')
#impute
r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_impute, X_test_impute, y_train_impute, y_test_impute = train_test_split(all_data_impute.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
    output=model1(X_train_impute, X_test_impute, y_train_impute, y_test_impute)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_impute1,r2_score_test_impute1,MSE_impute1, MAE_impute1=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: impute','\n','r2_score train for LinearRegression:',r2_score_train_impute1,'\n','r2_score test for LinearRegression:',r2_score_test_impute1,'\n','MSE for LinearRegression  :', MSE_impute1,'\n','MAE for LinearRegression  :', MAE_impute1,'\n')
# dropna_stepwise
r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_dropna_stepwise, X_test_dropna_stepwise, y_train_dropna_stepwise, y_test_dropna_stepwise = train_test_split(all_data_dropna_stepwise.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
    output=model1(X_train_dropna_stepwise, X_test_dropna_stepwise, y_train_dropna_stepwise, y_test_dropna_stepwise)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_dropna_stepwise1,r2_score_test_dropna_stepwise1,MSE_dropna_stepwise1, MAE_dropna_stepwise1=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: dropna_stepwise','\n','r2_score train for LinearRegression:',r2_score_train_dropna_stepwise1,'\n','r2_score test for LinearRegression:',r2_score_test_dropna_stepwise1,'\n','MSE for LinearRegression  :', MSE_dropna_stepwise1,'\n','MAE for LinearRegression  :', MAE_dropna_stepwise1,'\n')
# impute_stepwise
r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_impute_stepwise, X_test_impute_stepwise, y_train_impute_stepwise, y_test_impute_stepwise = train_test_split(all_data_impute_stepwise.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
    output=model1(X_train_impute_stepwise, X_test_impute_stepwise, y_train_impute_stepwise, y_test_impute_stepwise)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_impute_stepwise1,r2_score_test_impute_stepwise1,MSE_impute_stepwise1, MAE_impute_stepwise1=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: impute_stepwise','\n','r2_score train for LinearRegression:',r2_score_train_impute_stepwise1,'\n','r2_score test for LinearRegression:',r2_score_test_impute_stepwise1,'\n','MSE for LinearRegression  :', MSE_impute_stepwise1,'\n','MAE for LinearRegression  :', MAE_impute_stepwise1,'\n')

#Model2: lasso Regression 
def model2(xtrain,xtest,ytrain,ytest,alpha):
    model = Lasso(alpha=alpha,max_iter=5000)
    model.fit(xtrain,ytrain)
    y_hat_train=model.predict(xtrain)
    y_hat_test=model.predict(xtest)
    r2_score_train=r2_score(ytrain, y_hat_train)
    r2_score_test=r2_score(ytest, y_hat_test)
    MSE=mean_squared_error(ytest, y_hat_test)
    MAE=mean_absolute_error(ytest, y_hat_test)
    return r2_score_train,r2_score_test, MSE, MAE
# dropna
alpha= [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100 ]
X_train_dropna, X_test_dropna, y_train_dropna, y_test_dropna = train_test_split(all_data_dropna.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
cvscore=[]
for p in alpha:
    model = Lasso(alpha=p,max_iter=5000)
    cvscore.append([p,np.mean(cross_val_score(model, X_train_dropna, y_train_dropna, cv=5))])
cvscore=pd.DataFrame(cvscore)
plt.plot(cvscore.iloc[:,0],cvscore.iloc[:,1])
plt.xlabel('alphas') 
plt.ylabel('cvscore') 
plt.title('cross_validation_alpha')
bestalpha=cvscore[cvscore.iloc[:,1]==np.max(cvscore.iloc[:,1])].iloc[0,0]

r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_dropna, X_test_dropna, y_train_dropna, y_test_dropna = train_test_split(all_data_dropna.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
    output=model2(X_train_dropna, X_test_dropna, y_train_dropna, y_test_dropna,bestalpha)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_dropna2,r2_score_test_dropna2,MSE_dropna2, MAE_dropna2=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: dropna','\n','r2_score train for LinearRegression:',r2_score_train_dropna2,'\n','r2_score test for LinearRegression:',r2_score_test_dropna2,'\n','MSE for LinearRegression  :', MSE_dropna2,'\n','MAE for LinearRegression  :', MAE_dropna2,'\n')
#impute
alpha= [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]
X_train_impute, X_test_impute, y_train_impute, y_test_impute = train_test_split(all_data_impute.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
cvscore=[]
for p in alpha:
    model = Lasso(alpha=p,max_iter=5000)
    cvscore.append([p,np.mean(cross_val_score(model, X_train_impute, y_train_impute, cv=5))])
cvscore=pd.DataFrame(cvscore)
plt.plot(cvscore.iloc[:,0],cvscore.iloc[:,1])
plt.xlabel('alphas') 
plt.ylabel('cvscore') 
plt.title('cross_validation_alpha')
bestalpha=cvscore[cvscore.iloc[:,1]==np.max(cvscore.iloc[:,1])].iloc[0,0]

r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_impute, X_test_impute, y_train_impute, y_test_impute = train_test_split(all_data_impute.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
    output=model2(X_train_impute, X_test_impute, y_train_impute, y_test_impute,bestalpha)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_impute2,r2_score_test_impute2,MSE_impute2, MAE_impute2=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: impute','\n','r2_score train for LinearRegression:',r2_score_train_impute2,'\n','r2_score test for LinearRegression:',r2_score_test_impute2,'\n','MSE for LinearRegression  :', MSE_impute2,'\n','MAE for LinearRegression  :', MAE_impute2,'\n')
# dropna_stepwise
alpha= [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]
X_train_dropna_stepwise, X_test_dropna_stepwise, y_train_dropna_stepwise, y_test_dropna_stepwise = train_test_split(all_data_dropna_stepwise.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
cvscore=[]
for p in alpha:
    model = Lasso(alpha=p,max_iter=5000)
    cvscore.append([p,np.mean(cross_val_score(model, X_train_dropna_stepwise, y_train_dropna_stepwise, cv=5))])
cvscore=pd.DataFrame(cvscore)
plt.plot(cvscore.iloc[:,0],cvscore.iloc[:,1])
plt.xlabel('alphas') 
plt.ylabel('cvscore') 
plt.title('cross_validation_alpha')
bestalpha=cvscore[cvscore.iloc[:,1]==np.max(cvscore.iloc[:,1])].iloc[0,0]

r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_dropna_stepwise, X_test_dropna_stepwise, y_train_dropna_stepwise, y_test_dropna_stepwise = train_test_split(all_data_dropna_stepwise.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
    output=model2(X_train_dropna_stepwise, X_test_dropna_stepwise, y_train_dropna_stepwise, y_test_dropna_stepwise,bestalpha)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_dropna_stepwise2,r2_score_test_dropna_stepwise2,MSE_dropna_stepwise2, MAE_dropna_stepwise2=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: dropna_stepwise','\n','r2_score train for LinearRegression:',r2_score_train_dropna_stepwise2,'\n','r2_score test for LinearRegression:',r2_score_test_dropna_stepwise2,'\n','MSE for LinearRegression  :', MSE_dropna_stepwise2,'\n','MAE for LinearRegression  :', MAE_dropna_stepwise2,'\n')
# impute_stepwise
alpha= [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]
X_train_impute_stepwise, X_test_impute_stepwise, y_train_impute_stepwise, y_test_impute_stepwise = train_test_split(all_data_impute_stepwise.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
cvscore=[]
for p in alpha:
    model = Lasso(alpha=p,max_iter=5000)
    cvscore.append([p,np.mean(cross_val_score(model, X_train_impute_stepwise, y_train_impute_stepwise, cv=5))])
cvscore=pd.DataFrame(cvscore)
plt.plot(cvscore.iloc[:,0],cvscore.iloc[:,1])
plt.xlabel('alphas') 
plt.ylabel('cvscore') 
plt.title('cross_validation_alpha')
bestalpha=cvscore[cvscore.iloc[:,1]==np.max(cvscore.iloc[:,1])].iloc[0,0]

r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_impute_stepwise, X_test_impute_stepwise, y_train_impute_stepwise, y_test_impute_stepwise = train_test_split(all_data_impute_stepwise.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
    output=model2(X_train_impute_stepwise, X_test_impute_stepwise, y_train_impute_stepwise, y_test_impute_stepwise,bestalpha)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_impute_stepwise2,r2_score_test_impute_stepwise2,MSE_impute_stepwise2, MAE_impute_stepwise2=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: impute_stepwise','\n','r2_score train for LinearRegression:',r2_score_train_impute_stepwise2,'\n','r2_score test for LinearRegression:',r2_score_test_impute_stepwise2,'\n','MSE for LinearRegression  :', MSE_impute_stepwise2,'\n','MAE for LinearRegression  :', MAE_impute_stepwise2,'\n')

#Model3: ridge Regression 
def model3(xtrain,xtest,ytrain,ytest,alpha):
    model = Ridge(alpha=alpha,max_iter=5000)
    model.fit(xtrain,ytrain)
    y_hat_train=model.predict(xtrain)
    y_hat_test=model.predict(xtest)
    r2_score_train=r2_score(ytrain, y_hat_train)
    r2_score_test=r2_score(ytest, y_hat_test)
    MSE=mean_squared_error(ytest, y_hat_test)
    MAE=mean_absolute_error(ytest, y_hat_test)
    return r2_score_train,r2_score_test, MSE, MAE
# dropna
alpha= [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100 ]
X_train_dropna, X_test_dropna, y_train_dropna, y_test_dropna = train_test_split(all_data_dropna.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
cvscore=[]
for p in alpha:
    model = Lasso(alpha=p,max_iter=5000)
    cvscore.append([p,np.mean(cross_val_score(model, X_train_dropna, y_train_dropna, cv=5))])
cvscore=pd.DataFrame(cvscore)
plt.plot(cvscore.iloc[:,0],cvscore.iloc[:,1])
plt.xlabel('alphas') 
plt.ylabel('cvscore') 
plt.title('cross_validation_alpha')
bestalpha=cvscore[cvscore.iloc[:,1]==np.max(cvscore.iloc[:,1])].iloc[0,0]
    
r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_dropna, X_test_dropna, y_train_dropna, y_test_dropna = train_test_split(all_data_dropna.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
    output=model3(X_train_dropna, X_test_dropna, y_train_dropna, y_test_dropna,bestalpha)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_dropna3,r2_score_test_dropna3,MSE_dropna3, MAE_dropna3=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: dropna','\n','r2_score train for LinearRegression:',r2_score_train_dropna3,'\n','r2_score test for LinearRegression:',r2_score_test_dropna3,'\n','MSE for LinearRegression  :', MSE_dropna3,'\n','MAE for LinearRegression  :', MAE_dropna3)
#impute
alpha= [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]
X_train_impute, X_test_impute, y_train_impute, y_test_impute = train_test_split(all_data_impute.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
cvscore=[]
for p in alpha:
    model = Lasso(alpha=p,max_iter=5000)
    cvscore.append([p,np.mean(cross_val_score(model, X_train_impute, y_train_impute, cv=5))])
cvscore=pd.DataFrame(cvscore)
plt.plot(cvscore.iloc[:,0],cvscore.iloc[:,1])
plt.xlabel('alphas') 
plt.ylabel('cvscore') 
plt.title('cross_validation_alpha')
bestalpha=cvscore[cvscore.iloc[:,1]==np.max(cvscore.iloc[:,1])].iloc[0,0]

r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_impute, X_test_impute, y_train_impute, y_test_impute = train_test_split(all_data_impute.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
    output=model3(X_train_impute, X_test_impute, y_train_impute, y_test_impute,bestalpha)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_impute3,r2_score_test_impute3,MSE_impute3, MAE_impute3=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: impute','\n','r2_score train for LinearRegression:',r2_score_train_impute3,'\n','r2_score test for LinearRegression:',r2_score_test_impute3,'\n','MSE for LinearRegression  :', MSE_impute3,'\n','MAE for LinearRegression  :', MAE_impute3)
# dropna_stepwise
alpha= [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]
X_train_dropna_stepwise, X_test_dropna_stepwise, y_train_dropna_stepwise, y_test_dropna_stepwise = train_test_split(all_data_dropna_stepwise.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
cvscore=[]
for p in alpha:
    model = Lasso(alpha=p,max_iter=5000)
    cvscore.append([p,np.mean(cross_val_score(model, X_train_dropna_stepwise, y_train_dropna_stepwise, cv=5))])
cvscore=pd.DataFrame(cvscore)
plt.plot(cvscore.iloc[:,0],cvscore.iloc[:,1])
plt.xlabel('alphas') 
plt.ylabel('cvscore') 
plt.title('cross_validation_alpha')
bestalpha=cvscore[cvscore.iloc[:,1]==np.max(cvscore.iloc[:,1])].iloc[0,0]

r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_dropna_stepwise, X_test_dropna_stepwise, y_train_dropna_stepwise, y_test_dropna_stepwise = train_test_split(all_data_dropna_stepwise.loc[price_dropna.iloc[:,0]!=0,:],price_dropna.loc[price_dropna.iloc[:,0]!=0,:], test_size=0.1)
    output=model3(X_train_dropna_stepwise, X_test_dropna_stepwise, y_train_dropna_stepwise, y_test_dropna_stepwise,bestalpha)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_dropna_stepwise3,r2_score_test_dropna_stepwise3,MSE_dropna_stepwise3, MAE_dropna_stepwise3=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: dropna_stepwise','\n','r2_score train for LinearRegression:',r2_score_train_dropna_stepwise3,'\n','r2_score test for LinearRegression:',r2_score_test_dropna_stepwise3,'\n','MSE for LinearRegression  :', MSE_dropna_stepwise3,'\n','MAE for LinearRegression  :', MAE_dropna_stepwise3)
# impute_stepwise
alpha= [0.000001,0.00001,0.0001,0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100]
X_train_impute_stepwise, X_test_impute_stepwise, y_train_impute_stepwise, y_test_impute_stepwise = train_test_split(all_data_impute_stepwise.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
cvscore=[]
for p in alpha:
    model = Lasso(alpha=p,max_iter=5000)
    cvscore.append([p,np.mean(cross_val_score(model, X_train_impute_stepwise, y_train_impute_stepwise, cv=5))])
cvscore=pd.DataFrame(cvscore)
plt.plot(cvscore.iloc[:,0],cvscore.iloc[:,1])
plt.xlabel('alphas') 
plt.ylabel('cvscore') 
plt.title('cross_validation_alpha')
bestalpha=cvscore[cvscore.iloc[:,1]==np.max(cvscore.iloc[:,1])].iloc[0,0]

r2_score_train,r2_score_test, MSE, MAE=[],[],[],[]
for i in range(iteration):
    X_train_impute_stepwise, X_test_impute_stepwise, y_train_impute_stepwise, y_test_impute_stepwise = train_test_split(all_data_impute_stepwise.loc[price_impute.iloc[:,0]!=0,:],price_impute.loc[price_impute.iloc[:,0]!=0,:], test_size=0.1)
    output=model3( X_train_impute_stepwise, X_test_impute_stepwise, y_train_impute_stepwise, y_test_impute_stepwise,bestalpha)
    r2_score_train.append(output[0])
    r2_score_test.append(output[1])
    MSE.append(output[2])
    MAE.append(output[3])
r2_score_train_impute_stepwise3,r2_score_test_impute_stepwise3,MSE_impute_stepwise3, MAE_impute_stepwise3=np.mean(r2_score_train),np.mean(r2_score_test),np.mean(MSE),np.mean(MAE)
print('method: impute_stepwise','\n','r2_score train for LinearRegression:',r2_score_train_impute_stepwise3,'\n','r2_score test for LinearRegression:',r2_score_test_impute_stepwise3,'\n','MSE for LinearRegression  :', MSE_impute_stepwise3,'\n','MAE for LinearRegression  :', MAE_impute_stepwise3)

# test price
model = Lasso(alpha=0.001,max_iter=5000)
final_train=all_data_dropna_stepwise.loc[price_dropna.iloc[:,0]!=0,:]
model.fit(final_train,price_dropna.loc[price_dropna.iloc[:,0]!=0,:])
y_hat_test=pd.DataFrame(model.predict(all_data_dropna_stepwise.loc[price_dropna.iloc[:,0]==0,:]))
price_predict=np.exp(y_hat_test)
aa=all_data_dropna1.drop(['SalePrice'], axis=1).loc[price_dropna.iloc[:,0]==0,:].reset_index()
aa=aa.drop(['index'], axis=1)
test_data_with_price= pd.concat([aa, price_predict], axis=1)
test_data_with_price.rename({0: 'SalePrice'}, axis=1, inplace=True)
test_data_with_price.to_csv(r'D:/Data Mining/Data Minhig Class (Zerehsaz)/Data-Project/Data-Project01/Project/test_data_with_price1.csv')
test_data_with_price1=pd.concat([z,test_data_with_price.SalePrice],axis=1,ignore_index=True)
z1=z.reset_index()
test_data_with_price5=test_data_with_price['SalePrice']
test_data_with_price5=test_data_with_price5.head(1320)
test_data_with_price3['index']=z1['index']
test_data_with_price2['SalePrice']=test_data_with_price.SalePrice
test_data_with_price4=pd.concat([test_data_with_price5,z1],axis=1)
all_data1=all_data.reset_index()
result=pd.merge(all_data1,test_data_with_price4,how='left' ,on='index')
result1=result.iloc[1314:,:]
result.to_csv(r'D:/Data Mining/Data Minhig Class (Zerehsaz)/Data-Project/Data-Project01/Project/result.csv')
result2=result1.drop(['SalePrice_x','level_0','index'], axis=1)
result2=result2.reset_index()
result2=result2.drop(['index'], axis=1)
result2.to_csv(r'D:/Data Mining/Data Minhig Class (Zerehsaz)/Data-Project/Data-Project01/Project/project_result.csv')

#Part2
qq= pd.concat([all_data_dropna1.drop(['SalePrice'], axis=1).loc[price_dropna.iloc[:,0]!=0,:],np.exp(price_dropna.loc[price_dropna.iloc[:,0]!=0,:])], axis=1)     
all_final=pd.concat([qq, test_data_with_price], axis=0)
suggest=all_final.loc[all_final['BedroomAbvGr']==4,:]
suggest=suggest.loc[suggest['MSZoning']!='A',:]
suggest=suggest.loc[suggest['HouseStyle']=='2Story',:]
#suggest['area']=suggest['1stFlrSF']+suggest['2ndFlrSF']
suggest=suggest.loc[suggest['diff_yearbuilt']<=20,:]
suggest=suggest.loc[suggest['1stFlrSF']<=1060,:]
suggest=suggest.loc[suggest['1stFlrSF']>=860,:]
suggest.to_csv(r'seggest.csv)


#part3
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

#should check if the new data is highly collinear with others...
New_Add = sm.add_constant(all_data_dropna_stepwise, prepend=False)

#print(New_Add.columns.get_loc("LotArea"))
#print(New_Add.columns.get_loc("Volume"))
#print(New_Add.columns.get_loc("Market Cap"))

#model = LinearRegression()
#OLS=model.fit(X_train_dropna_stepwise,y_train_dropna_stepwise)
print(OLS.summary())
#to know if this "feature" should be added or not
#VIF=pd.DataFrame()
#for i in range(0,181):
#    vif = variance_inflation_factor(New_Add,i)
#    VIF.append(vif)
#print(vif_ETH, vif_MC)
#vif=np.ones(183)
vif = [variance_inflation_factor(np.array(X_train_dropna_stepwise), i) for i in range(X_train_dropna_stepwise.shape[1])]
print(vif)

#Scale
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler
y=X_train_dropna_stepwise.describe()
#time=data.iloc[:,0]
#data1=data.drop(['Timestamp'], axis=1)
X_train_dropna_stepwise_var=X_train_dropna_stepwise.iloc[:,0:21]
X_train_dropna_stepwise_var_scale=pd.DataFrame(scale(X_train_dropna_stepwise_var),columns=X_train_dropna_stepwise_var.columns)
yy=X_train_dropna_stepwise_scale.describe()
X_train_dropna_stepwise_cat=X_train_dropna_stepwise.iloc[:,22:].reset_index().drop(['index'],axis=1)
X_train_dropna_stepwise_scale=pd.concat([X_train_dropna_stepwise_var_scale,X_train_dropna_stepwise_cat], axis=1)

vif2= [variance_inflation_factor(np.array(X_train_dropna_stepwise_scale), i) for i in range(X_train_dropna_stepwise_scale.shape[1])]
print(vif2)

#vif3= [variance_inflation_factor(np.array(X_train_dropna_stepwise_scale), i) for i in range(X_train_dropna_stepwise_scale.shape[1])]

X_train_dropna_stepwise_PoolArea=X_train_dropna_stepwise.drop(['PoolArea'], axis=1)
vif2= [variance_inflation_factor(np.array(X_train_dropna_stepwise_PoolArea), i) for i in range(X_train_dropna_stepwise_PoolArea.shape[1])]
print(vif3)

X_train_dropna_stepwise1=X_train_dropna_stepwise.diff()
X_train_dropna_stepwise1=X_train_dropna_stepwise1.dropna()
vif1= [variance_inflation_factor(np.array(X_train_dropna_stepwise1), i) for i in range(X_train_dropna_stepwise1.shape[1])]
print(vif1)



#Model "New"
#modelNew = sm.OLS(np.log(Y), np.log(New_Add))
y_train_dropna_stepwise1=y_train_dropna_stepwise.tail(967)
modelNew = sm.OLS(y_train_dropna_stepwise1, X_train_dropna_stepwise1)
resultNew = modelNew.fit()
print(resultNew.summary())
#errors for this model
errors = resultNew.resid
print(resultNew.summary())

#
X_test_dropna_stepwise1=X_test_dropna_stepwise.diff()
X_test_dropna_stepwise1=X_test_dropna_stepwise1.dropna()
vif2= [variance_inflation_factor(np.array(X_test_dropna_stepwise1), i) for i in range(X_test_dropna_stepwise1.shape[1])]
print(vif2)
resultNew1 = modelNew.fit()
print(resultNew.summary())

mod_New = sm.OLS(y_train_dropna_stepwise, X_train_dropna_stepwise_PoolArea)
result_New = mod_New.fit()
print(result_New.summary())


#predicting the prices using this model
predictions = resultNew.predict(Y)
print(predictions)

#hetrodedasticity
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.diagnostic import het_white
import statsmodels.api as sm
from statsmodels.formula.api import ols
emad=X_train_dropna_stepwise_PoolArea['LotArea'].apply(np.log)
emad1=X_train_dropna_stepwise_PoolArea[ '1stFlrSF'].apply(np.log)
X_train_dropna_stepwise_PoolArea1=X_train_dropna_stepwise_PoolArea
X_train_dropna_stepwise_PoolArea1['LotArea']=emad
X_train_dropna_stepwise_PoolArea1['1stFlrSF']=emad1

mod_New1 = sm.OLS(y_train_dropna_stepwise, X_train_dropna_stepwise_PoolArea1)
result_New1 = mod_New.fit()
print(result_New1.summary())

 import statsmodels.api as sm
 import statsmodels.stats.diagnostic as sm_diagnostic
 from statsmodels.compat import lzip
e1   = result_New1.resid
e   = result_New.resid

BP_t = sm_diagnostic.het_breuschpagan(e, exog_het = mod_New1.exog)
print(pd.DataFrame(lzip(['LM statistic', 'p-value',  'F-value', 'F: p-value'], BP_t)))

BP_t = sm_diagnostic.het_breuschpagan(e, exog_het = mod_New.exog)
print(pd.DataFrame(lzip(['LM statistic', 'p-value',  'F-value', 'F: p-value'], BP_t)))

##
statecrime_df = sm.datasets.statecrime.load_pandas().data
f ='violent~hs_grad+poverty+single+urban'
statecrime_model = ols(formula=f, data=statecrime_df).fit()

statecrime_df = sm.datasets.statecrime.load_pandas().data
f ='violent~hs_grad+poverty+single+urban'
statecrime_model = ols(formula=f, data=statecrime_df).fit()

white_test = het_white(statecrime_model.resid,  statecrime_model.model.exog)
bp_test = het_breuschpagan(statecrime_model.resid, [statecrime_df.var1, statecrime_df.var2...])

labels = ['LM Statistic', 'LM-Test p-value', 'F-Statistic', 'F-Test p-value']
print(dict(zip(labels, white_test))

##
#y_train_dropna_stepwise2=list(y_train_dropna_stepwise1.iloc[:,0])
mod_wls = sm.WLS(np.array(y_train_dropna_stepwise1), np.array(X_train_dropna_stepwise), weights=1./(resultNew1.bse**2))
mod_wls = sm.WLS(y_train_dropna_stepwise1, X_train_dropna_stepwise, weights=1./(resultNew1.bse**2))

res_wls = mod_wls.fit()
print(res_wls.summary())


resultNew2 = modelNew.fit(cov_type='HC1')
statecrime_df = sm.datasets.statecrime.load_pandas().data
f ='violent~hs_grad+poverty+single+urban'
statecrime_model = ols(formula=f, data=statecrime_df).fit(cov_type='HC2')

statecrime_df = sm.datasets.statecrime.load_pandas().data
f ='violent~hs_grad+poverty+single+urban'
statecrime_model = ols(formula=f, data=statecrime_df).fit(cov_type='HC2')

white_test2 = het_white(statecrime_model.resid,  statecrime_model.model.exog)

resultNew2 = modelNew.fit(cov_type='HAC', maxlags=30)


resultNew3 = modelNew.fit(cov_type='HC3')
resultNew3.bse
resultNew3.t_test(....)

#visualize errors - looks normal! 
plt.hist(errors)
plt.title("Errors of Model ")
plt.show()

# save results in a png file ...
from PIL import Image, ImageDraw, ImageFont
image = Image.new('RGB', (800, 400))
draw = ImageDraw.Draw(image)
font = ImageFont.truetype("arial.ttf", 16)
draw.text((0, 0), str(resultNew.summary()), font=font)
image = image.convert('1') # bw
image = image.resize((600, 300), Image.ANTIALIAS)
image.save('output_New.png')
plt.rc('figure', figsize=(12, 7))
plt.text(0.01, 0.05, str(resultNew.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig('output_New.png')

type(train)
train.dtypes
dtype(train)

e.plot()
e1.mean()

emad_final=X_train_dropna_stepwise_PoolArea.drop(['WoodDeckSF','3SsnPorch','KitchenAbvGr','TotRmsAbvGrd','BedroomAbvGr','MasVnrArea'], axis=1)

mod_New2 = sm.OLS(y_train_dropna_stepwise, emad_final)
result_New2 = mod_New2.fit()
print(result_New2.summary())

e3   = result_New2.resid

BP_t = sm_diagnostic.het_breuschpagan(e3, exog_het = mod_New2.exog)
print(pd.DataFrame(lzip(['LM statistic', 'p-value',  'F-value', 'F: p-value'], BP_t)))
vif3= [variance_inflation_factor(np.array(emad_final), i) for i in range(emad_final.shape[1])]
