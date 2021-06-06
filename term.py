import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore") #ignore warnings

#matrix
def matrix(temp,d):
    predict_matrix=pd.DataFrame(temp)
    result=predict_matrix
    error=0
    result=result.astype(int)
    print("\nconfusion matrix")
    c_matrix=confusion_matrix(d,result)
    print(c_matrix,"\n")

#LabelEncoder
def convertToContinous(data, column):
    label = LabelEncoder()
    convert = label.fit_transform(data[column])
    data.loc[:, column] = convert
    return data

#convert numerous data using oneHot Encoding
def oneHotEncoding(data, column):
    oneHotResult = pd.get_dummies(data[column])
    for i in range(len(oneHotResult.columns)):
        data[column + '_' + oneHotResult.columns[i]] = oneHotResult.iloc[:, i]
    data = data.drop(column, axis = 1)
    return data

#scaler
def scaler(data,ecd,count):

    if(count==3):
        scaler = preprocessing.StandardScaler()
        standard_data=scaler.fit_transform(data[['Category','Rating','Rating Count',
                                     'Installs','Price','Content Rating',
                                     'Ad Supported','In App Purchases']])
        standard_data=pd.DataFrame(standard_data,columns=['Category','Rating','Rating Count',
                                     'Installs','Price','Content Rating',
                                     'Ad Supported','In App Purchases'])
        return standard_data
        
    elif (count==2):
        scaler = preprocessing.MinMaxScaler()
        minmax_data=scaler.fit_transform(data[['Category','Rating','Rating Count',
                                     'Installs','Price','Content Rating',
                                     'Ad Supported','In App Purchases']])
        minmax_data=pd.DataFrame(minmax_data,columns=['Category','Rating','Rating Count',
                                     'Installs','Price','Content Rating',
                                     'Ad Supported','In App Purchases'])
        return minmax_data
 
    elif (count==1):
        scaler = preprocessing.MaxAbsScaler()
        maxabs_data=scaler.fit_transform(data[['Category','Rating','Rating Count',
                                     'Installs','Price','Content Rating',
                                     'Ad Supported','In App Purchases']])
        maxabs_data=pd.DataFrame(maxabs_data,columns=['Category','Rating','Rating Count',
                                     'Installs','Price','Content Rating',
                                     'Ad Supported','In App Purchases'])
        return maxabs_data

    elif (count==0):
        scaler = preprocessing.RobustScaler()
        robust_data=scaler.fit_transform(data[['Category','Rating','Rating Count',
                                     'Installs','Price','Content Rating',
                                     'Ad Supported','In App Purchases']])
        robust_data=pd.DataFrame(robust_data,columns=['Category','Rating','Rating Count',
                                     'Installs','Price','Content Rating',
                                     'Ad Supported','In App Purchases'])
        return robust_data

#algorithm
def Analysis(scaled_data,data,count,ecd):
    
    p_data =scaled_data
    x=p_data
    y=data['Maximum Installs'].values #taregt data

    x_train,x_test,y_train,y_test=train_test_split(x,y, random_state = 1)

    if(count==1):
        #DecisionTreeClassifier
        from sklearn.tree import DecisionTreeClassifier
        dt_modelr=DecisionTreeClassifier(random_state=0).fit(x_train,y_train)

        #Accuracy
        print('\n<<<<<<',ecd,' + DecisionTreeClassifier',">>>>>>\n")
        print("Accuracy : {:.3f}".format(dt_modelr.score(x_test,y_test)))
        y_pred = dt_modelr.predict(x_test)

        #Matrix
        print("\nconfusion matrix")
        conf_matrix = confusion_matrix(y_test, y_pred)
        #print(conf_matrix)

        #f1 score
        #print("f1 score : {:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    if(count==2):
        #DecisionTreeRegressor
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import cross_val_score
        from sklearn.tree import DecisionTreeRegressor
        from sklearn.ensemble import BaggingClassifier
        
        regressor = DecisionTreeRegressor(random_state=0).fit(x_train,y_train)
        print('\n<<<<<<',ecd,' + DecisionTreeRegressor',">>>>>>\n")

        #Accuracy
        print("Accuracy : {:.3f}".format(regressor.score(x_test,y_test)))
        y_pred = regressor.predict(x_test)

        #Matrix
        #matrix(y_pred,y_test)

        #check
        df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
        #print(df)

    if(count==3):
        #KNeighborsClassifier
        from sklearn.neighbors import KNeighborsClassifier

        print('\n<<<<<<',ecd,' + KNeighborsClassifier',">>>>>>\n")
        
        import math
        #print(math.sqrt(len(y_test)))
        # 보통 k를 sqrt(n)으로 정함.(보통 짝수로 정함) 
        # 현재 data size는 353.98305044168427이므로 k를 짝수인 354로 정함.

        # Define the model: Init K-NN
        classifier = KNeighborsClassifier(n_neighbors=34, p=2, metric='euclidean').fit(x_train,y_train)

        #Accuracy
        y_pred = classifier.predict(x_test)
        print("Accuracy : {:.3f}".format(accuracy_score(y_test, y_pred)))

        #Matrix
        cm = confusion_matrix(y_test, y_pred)
        #print(cm)

        #f1 score
        #print("f1 score : {:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    if(count==4):
        #Knn regressor
        from sklearn.neighbors import KNeighborsRegressor

        print('\n<<<<<<',ecd,' + KNeighborsRegressor',">>>>>>\n")
        clf = KNeighborsRegressor(n_neighbors=5).fit(x_train, y_train)

        #Accuracy
        print("Accuracy : {:.3f}".format(clf.score(x_test,y_test)))
        y_pred = clf.predict(x_test)

        #Matrix
        #matrix(y_pred,y_test)

        #check
        df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
        #print(df)

    if(count==5):
        #BaggingClassifier
        from sklearn.ensemble import BaggingClassifier
        bg_modelr = BaggingClassifier(random_state=0).fit(x_train,y_train)

        print('\n<<<<<<',ecd,' + BaggingClassifier',">>>>>>\n")

        #Accuracy
        y_pred = bg_modelr.predict(x_test)
        print("Accuracy : {:.3f}".format(bg_modelr.score(x_test, y_test)))

        #Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        #print(conf_matrix)

        #f1 score
        #print("f1 score : {:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    if(count==6):
        #BaggingRegressor
        from sklearn.ensemble import BaggingRegressor
        from sklearn.datasets import make_regression
        
        regr = BaggingRegressor(random_state=0).fit(x_train,y_train)

        print('\n<<<<<<',ecd,' + BaggingRegressor',">>>>>>\n")

        #Accuracy
        print("Accuracy : {:.3f}".format(regr.score(x_test,y_test)))
        y_pred = regr.predict(x_test)

        #Matrix
        #matrix(y_pred,y_test)

        #check
        df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
        #print(df)

    if(count==7):
        #RandomForestClassifier
        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(n_estimators = 50, random_state = 1).fit(x_train,y_train)

        print('\n<<<<<<',ecd,' + RandomForestClassifier',">>>>>>\n")

        #Accuracy
        y_pred = rf.predict(x_test)
        print("Accuracy : {:.3f}".format(rf.score(x_test, y_test)))

        #Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        #print(conf_matrix)

        #f1 score
        #print("f1 score : {:.3f}".format(f1_score(y_test, y_pred, average='macro')))

    if(count==8):
        #RandomForestRegressor
        from sklearn.ensemble import RandomForestRegressor
        
        regr = RandomForestRegressor(max_depth=2, random_state=0).fit(x_train,y_train)

        print('\n<<<<<<',ecd,' + RandomForestRegressor',">>>>>>\n")

        #Accuracy
        print("Accuracy : {:.3f}".format(regr.score(x_test,y_test)))
        y_pred = regr.predict(x_test)

        #Matrix
        #matrix(y_pred,y_test)

        #check
        df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
        #print(df)

    if(count==9):
        #LinearRegression
        from sklearn.linear_model import LinearRegression
        regr = LinearRegression(fit_intercept = True, normalize = True, copy_X = True).fit(x_train,y_train)

        print('\n<<<<<<',ecd,' + LinearRegression',">>>>>>\n")

        #Accuracy
        y_pred = regr.predict(x_test)
        print("Accuracy : {:.3f}".format(regr.score(x_test, y_test)))

        #Matrix
        #matrix(y_pred,y_test)

        #check
        df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
        #print(df)

    if(count==10):
        #LogisticRegression
        from sklearn.linear_model import LogisticRegression
        
        regr = LogisticRegression(max_iter = 1000, random_state = 0).fit(x_train,y_train)

        print('\n<<<<<<',ecd,' + LogisticRegression',">>>>>>\n")

        #Accuracy
        print("Accuracy : {:.3f}".format(regr.score(x_test,y_test)))
        y_pred = regr.predict(x_test)

        #Matrix
        #matrix(y_pred,y_test)

        #check
        df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})
        #print(df)

data = pd.read_csv(r'C:\Users\82102\OneDrive\바탕 화면/DS_Google-Playstore.csv',encoding='ISO-8859-1')
#print(data)

#<<<<<<<<<<Drop wrong Data>>>>>>>>>>>>>
#replace missing values
data.replace('',np.nan,inplace=True)

#Delete unused columns
data = data.drop(['App Name','App Id','Minimum Installs','Minimum Android','Currency','Size',
                  'Minimum Android','Developer Id','Developer Website','Developer Email','Released',
                  'Last Updated','Privacy Policy','Editors Choice'],axis=1)

data.dropna(inplace=True)

#Drop pushed data
processing = data[(data.iloc[:,5]!='TRUE')&(data.iloc[:,5]!='FALSE')].index
data=data.drop(processing)

#Delete unused columns
data = data.drop(['Free'],axis=1)

#Replace a data type with an float 
list = ["Rating","Rating Count", "Maximum Installs", "Price"]
for i in list:
    data[i] = data[i].astype(float)
    
#Drop outlier
idx=data[data['Rating']==0].index
data.drop(idx,inplace=True)

idx2=data[data['Maximum Installs']<100].index
data.drop(idx2,inplace=True)

idx3=data[data['Maximum Installs']>100000000].index
data.drop(idx3,inplace=True)

#print('\n',data)
#Convert categorical Data
categorical_list = ['Category','Installs', 'Content Rating', 'Ad Supported','In App Purchases']
scaled_data = pd.DataFrame()

def combination(data, count, ecd):

    data=data.loc[0:1000]
    
    for i in categorical_list:
        scaled_data=convertToContinous(data,i)

    scaled_data = scaler(data, ecd, count)

    if(count==3):
        ecd+=" + StandatdScale"
    elif(count==2):
        ecd+=" + MinMaxScale"
    elif(count==1):
        ecd+=" + MaxAbsScale"
    elif(count==0) :
        ecd+=" + RobustScale"

    for i in range(1,11):
        Analysis(scaled_data,data,i,ecd)
    

    #result = Analysis(scaled_data,data,1)

    
#scaled_data = combination(data, 3, "labeling")
combination(data, 3, "labeling")
combination(data, 2, "labeling")
combination(data, 1, "labeling")
combination(data, 0, "labeling")

#data=oneHotEncoding(data, 'Category')
#data=oneHotEncoding(data, 'Installs')
#data=oneHotEncoding(data, 'Content Rating')
#data=oneHotEncoding(data, 'Ad Supported')
#data=oneHotEncoding(data, 'In App Purchases')




