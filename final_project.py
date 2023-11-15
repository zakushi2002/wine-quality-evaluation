#IMPORT AND FUNCTIONS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder      
from statistics import mean
import joblib 
import seaborn as sns

# GET THE DATA . LOAD DATA
wine = pd.read_csv('./winequality_train_set.csv')

# Quick view of the data
print('\n____________ Dataset info ____________')
print(wine.info())

# Có nhiều feature bị khuyết như: fixed acidity,volatile acidity...
# Hầu hết data có kiểu dữ liệu là float, trừ feature type có kiểu dữ liệu là string
# và feature quality có kiểu dữ liệu là int
# Dữ liệu yêu cầu 362,7+ KB bộ nhớ

#Exploratory Data Analysis
#Data Manipulation

#In 3 dòng đầu của dữ liệu
print('\n____________ Some first data examples ____________')
print(wine.head(3)) 

    
# Kiểm tra NULL trong dataset
wine.isna().sum()

## Missing Value Handling
# Fill Null data with mean value of each feature
#Replace the missing the values with the column mean
missing_val_cols = ["fixed acidity", "pH", "volatile acidity", "sulphates", "citric acid", "residual sugar", "chlorides"]
for col in missing_val_cols:
    mean = wine[col].mean()
    wine[col].fillna(mean, inplace=True)
    
#Thống kê dataset
print('\n____________ Statistics of numeric features ____________')
print(wine.describe())  

#Data features
print(wine.columns)

## Scatter plot between 2 features density and quality
wine.plot(kind="scatter", y="density", x="quality", alpha=0.2)
plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
plt.show() 

##  Citric acid vs quality
plt.bar(wine['quality'],wine['citric acid'])
plt.xlabel('quality')
plt.ylabel('citric acid')
plt.show()

# Plot histogram of features
fig = plt.figure(figsize=(15, 12))
plt.suptitle('Histograms of Numerical Columns', fontsize=20)
for i in range(wine.shape[1]):
    plt.subplot(6, 3, i + 1)
    f = plt.gca()
    f.set_title(wine.columns.values[i])

    vals = np.size(wine.iloc[:, i].unique())
    if vals >= 100:
        vals = 100
    plt.hist(wine.iloc[:, i], bins=vals, color='#3F5F7D')
plt.savefig('figures/hist_feat.png', format='png', dpi=300)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Plot histogram of numeric features
wine.hist(figsize=(15,10)) #bins: no. of intervals
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.tight_layout()
plt.savefig('figures/hist_raw_data.png', format='png', dpi=300) # must save before show()
plt.show()

#Compute correlations b/w features
corr_matrix = wine.corr()
print(corr_matrix) # print correlation matrix
print('\n',corr_matrix["quality"].sort_values(ascending=False))

#Correlation with Quality with respect to attributes
wine.corrwith(wine.quality).plot.bar(figsize = (20, 10), title = "Correlation with quality", fontsize = 15,rot = 45, grid = True)

#Check if we need to do Dimentionality reduction
sns.heatmap(wine.corr(),annot=True,cmap='terrain')
figure = plt.gcf()
figure.set_size_inches(20,10)
plt.savefig('figures/heatmap_wine.png', format='png', dpi=300)
plt.show()

# view values of quality's feature
wine['quality'].unique()
# Kết quả của cột quality nhiều nhưng ta sẽ chú trọng vào kết quả xác nhận rượu có tốt hay không, nên ta sẽ đưa số liệu về dạng nhị phân với 
# 0: không đủ tiêu chuẩn nếu quality < 6.5 và 1: đủ tiêu chuẩn với những trường hợp còn lại.

#Label Scaling
bins = (2, 6.5, 10)
group_names = ['không đủ tiêu chuẩn', 'đủ tiêu chuẩn']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)
label_quality = LabelEncoder()
wine['quality'] = label_quality.fit_transform(wine['quality'])

# Check number of samples of feature quality
wine['quality'].value_counts()

# Check the outlier using Boxplot
wine.boxplot(figsize=(15,8))
plt.show()

wine['residual sugar'].describe()
# Không loại bỏ các value dị biệt vì nó có thể làm ảnh hưởng đến việc phân loại chất lượng rượu

# Vì dữ liệu tất cả dữ liệu type đều là white nên ta có thể drop feature type để giảm độ nhiễu dữ liệu
wine.drop(columns='type',inplace = True)

## Partitioning
X = wine.drop('quality',axis=1) #Input data
y = wine['quality']

# View data
print(X)
print(y)

# Start training
from sklearn.model_selection  import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle = True, random_state = 2)

# View data
print(X_train)
print(X_test)
print(y_train)
print(y_test)

wine.describe()
# Nhìn qua dữ liệu ta thấy độ chênh lệch giữa các feature lớn và 
# độ tương quan khá nhỏ nên ta sẽ áp dụng chuẩn hoá

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# View data after normalization
print(X_train)
print(X_test)

#Using Principal Dimensional Reduction
from sklearn.decomposition import PCA
pca = PCA(n_components=0.9)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(pca.explained_variance_ratio)

sum(pca.explained_variance_ratio_)

## Training using models:
##Logictis regression
from sklearn.linear_model import LogisticRegression
lgt = LogisticRegression(random_state=0)
lgt.fit(X_train,y_train)
y_predict1 = lgt.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_predict1)
prec = precision_score(y_test, y_predict1)
rec = recall_score(y_test, y_predict1)
f1 = f1_score(y_test, y_predict1)
cm = confusion_matrix(y_test,y_predict1)
results = pd.DataFrame([['Logistic Regression', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(results)

## SVC(linear)
from sklearn.svm import SVC
svc = SVC(random_state = 0, kernel = 'linear')
svc.fit(X_train,y_train)
y_predict2 = svc.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_predict2)
prec = precision_score(y_test, y_predict2)
rec = recall_score(y_test, y_predict2)
f1 = f1_score(y_test, y_predict2)
cm = confusion_matrix(y_test,y_predict2)
model_results = pd.DataFrame([['SVM (Linear)', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)

## SVC(RBF)
from sklearn.svm import SVC
svc = SVC(random_state = 0, kernel = 'rbf')
svc.fit(X_train,y_train)
y_predict6 = svc.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_predict6)
prec = precision_score(y_test, y_predict6)
rec = recall_score(y_test, y_predict6)
f1 = f1_score(y_test, y_predict6)
cm = confusion_matrix(y_test,y_predict6)
model_results = pd.DataFrame([['SVM (RBF)', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)

## KNeighbors Classifier
from sklearn.neighbors import KNeighborsClassifier
Knn = KNeighborsClassifier()
Knn.fit(X_train,y_train)
y_predict3 = Knn.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_predict3)
prec = precision_score(y_test, y_predict3)
rec = recall_score(y_test, y_predict3)
f1 = f1_score(y_test, y_predict3)
cm = confusion_matrix(y_test,y_predict3)
model_results = pd.DataFrame([['KNeighborsClassifier', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)

##Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion='entropy', min_samples_split=10, splitter='best')
dtc.fit(X_train,y_train)
y_predict4 = dtc.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_predict4)
prec = precision_score(y_test, y_predict4)
rec = recall_score(y_test, y_predict4)
f1 = f1_score(y_test, y_predict4)
cm = confusion_matrix(y_test,y_predict4)
model_results = pd.DataFrame([['DecisionTree', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state = 0, n_estimators = 100, criterion = 'entropy')
rfc.fit(X_train,y_train)
y_predict5 = rfc.predict(X_test)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
acc = accuracy_score(y_test, y_predict5)
prec = precision_score(y_test, y_predict5)
rec = recall_score(y_test, y_predict5)
f1 = f1_score(y_test, y_predict5)
cm = confusion_matrix(y_test,y_predict5)
model_results = pd.DataFrame([['RandomForest', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)

# Show result of classification as well as data report
from sklearn.metrics import classification_report
print('Logistic Regression:')
print(classification_report(y_test, y_predict1))
print('SVM (Linear):')
print(classification_report(y_test, y_predict2))
print('SVM (RBF):')
print(classification_report(y_test, y_predict3))
print('KNeighborsClassifier:')
print(classification_report(y_test, y_predict4))
print('DecisionTree:')
print(classification_report(y_test, y_predict5))
print('RandomForest:')
print(classification_report(y_test, y_predict6))

# So the best model is Random Forest

# Show feature importances of Random Forest Model

rfc.feature_importances_

#Fine-tune model Random Forest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

rf = RandomForestClassifier()
# criterion=["entropy","gini"]
# min_samples_split = range(1,10)
# splitter=['best','random']

parameter = param_grid = [
            {'bootstrap': [True], 'n_estimators': [3, 15, 30], 'max_features': [2, 12, 20, 39]},
            {'bootstrap': [False], 'n_estimators': [3, 5, 10, 20], 'max_features': [2, 6, 10]} ]

cv = RepeatedStratifiedKFold(n_splits=5,random_state=100)

grid_search_cv_rf = GridSearchCV(rf,param_grid=parameter, scoring='accuracy', cv=cv, return_train_score=True, refit=True)
grid_search_cv_rf.fit(X_train,y_train)
joblib.dump(grid_search_cv_rf,'figures/RandomForestRegressor_gridsearch.pkl')

print(f'Best: {grid_search_cv_rf.best_score_:.3f} using {grid_search_cv_rf.best_params_}')
means = grid_search_cv_rf.cv_results_['mean_test_score']
stds = grid_search_cv_rf.cv_results_['std_test_score']
params = grid_search_cv_rf.cv_results_['params']

for mean,stdev,params in zip(means, stds, params):
    print(f"{mean:.3f}({stdev:.3f}) with: {params}")
print("Training score: ",grid_search_cv_rf.score(X_train,y_train)*100)
print("Testing score: ",grid_search_cv_rf.score(X_test,y_test)*100)

#Run on test data
new_data = pd.DataFrame({'fixed acidity':5.23,'volatile acidity':0.5,'citric acid':0.1,'residual sugar':8.4,'chlorides':0.04,'free sulfur dioxide':42.2,'total sulfur dioxide':143.6,'density':0.5,'pH':3.1,'sulphates':0.4,'alcohol':12},index=[0])
test = pca.transform(scaler.transform(new_data))
p=rfc.predict(test)
if p[0]==1:
    print('good quality wine')
else: print('bad quality wine')

#Link video thuyết trình của nhóm: https://youtu.be/3yspKknHO5A