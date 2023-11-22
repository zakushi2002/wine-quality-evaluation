# Import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder      
from statistics import mean
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import joblib 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold


# Load data
wine = pd.read_csv('./data.csv')

# Quick view of the data
print('\n__________________ Dataset info __________________')
print(wine.info())

# There are many missing features such as: fixed acidity, volatile acidity...
# Most data has the data type float, except for feature type which has data type string and feature quality which has data type int.
# Data requires 162.5+ KB of memory

# Exploratory Data Analysis
# Data Manipulation

# Print the first 3 lines of data
print('\n____________ Some first data examples ____________')
print(wine.head(3)) 

print(wine['quality'].value_counts()) 

wine.shape
# The data has 10000 samples and 13 features
print("Number of samples: ", wine.shape[0])
print("Number of features: ", wine.shape[1])
    
# Check for NULL in dataset
wine.isna().sum()

# Missing Value Handling
# Fill Null data with mean value of each feature
# Replace the missing the values with the column mean
missing_val_cols = ["fixed acidity", "pH", "volatile acidity", "sulphates", "citric acid", "residual sugar", "chlorides"]
for col in missing_val_cols:
    mean = wine[col].mean()
    wine[col].fillna(mean, inplace=True)
    
# Check again for Null value
wine.isna().any()
# Null value has been filled

# Statistics dataset
print('\n_______________ Statistics of numeric features _______________')
print(wine.describe())

# Data features
wine.columns

# Scatter plot between 2 features density and quality
wine.plot(kind="scatter", y="density", x="quality", alpha=0.2)
plt.savefig('figures/scatter_1_feat.png', format='png', dpi=300)
plt.show() 

# Citric acid vs quality
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
wine.hist(figsize=(15,10)) # bins: no. of intervals
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.tight_layout()
plt.savefig('figures/hist_raw_data.png', format='png', dpi=300) # must save before show()
plt.show()

# Drop feature type to reduce data noise
wine.drop(columns='type',inplace = True)

# Compute correlations between features
corr_matrix = wine.corr()
# print correlation matrix
print(corr_matrix) 
print('\n',corr_matrix["quality"].sort_values(ascending=False))

# Correlation with quality with respect to attributes
wine.corrwith(wine.quality).plot.bar(
    figsize = (20, 10), title = "Correlation with quality", fontsize = 15,
    rot = 45, grid = True)

# Check if we need to do Dimensionality reduction
sns.heatmap(wine.corr(),annot=True,cmap='terrain')
figure = plt.gcf()
figure.set_size_inches(20,10)
plt.savefig('figures/heatmap_wine.png', format='png', dpi=300)
plt.show()


# view values of quality's feature
wine['quality'].unique()
# The results of the quality column are many, but we will focus on the results to confirm whether the wine is good or not, 
# so we will put the data in binary form with 0: unqualified if quality < 6.5 and 1: qualified with the remaining cases.

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
# Do not remove outlier values as it may affect the classification of wine quality.

# Vì dữ liệu tất cả dữ liệu type đều là white nên ta có thể drop feature type để giảm độ nhiễu dữ liệu
wine.drop(columns='type',inplace = True)

# Partitioning
X = wine.drop('quality',axis=1) #Input data
y = wine['quality']

# View data
print(X)
print(y)

# Start training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, shuffle = True, random_state = 2)

# View data
print(X_train)
print(X_test)
print(y_train)
print(y_test)

wine.describe()
# Looking through the data, we see that the difference between the features is large and the correlation is quite small, 
# so we will apply normalization.

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# View data after normalization
print(X_train)
print(X_test)

# Using Principal Components Analysis (Principal Dimensional Reduction)
pca = PCA(n_components=0.9)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

print(pca.explained_variance_ratio)

sum(pca.explained_variance_ratio_)

## Training using models:
# Logictis regression
lgt = LogisticRegression(random_state=0)
lgt.fit(X_train,y_train)
y_predict1 = lgt.predict(X_test)

acc = accuracy_score(y_test, y_predict1)
prec = precision_score(y_test, y_predict1)
rec = recall_score(y_test, y_predict1)
f1 = f1_score(y_test, y_predict1)
cm = confusion_matrix(y_test,y_predict1)
results = pd.DataFrame([['Logistic Regression', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])
print(results)

# SVM (Linear) - SVC
svc = SVC(random_state = 0, kernel = 'linear')
svc.fit(X_train,y_train)
y_predict2 = svc.predict(X_test)

acc = accuracy_score(y_test, y_predict2)
prec = precision_score(y_test, y_predict2)
rec = recall_score(y_test, y_predict2)
f1 = f1_score(y_test, y_predict2)
cm = confusion_matrix(y_test,y_predict2)
model_results = pd.DataFrame([['SVM (Linear)', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)

# SVM (RBF) - SVC
svc = SVC(random_state = 0, kernel = 'rbf')
svc.fit(X_train,y_train)
y_predict6 = svc.predict(X_test)

acc = accuracy_score(y_test, y_predict6)
prec = precision_score(y_test, y_predict6)
rec = recall_score(y_test, y_predict6)
f1 = f1_score(y_test, y_predict6)
cm = confusion_matrix(y_test,y_predict6)
model_results = pd.DataFrame([['SVM (RBF)', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)

# KNeighbors Classifier
Knn = KNeighborsClassifier()
Knn.fit(X_train,y_train)
y_predict3 = Knn.predict(X_test)

acc = accuracy_score(y_test, y_predict3)
prec = precision_score(y_test, y_predict3)
rec = recall_score(y_test, y_predict3)
f1 = f1_score(y_test, y_predict3)
cm = confusion_matrix(y_test,y_predict3)
model_results = pd.DataFrame([['KNeighborsClassifier', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)

# Decision Tree Classifier
dtc = DecisionTreeClassifier(criterion='entropy', min_samples_split=10, splitter='best')
dtc.fit(X_train,y_train)
y_predict4 = dtc.predict(X_test)

acc = accuracy_score(y_test, y_predict4)
prec = precision_score(y_test, y_predict4)
rec = recall_score(y_test, y_predict4)
f1 = f1_score(y_test, y_predict4)
cm = confusion_matrix(y_test,y_predict4)
model_results = pd.DataFrame([['DecisionTree', acc*100, prec, rec, f1]],
               columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score'])

results = results.append(model_results, ignore_index = True)
print(results)

# Random Forest Classifier
rfc = RandomForestClassifier(random_state = 0, n_estimators = 100, criterion = 'entropy')
rfc.fit(X_train,y_train)
y_predict5 = rfc.predict(X_test)

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

# => So the best model is Random Forest

# Show feature importances of Random Forest Model

rfc.feature_importances_

#Fine-tune model Random Forest

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

# Run on test data
new_data = pd.DataFrame({'fixed acidity':5.23,'volatile acidity':0.5,'citric acid':0.1,'residual sugar':8.4,'chlorides':0.04,'free sulfur dioxide':42.2,'total sulfur dioxide':143.6,'density':0.5,'pH':3.1,'sulphates':0.4,'alcohol':12},index=[0])
test = pca.transform(scaler.transform(new_data))
p=rfc.predict(test)
if p[0]==1:
    print('good quality wine')
else: print('bad quality wine')

new_data1 = pd.DataFrame({'fixed acidity':7.3,'volatile acidity':0.23,'citric acid':0.27,'residual sugar':2.6,'chlorides':0.035,'free sulfur dioxide':39,'total sulfur dioxide':120,'density':0.99138,'pH':3.04,'sulphates':0.59,'alcohol':11.3},index=[0])
test1 = pca.transform(scaler.transform(new_data1))
p1=rfc.predict(test1)
if p1[0]==1:
    print('good quality wine')
else: print('bad quality wine')