import pandas as pd    
import numpy as np
import nltk
import string
from sklearn import model_selection 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report , accuracy_score , confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

filecsv = 'dataDistribusi.csv'
teks = pd.read_csv(filecsv, header=0, delimiter=",", encoding='utf-8')
df = pd.DataFrame(teks)
print(df)

xTarget = df.drop(['dscription', 'codeBars', 'mnfctrCode', 'salesItem', 'purchItem', 'returnItem', 'classCode', 
                   'length', 'length1', 'width', 'width1', 'height', 'height1', 'weight1', 'weight2',
                   'locked', 'auditDate', 'auditUser', 'backupSts', 'isFocus', 'isTM', 'isNPL', 
                   'longdesc', 'forCod'], axis=1)
print(xTarget)

yTarget = df['purchItem']
print(yTarget)

from sklearn.preprocessing import LabelBinarizer, OrdinalEncoder
encoder = LabelBinarizer()
Y = encoder.fit_transform(yTarget)
print(Y.shape)

tfidf_transformer = OrdinalEncoder()
X = tfidf_transformer.fit_transform(xTarget)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

NaiveBayes = MultinomialNB().fit(X_train,np.ravel(y_train,order='C'))
Knn       = KNeighborsClassifier(n_neighbors = 3).fit(X_train,np.ravel(y_train,order='C'))
RandomForest = RandomForestClassifier(n_estimators=50, max_depth=3).fit(X_train,np.ravel(y_train,order='C'))
DTree =    DecisionTreeClassifier().fit(X_train,np.ravel(y_train,order='C'))
MultiLP =   MLPClassifier(max_iter= 100).fit(X_train,np.ravel(y_train,order='C'))
SuppVM =  SVC(gamma='scale', decision_function_shape='ovo', kernel = 'linear').fit(X_train,np.ravel(y_train,order='C'))

models = [
    NaiveBayes,
    Knn,      
    RandomForest, 
    DTree, 
    MultiLP, 
    SuppVM, 
] 
dlist = [
    'NB',
    'K-NN',      
    'RF', 
    'DT', 
    'MLP', 
    'SVM', 
]

i = 0
print(y_test.shape)
entries = []
for model in models:
  prediction = model.predict(X_test)
  accuracies = accuracy_score(y_test, prediction)
  nameS = dlist[i]
  entries.append((nameS, model, accuracies))
  i = i + 1

cv_df = pd.DataFrame(entries, columns=['Classifier','prediction','accuracy'])
cv_df.to_csv('akurasi.csv')
print(cv_df)


sns.boxplot(x='Classifier', y='accuracy', data=cv_df)
sns.stripplot(x='Classifier', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)

print(cv_df.groupby('accuracy').accuracy.mean())
plt.show()
