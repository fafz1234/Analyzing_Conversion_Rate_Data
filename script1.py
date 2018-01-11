
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("ggplot")
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,confusion_matrix

def loading(filename,num_objects):
    """ load single object from binary file """
    objects = []
    with open(filename, "rb") as file_input:
        for i in xrange(num_objects):
            objects.append(cPickle.load(file_input))
        return objects

dataframe = pd.read_csv("conversion_data.csv")

dataframe = dataframe.loc[dataframe.age <=90,:]

# encode as categorical features
X = pd.get_dummies(dataframe.loc[:,('country', 'age', 'new_user', 'source', 'total_pages_visited')])

y = dataframe.converted
y.mean() # 3.2% converted


Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.333)

# using a Decision Tree model
dt = DecisionTreeClassifier()
params = {
    "criterion":['gini','entropy'],
    "max_depth": [None,5,10],
    "min_samples_split": [2,10,20],
    "min_samples_leaf": [5,10,20]
}

searchcv = GridSearchCV(estimator=dt, param_grid =params, scoring="roc_auc", n_jobs=-1, verbose=1)
searchcv.fit(Xtrain,ytrain)

dt = searchcv.best_estimator_

print ('train score', dt.score(Xtrain,ytrain))
print (1 - ytrain.mean())

print ('train score', dt.score(Xtest,ytest))
print (1 - ytest.mean())

featnames = Xtrain.columns.values
featimportances = dt.feature_importances_
feat_importances = pd.DataFrame({"name":featnames,"importances":featimportances})
feat_importances = feat_importances[['name','importances']]# reorder the columns
feat_importances.sort_values(by="importances",inplace=True,ascending=False)

print (feat_importances)

############## use LR to detect feature importances
LR = LogisticRegressionCV(Cs = np.logspace(-3,3,7),
                            dual=False,
                            scoring='roc_auc',
                            max_iter=1000,
                            n_jobs=-1,
                            verbose=1)
LR.fit(Xtrain,ytrain)
LR.C_ # 10

ytest_predict = LR.predict(Xtest)
print (classification_report(y_true=ytest,y_pred=ytest_predict))

ytest_proba = LR.predict_proba(Xtest)

feat_importances = pd.DataFrame({"name":featnames,"coef":LR.coef_[0]})
feat_importances = feat_importances[['name','coef']]# reorder the columns
feat_importances['importances'] = np.abs( feat_importances['coef'] )






