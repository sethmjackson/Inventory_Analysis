### Problem 5: Fitting Models
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from typing import List, Dict, Callable
import pandas as pd
import Modules.Util as ut
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
from sklearn.metrics import confusion_matrix

# - You can use any binary classification method you have learned so far.
# - Use 80/20 training and test splits to build your model.
# - Double check the column types before you fit the model.
# - Only include useful features. i.e all the `ID`s should be excluded from your training set.
# - Note that there are only less than 5% of the orders have been returned, so you should consider using the
# from `caret` package and [StratifiedKfold](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn-model-selection-stratifiedkfold)
# from sklearn when running cross-validation.

# - Do forget to `set.seed()` before the spilt to make your result reproducible.
# - **Note:** We are not looking for the best tuned model in the lab so don't spend too much time on grid search. Focus on model evaluation and the business use case of each model.

class Regression:
    def __init__(self, model, name: str, hyperparams={}):
        self.model = model
        self.name = name
        self.hyperparams = hyperparams
        self.bestParams: Dict
        self.time=''

    def fitCV(self,xTrain, xTest, yTrain, yTest, cv=5):
        print('Starting ', self.name)


        grid = GridSearchCV(self.model, self.hyperparams, cv=cv, return_train_score = False, n_jobs=-1)
        self.modelCV = grid.fit(xTrain,yTrain)
        self.bestParams = self.modelCV.best_params_
        self.trainScore = self.modelCV.best_estimator_.score(xTrain, yTrain)
        self.testScore = self.modelCV.best_estimator_.score(xTest, yTest)

        # if 'max_depth' in self.hyperparams:
        #     featureSelection = self.model.feature_importances_
        #     print(featureSelection)





    def plotHyperParams(self, trainX, testX, trainY, testY, i):
        plt.clf()
        for name, params in self.hyperparams.items():
            coefs = []
            intercepts = []
            trainScore = []
            testScore = []

            if len(params) < 2:
                continue

            for value in params:
                self.model.set_params(**{name: value})
                #print(name, '   Value: ',value)
                self.model.fit(trainX, trainY)
           # intercepts.append(self.model.intercept_)
           # coefs.append(self.model.coef_)
                trainScore.append(self.model.score(trainX, trainY))
                testScore.append(self.model.score(testX, testY))

            plt.plot(params, trainScore, label=r'train set $R^2$')
            plt.plot(params, testScore, label=r'test set $R^2$')

            plt.xlabel(name+' Value')
            plt.ylabel('R^2 Value')
            plt.title(self.name+' R^2 VS. '+ name)
            plt.legend(loc=4)
            plt.savefig('Plots/Regressions/'+str(i)+' - '+self.name+' '+name+'.png')
            plt.clf()


def assembleModels():

    models = {
        # 'SVM': Regression(SVC(), 'Support Vector Classifier',
        #        {'C': np.linspace(1, 100, 10),
        #         'gamma': np.linspace(1e-7, 0.1, 10)}),

    'Log': Regression(LogisticRegression(), 'Logistic Regression', {'C': np.linspace(1e-6,100,40)})
    }
    return models

def performRegressions(df: pd.DataFrame):
    models = assembleModels()
    y = df['Returned']
    y.replace({'Yes': 1, 'No': 0}, inplace=True)
    x = ut.scaleData(df.drop(columns=['Returned']), ['Sales', 'Discount', 'Profit', 'Shipping.Cost'])
    trainTestData = train_test_split(x, y, test_size=0.3, random_state=0, stratify=y)

    i=0
    for name, model in models.items():
        ut.getExecutionTime(lambda: model.plotHyperParams(*trainTestData, i))
        i+=1

    #models['SVM'].time, returnValue = ut.getExecutionTime(lambda: models['SVM'].fitCV(*trainTestData))
    models['Log'].time, returnValue = ut.getExecutionTime(lambda: models['Log'].fitCV(*trainTestData))

    results = pd.DataFrame([r.__dict__ for r in models.values()]).drop(columns=['model', 'modelCV', 'hyperparams'] )

    roundColumns4Digits = ['trainScore', 'testScore']
    #roundColumns8Digits = ['trainRMSE', 'testRMSE']
    for c in roundColumns4Digits:
        results[c] = results[c].apply(ut.roundTraditional, args = (4,) )

    matrix = confusion_matrix(y,models['Log'].model.predict(x))
    print(matrix)
    results.to_excel('Model Results.xlsx')
    print('Finished Regressions')
    return models







### Problem 6: Evaluating Models
# - What is the best metric to evaluate your model. Is accuracy good for this case?
#Since this is a classification problem, the gini coefficient is ideal

# - Now you have multiple models, which one would you pick?
# I pick Logistic Regression

# - Can you get any clue from the confusion matrix? What is the meaning of precision and recall in this case? Which one do you care the most?
# 49070 0
# 2217  3

#precision = 49070 / (49070 + 2217) = 49070 / 51287 = 0.96
#recall = 49070 / (49070 + 0) = 1

# the confusion matrix shows that the model is NOT good at predicting which orders will be returned.
# It predicts that most orders will not be returned, which is correct simply because few orders are returned.
# The model is lacking the underlying trend that explains why orders are returned.

# How will your model help the manager make decisions?
# Clearly the model is lacking the information needed to predict order returns.


### Problem 7: Feature Engineering Revisit
# - Is there anything wrong with the new feature we generated? How should we fix it?
# order process time will not be known when the predictions are made. This is an example of data leakage
# the issue can be fixed by creating a model that will predict order time.
