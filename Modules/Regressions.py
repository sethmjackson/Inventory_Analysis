### Problem 5: Fitting Models
from sklearn.ensemble import GradientBoostingRegressor

from Modules.Main import *
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression, Ridge
from typing import List, Dict, Callable


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

        self.trainRMSE = self.getRMSE(yTrain, self.modelCV.predict(xTrain))
        self.testRMSE = self.getRMSE(yTest, self.modelCV.predict(xTest))




    def plotHyperParams(self, trainX, testX, trainY, testY, i):


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
    'Ridge'      :  Regression(Ridge(), 'Ridge', {'alpha': np.linspace(1,100,100)}),

    'Gradient Boost': Regression(GradientBoostingRegressor(), 'Gradient Boost',
               {'learning_rate': np.linspace(.001, 0.1, 10),
                'n_estimators': range(60, 80, 5),
                'max_depth': range(1, 6),
                'loss': ['ls']}), # use feature_importances for feature selection
    }
    return models

def performRegressions(df: pd.DataFrame):
    models = assembleModels()
    y = df['LogSalePrice']





    continuousColumns.remove('LogSalePrice')
    x = scaleData(df.drop(columns=['LogSalePrice']), continuousColumns)
    #x = df.drop(columns=['LogSalePrice'])
    trainTestData = train_test_split(x, y, test_size=0.3, random_state=0)

    #models['Ridge'].plotHyperParams(*trainTestData, 1)
    # models['Lasso'].plotHyperParams(*trainTestData,2)
    # models['Elastic Net'].plotHyperParams(*trainTestData)

    # models['Ridge'].plotHyperParams(*trainTestData)
    # models['SVM'].plotHyperParams(*trainTestData)
    i=0
    for name, model in models.items():
        model.plotHyperParams(*trainTestData,i)
        i+=1

    models['Ridge'].time,          returnValue = ut.getExecutionTime(lambda: models['Ridge'].fitCV(*trainTestData))
    models['Gradient Boost'].time, returnValue = ut.getExecutionTime(lambda: models['Gradient Boost'].fitCV(*trainTestData))

    results = pd.DataFrame([r.__dict__ for r in models.values()]).drop(columns=['model', 'modelCV'] )

    roundColumns4Digits = ['trainScore', 'testScore']
    #roundColumns8Digits = ['trainRMSE', 'testRMSE']
    for c in roundColumns4Digits:
        results[c] = results[c].apply(ut.roundTraditional, args = (4,) )

    results.to_excel('Output/Model Results.xlsx')
    print('Finished Regressions')
    return models







### Problem 6: Evaluating Models
# - What is the best metric to evaluate your model. Is accuracy good for this case?
# - Now you have multiple models, which one would you pick?
# - Can you get any clue from the confusion matrix? What is the meaning of precision and recall in this case? Which one do you care the most?
# How will your model help the manager make decisions?
# - **Note:** The last question is open-ended. Your answer could be completely different depending on your understanding of this business problem.

### Problem 7: Feature Engineering Revisit
# - Is there anything wrong with the new feature we generated? How should we fix it?
# - ***Hint***: For the real test set, we do not know it will get returned or not.
