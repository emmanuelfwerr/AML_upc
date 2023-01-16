# General Tools:
import pandas as pd
import numpy as np

# Machine Learning Classifiers:
from sklearn.ensemble import AdaBoostClassifier

# Feature and Model Selection:
from sklearn.model_selection import StratifiedKFold 
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc  # scoring metrics

# visualization libraries
import seaborn as sns
import matplotlib.pyplot as plt



def train_classifier_ensemble_CV(classifiers, X_data, y_data, clf_params=None, cv_splits=10, 
                                 random_state=42, return_trained_classifiers=False, verbose=0):

    # initialization
    kfold = StratifiedKFold(n_splits=cv_splits, random_state=42,shuffle=True)
    
    train_accuracy_mean = []
    train_accuracy_std = []
    test_accuracy_mean = []
    test_accuracy_std = []
    f1_score_mean = []
    f1_score_std = []
    mean_feature_importances = []
    trained_classifiers = []
    classifier_name = []
    
    if clf_params is None:  # construct using classifier's existing parameter assignment
        clf_params = []
        for clf in classifiers:
            params = clf.get_params() 
            if 'random_state' in params.keys():  # assign random state / seed
                params['random_state'] = random_state
            elif 'seed' in params.keys():
                params['seed'] = random_state
            clf_params.append(params)
    
    # step through the classifiers for training and scoring with cross-validation
    for clf, params in zip(classifiers, clf_params):
        
        # automatically obtain the name of the classifier
        name = get_clf_name(clf)
        classifier_name.append(name)
        
        if verbose == 1:  # print status
            print('\nPerforming Cross-Validation on Classifier %s of %s:' 
                  % (len(classifier_name), len(classifiers)))
            print(name)
        
        # perform k-fold cross validation for this classifier and calculate scores for each split
        kth_train_accuracy = []
        kth_test_accuracy = []
        kth_test_f1_score = []
        kth_feature_importances = []
        
        for (train, test) in kfold.split(X_data, y_data):
        
            clf.set_params(**params)
            clf.fit(X_data.iloc[train], y_data.iloc[train])
            
            kth_train_accuracy.append(clf.score(X_data.iloc[train], y_data.iloc[train]))
            kth_test_accuracy.append(clf.score(X_data.iloc[test], y_data.iloc[test]))
            kth_test_f1_score.append(f1_score(y_true=y_data.iloc[test], y_pred=clf.predict(X_data.iloc[test])))
            
            if hasattr(clf, 'feature_importances_'):  # some classifiers (like linReg) lack this attribute
                kth_feature_importances.append(clf.feature_importances_)
        
        # populate scoring statistics for this classifier (over all cross-validation splits)
        train_accuracy_mean.append(np.mean(kth_train_accuracy))
        train_accuracy_std.append(np.std(kth_train_accuracy))
        test_accuracy_mean.append(np.mean(kth_test_accuracy))
        test_accuracy_std.append(np.std(kth_test_accuracy))
        f1_score_mean.append(np.mean(kth_test_f1_score))
        f1_score_std.append(np.std(kth_test_f1_score))
    
        # obtain array of mean feature importances, if this classifier had that attribute
        if len(kth_feature_importances) == 0:
            mean_feature_importances.append(False)
        else:
            mean_feature_importances.append(np.mean(kth_feature_importances, axis=0))
        
        # if requested, also export classifier after fitting on the complete training set 
        if return_trained_classifiers is not False:
            clf.fit(X_data, y_data)
            trained_classifiers.append(clf)
            
        # remove AdaBoost feature importances (we won't discuss their interpretation)
        if type(clf) == type(AdaBoostClassifier()):
            mean_feature_importances[-1] = False
        
    
    # construct dataframe for comparison of classifiers
    clf_comparison = pd.DataFrame({'Classifier Name' : classifier_name, 
                                   'Mean Train Accuracy' : train_accuracy_mean, 
                                   'Train Accuracy Standard Deviation' : train_accuracy_std,
                                   'Mean Test Accuracy' : test_accuracy_mean, 
                                   'Test Accuracy Standard Deviation' : test_accuracy_std, 
                                   'Mean Test F1-Score' : f1_score_mean,
                                   'F1-Score Standard Deviation' : f1_score_std})
    
    # enforce the desired column order
    clf_comparison = clf_comparison[['Classifier Name', 'Mean Train Accuracy',
                                     'Train Accuracy Standard Deviation', 'Mean Test Accuracy',
                                     'Test Accuracy Standard Deviation', 'Mean Test F1-Score',
                                     'F1-Score Standard Deviation']]
                   
    
    # add return_trained_classifiers to the function return, if requested, otherwise omit                                                
    if return_trained_classifiers is not False:
        return clf_comparison, mean_feature_importances, trained_classifiers
    else:
        return clf_comparison, mean_feature_importances
        

def get_clf_name(classifier_object):
    name_ = str(type(classifier_object)).split('.')[-1]
    for char in """ "'>() """:  # triple-quoted string definition allows us to include " and ' as characters
        name_ = name_.replace(char,"")  # delete unwanted characters from the name
    
    return name_


def plot_mean_feature_importances(clf_comparison, mean_feature_importances, X_data):
    """
    Generates bar plots of feature importances using the results of train_classifier_ensemble_CV.
    
    : param clf_comparison : A pandas dataframe comparing cross-validated classifier performances; 
                             one of the return parameters of the train_classifier_ensemble_CV function. 
    
    : param mean_feature_importances : An array of feature importances, generated by the 
                                       train_classifier_ensemble_CV function. 
    
    : param X_data : A pandas dataframe of the feature data used in the creation of clf_comparison and 
                     mean_feature_importances.
                     
    : return : None. 

    """
    for clf_name, importances in zip(clf_comparison['Classifier Name'], mean_feature_importances):

        if importances is not False:

            indices = np.argsort(importances)[::-1]
            feature_labels = X_data.columns

            plt.figure(figsize=(12,5))
            plt.title('Feature Importances for ' + clf_name)
            plt.bar(range(X_data.shape[1]), importances[indices], color='lightblue', align='center')
            plt.xticks(range(X_data.shape[1]), feature_labels[indices], rotation=90)
            plt.xlim([-1, X_data.shape[1]])
            plt.tight_layout()
            plt.show()
        
    return


def plot_top_feature_importances(clf_comparison, mean_feature_importances, X_data):
    '''  '''
    best_features = []
    for clf_name, importances in zip(clf_comparison['Classifier Name'], mean_feature_importances):

        if importances is not False:

            indices = np.argsort(importances)[::-1]
            feature_labels = X_data.columns

            for col in feature_labels[indices][:20]:
                best_features.append(col)

            plt.figure(figsize=(12,5))
            plt.title('Feature Importances for ' + clf_name)
            plt.bar(range(20), importances[indices][:20], color='lightblue', align='center')
            plt.xticks(range(20), feature_labels[indices][:20], rotation=90)
            plt.xlim([-1, 21])
            plt.tight_layout()
            plt.show()
        
    return set(best_features)