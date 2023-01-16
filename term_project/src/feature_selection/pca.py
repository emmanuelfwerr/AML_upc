import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold, cross_val_score

import seaborn as sns
import matplotlib.pyplot as plt


def get_clf_name(classifier_object):
    name_ = str(type(classifier_object)).split('.')[-1]
    for char in """ "'>() """:  # triple-quoted string definition allows us to include " and ' as characters
        name_ = name_.replace(char,"")  # delete unwanted characters from the name
    
    return name_


def get_optimal_n_components(clf, X_data_pca, y_data, cv_splits=10, plot_scores=False):
    """
    Trains a classifier on input principal component data, iteratively dropping the least significant 
    component to determine the number of components yielding the highest cross-validated scores (accuracy 
    and f1-score).
    
    Assumes classifier parameters are already initialized. 
    
    : param clf : Classifier object, assumed to have a scikit-learn wrapper. 
    
    : param X_data_pca : Array of training data in principal component space, arranged from most significant
                         to least significant principal component.
                         
    : param y_data : Vector of class labels corresponding to the training data. 
                         
    : param cv_splits : Integer number of cross-validation splits. 
    
    : param random_state : Seed for reproducability. 
    
    : param plot_scores : Boolean, if True the accuracy and f1-scores will be plotted as a function of the 
                          number of kept principal components. 
                    
    : return results_dict: Python dictionary containing the best-achieved scores and the corresponding
                           number of kept principal components. 
    
    """
    num_dropped_components = np.arange(1, X_data_pca.shape[1])

    mean_accuracies = []
    mean_f1scores = []
    
    # compute cross-validated accuracies and f1-scores after dropping n components
    kfold = StratifiedKFold(n_splits=cv_splits)
    for n_dropped in num_dropped_components:
        
        accuracy_scores = cross_val_score(clf, X_data_pca[:, 0:-n_dropped], y_data, 
                                          cv=kfold, scoring='accuracy')
        
        f1_scores = cross_val_score(clf, X_data_pca[:, 0:-n_dropped], y_data, 
                                    cv=kfold, scoring='f1_macro')
        
        mean_accuracies.append(accuracy_scores.mean())
        mean_f1scores.append(f1_scores.mean())
    
    # obtain and return the best results
    index_best_accuracy = np.argmax(mean_accuracies)
    index_best_f1score = np.argmax(mean_f1scores)
    
    results_dict = {'best_accuracy' : mean_accuracies[index_best_accuracy],
                    'best_accuracy_n' : X_data_pca.shape[1] - num_dropped_components[index_best_accuracy],
                    'best_f1score' : mean_f1scores[index_best_f1score],
                    'best_f1score_n' : X_data_pca.shape[1] - num_dropped_components[index_best_f1score]
                   }
    
    # plot the scores if requested
    if plot_scores is not False:
        plt.figure(figsize=(14,5))
        plt.plot(X_data_pca.shape[1] - num_dropped_components, mean_accuracies, label='Accuracy')
        plt.plot(X_data_pca.shape[1] - num_dropped_components, mean_f1scores, label='F1-Score')
        plt.xticks(X_data_pca.shape[1] - num_dropped_components)
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Score')
        plt.legend(loc='best')
        plt.title(get_clf_name(clf))
        plt.show()
        
    return results_dict


def compare_optimal_n_components(classifiers, X_data_pca, y_data, clf_params=None, 
                                 cv_splits=10, verbose=0, plot_scores=False):
    """
    Obtains and tabulates the optimal number of principal components and corresponding best cross-validated
    test scores for a list of classifier objects. 
        
    : param classifiers : List of classifier objects, assumed to have scikit-learn wrappers. 
    
    : param X_data_pca : Array of training data in principal component space, arranged from most significant
                         to least significant principal component.
                         
    : param y_data : Vector of class labels corresponding to the training data. 
    
    : param clf_params : Optional list of dictionaries stating the classifier parameters to be used. If not 
                         provided, then we use the parameters already initialized for the classifier objects. 
                         
    : param cv_splits : Integer number of cross-validation splits. 
    
    : param random_state : Seed for reproducability. 
    
    : param plot_scores : Boolean, if True the accuracy and f1-scores will be plotted as a function of the 
                          number of kept principal components. 
                          
    : return comparison : Pandas dataframe tabulating the best scores and optimal number of principal 
                          components for each classifier object in 'classifiers'. 
    
    """
    # initialization
    clf_names = []
    best_accuracy = []
    best_accuracy_n_components = []
    best_f1score = []
    best_f1score_n_components = []
    
    if clf_params is None:  # construct using classifier's existing parameter assignment
        clf_params = []
        for clf in classifiers:
            params = clf.get_params() 
            clf_params.append(params)
    
    # step through the classifiers
    for clf, params in zip(classifiers, clf_params):
        
        clf.set_params(**params)
        
        classifier_name = get_clf_name(clf)
        
        clf_names.append(classifier_name)
                
        if verbose == 1:  # print status
            print('\nFinding Optimal Number of Principal Components for Classifier %s of %s:' 
                  % (len(clf_names), len(classifiers)))
            print(classifier_name)
        
        # find optimal number of principal components using cross-validated scoring,
        # and return their scores (both accuracy and f1-score)
        results = get_optimal_n_components(clf, X_data_pca, y_data, 
                                           cv_splits=10,
                                           plot_scores=plot_scores)
        
        best_accuracy.append(results['best_accuracy'])
        best_accuracy_n_components.append(results['best_accuracy_n'])
        best_f1score.append(results['best_f1score'])
        best_f1score_n_components.append(results['best_f1score_n'])
        
        
    # create dataframe for comparing results
    comparison = pd.DataFrame(columns=['Classifier', 
                                       'Best Accuracy', 
                                       'Num Components (Best Accuracy)', 
                                       'Best F1-Score', 
                                       'Num Components (Best F1-Score)'])
    
    comparison['Classifier'] = clf_names
    comparison['Best Accuracy'] = best_accuracy
    comparison['Num Components (Best Accuracy)'] = best_accuracy_n_components
    comparison['Best F1-Score'] = best_f1score
    comparison['Num Components (Best F1-Score)'] = best_f1score_n_components
    
    
    return comparison