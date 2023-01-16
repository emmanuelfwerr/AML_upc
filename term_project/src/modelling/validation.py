import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline



def get_validation_curve(classifier, X_data, y_data, param_feed, cv_splits=10, 
                         figure_size=(7, 5), x_scale='linear', y_lim=[0.70, 1.0]):
    """
    Generates a validation curve over a specified range of hyperparameter values for a given classifier,
    and prints the optimal values yielding i) the highest cross-validated mean accuracy and ii) the smallest
    absolute difference between the mean test and train accuracies (to assess overfitting). 
    
    : param classifier : Classifier object, assumed to have a scikit-learn wrapper. 
    
    : param X_data : Pandas dataframe containing the training feature data. 
    
    : param y_data : Pandas dataframe containing the training class labels. 
    
    : param param_feed : Dictionary of form {'parameter_name' : parameter_values}, where parameter_values
                         is a list or numpy 1D-array of parameter values to sweep over. 
               
    : param cv_splits : Integer number of cross-validation splits. 
    
    : param figure_size : Tuple of form (width, height) specifying figure size. 
    
    : param x_scale : String, 'linear' or 'log', controls x-axis scale type of plot. 
    
    : param y_lim : List of form [y_min, y_max] for setting the plot y-axis limits. 
    
    : return : None. 
    
    """
    base_param_name = list(param_feed.keys())[0]
    param_range_ = param_feed[base_param_name]
    
    piped_clf = Pipeline([('clf', classifier)]) # I use this merely to assign the handle 'clf' to our classifier

    # Obtain the cross-validated scores as a function of hyperparameter value
    train_scores, test_scores = validation_curve(estimator=piped_clf,
                                                 X=X_train[feature_subset_EDA],
                                                 y=y_train,
                                                 param_name='clf__' + base_param_name,
                                                 param_range=param_range_,
                                                 cv=cv_splits)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    # Generate the validation curve plot
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=figure_size)

    plt.plot(param_range_, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(param_range_, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(param_range_, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
    plt.fill_between(param_range_, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

    plt.xscale(x_scale)
    plt.xlabel(base_param_name)
    plt.ylim(y_lim)
    plt.ylabel('Accuracy')
    plt.grid(b=True, which='major', color='black', linestyle=':')
    plt.legend(loc='lower right')
    plt.title('Validation Curve for Parameter ' + base_param_name)
    plt.show()

    # Display optimal parameter values for best accuracy and smallest train-test difference
    diffs = abs(train_mean - test_mean)
    id_best_diff = np.argmin(diffs)
    id_best_acc = np.argmax(test_mean)
    
    print('Best Accuracy is %.5f occuring at %s = %s' % (test_mean[id_best_acc],
                                                         base_param_name,
                                                         param_range_[id_best_acc]))
    
    
    print('Smallest Train-Test Difference is %.5f occuring at %s = %s' % (diffs[id_best_diff],
                                                                          base_param_name,
                                                                          param_range_[id_best_diff]))
    
    return


def get_learning_curve(classifier, X_data, y_data, training_sizes=np.linspace(0.1, 1.0, 10), cv_splits=10,
                       figure_size=(7, 5), y_lim=[0.70, 1.0]):
    """
    Generates a learning curve to asses bias-variance tradeoff by plotting cross-validated train and test
    accuracies as a function of the number of samples used for training. 
    
    : param classifier : Classifier object, assumed to have a scikit-learn wrapper. 
    
    : param X_data : Pandas dataframe containing the training feature data. 
    
    : param y_data : Pandas dataframe containing the training class labels. 
    
    : param training_sizes : Numpy 1D array of the training sizes to sweep over, specified as fractions
                             of the total training set size. 
               
    : param cv_splits : Integer number of cross-validation splits. 
    
    : param figure_size : Tuple of form (width, height) specifying figure size. 
        
    : param y_lim : List of form [y_min, y_max] for setting the plot y-axis limits. 
    
    : return : None. 
    
    """
    # Obtain the cross-validated scores as a function of the training size
    train_sizes, train_scores, test_scores = learning_curve(estimator=classifier,
                                                            X=X_data,
                                                            y=y_data,
                                                            train_sizes=training_sizes,
                                                            cv=cv_splits,
                                                            n_jobs=-1)
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Generate the learning curve plot
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    plt.figure(figsize=figure_size)
    
    plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='Training Accuracy')
    plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
    plt.plot(train_sizes, test_mean, color='red', linestyle='--', marker='s', markersize=5, label='Validation Accuracy')
    plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='red')
    
    plt.xlabel('Number of Training Samples')
    plt.ylim(y_lim)
    plt.ylabel('Accuracy')
    plt.grid(b=True, which='major', color='black', linestyle=':')
    plt.legend(loc=4)
    plt.title('Learning Curve')
    plt.show()
    
    return
    