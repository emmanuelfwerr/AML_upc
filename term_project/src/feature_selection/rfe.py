import pandas as pd

from sklearn.feature_selection import RFE


def get_RFE_rankings(classifiers, X_data, y_data, verbose=0):
    """
    Performs recursive feature elimination and returns feature rankings for each classifier. 
    
    : param classifiers : List of classifiers objects; assumed to have scikit-learn wrappers. 
    
    : param X_data : Pandas dataframe containing our full feature training set. 
    
    : param y_data : Pandas dataframe containing our training class labels. 
    
    : param verbose: Int, controls amount of status-based text displayed; 1 for more, 0 for less.  
    
    : return feature_rankings : Pandas dataframe tabulating ranked feature importance for each classifier. 
    
    """
    feature_rankings = pd.DataFrame()
    feature_rankings['Feature Name'] = X_data.columns
    
    for clf in classifiers:
        
        # get classifier name
        name = str(type(clf)).split('.')[-1]
        for char in """ "'>() """:  # triple-quoted string definition allows us to include " and ' as characters
            name = name.replace(char,"")  # delete unwanted characters from the name
        
        if name == 'SVC': # SVC does not expose "coef_" or "feature_importances_" attributes, so skip it
            print('Skipped RFE for SVC')
            continue
            
        if name == 'AdaBoostClassifier':  # this classifier causes an error to be thrown with RFE
            print('Skipped RFE for AdaBoostClassifier')
            continue
        
        # print status if requested
        if verbose == 1:
            print('Now performing RFE for', name)
        
        # get freature ranking
        rfe = RFE(estimator=clf, n_features_to_select=1, step=1)
        rfe.fit(X_data, y_data)
        
        # save as new column in dataframe
        feature_rankings[name] = rfe.ranking_
      
    # now sum up totals to obtain an overall ranking 
    # (each classifier's feature ranking will be equally weighted)
    summed_rankings = feature_rankings.sum(axis=1)
    
    # here we turn the sum into a rank (lower rank goes with lower sum)
    sorted_rankings = [0] * len(summed_rankings)
    for i, x in enumerate(sorted(range(len(summed_rankings)), key=lambda y: summed_rankings[y])):
        sorted_rankings[x] = i + 1  # offset so that lowest rank is 1
    
    feature_rankings['Overall Ranking'] = sorted_rankings
    
    # re-order dataframe so that 'Overall Ranking' is the 2nd column
    cols = feature_rankings.columns.tolist()
    cols = [cols[0]] + [cols[-1]] + cols[1:-1]
    feature_rankings = feature_rankings[cols] 
    
    # sort the dataframe rows in terms of acending rank
    feature_rankings = feature_rankings.sort_values(by='Overall Ranking')
    
    return feature_rankings