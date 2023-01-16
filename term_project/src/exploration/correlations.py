import itertools



def get_mostProbable(dataframe, feature_list, feature_values):
    
    high_val, low_val = 0, 1
    high_set, low_set = [], []
    high_size, low_size = [], []
    
    for combo in itertools.product(*feature_values):
    
        subset = dataframe[dataframe[feature_list[0]] == combo[0]]
        for i in range(len(feature_list))[1:]:
            subset = subset[subset[feature_list[i]] == combo[i]]
        mean_survived = subset['Survived'].mean()
    
        if mean_survived > high_val:
            high_set = combo
            high_val = mean_survived
            high_size = subset.shape[0]
        
        if mean_survived < low_val:
            low_set = combo
            low_val = mean_survived
            low_size = subset.shape[0]
        
    print('\n*** Most likely to survive ***')
    for i in range(len(feature_list)):
        print('%s : %s' % (feature_list[i], high_set[i]))
    print('... with survival probability %.2f' % high_val)
    print('and total set size %s' % high_size)
        
    print('\n*** Most likely to perish ***')
    for i in range(len(feature_list)):
        print('%s : %s' % (feature_list[i], low_set[i]))
    print('... with survival probability %.2f' % low_val)
    print('and total set size %s' % low_size)