import math
import seaborn as sns
import matplotlib.pyplot as plt


def barplots(dataframe, features, cols=2, width=10, height=10, hspace=0.5, wspace=0.25):
    '''  '''
    # define style and layout
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataframe.shape[1]) / cols)
    # define subplots
    for i, column in enumerate(dataframe[features].columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        sns.barplot(column,'Survived', data=dataframe)
        plt.xticks(rotation=0)
        plt.xlabel(column, weight='bold')


def histograms(dataframe, features, force_bins=False, cols=2, width=10, height=10, hspace=0.2, wspace=0.25):
    '''  '''
    # define style and layout
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataframe.shape[1]) / cols)
    # define subplots
    for i, column in enumerate(dataframe[features].columns):
        ax = fig.add_subplot(rows, cols, i + 1)
        df_survived = dataframe[dataframe['Survived'] == 1]
        df_perished = dataframe[dataframe['Survived'] == 0]
        if force_bins is False:
            sns.distplot(df_survived[column].dropna().values, kde=False, color='blue')
            sns.distplot(df_perished[column].dropna().values, kde=False, color='red')
        else:
            sns.distplot(df_survived[column].dropna().values, bins=force_bins[i], kde=False, color='blue')
            sns.distplot(df_perished[column].dropna().values, bins=force_bins[i], kde=False, color='red')
        plt.xticks(rotation=25)
        plt.xlabel(column, weight='bold') 


def univariate_kdeplots(dataframe, plot_features, cols=2, width=10, height=10, hspace=0.2, wspace=0.25):
    '''  '''
    # define style and layout
    sns.set(font_scale=1.5)
    plt.style.use('seaborn-whitegrid')
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=wspace, hspace=hspace)
    rows = math.ceil(float(dataframe.shape[1]) / cols)
    # define subplots
    for i, feature in enumerate(plot_features):
        ax = fig.add_subplot(rows, cols, i + 1)
        g = sns.kdeplot(dataframe[plot_features[i]][(dataframe['Survived'] == 0)].dropna(), shade=True, color="red")
        g = sns.kdeplot(dataframe[plot_features[i]][(dataframe['Survived'] == 1)].dropna(), shade=True, color="blue")
        g.set(xlim=(0 , dataframe[plot_features[i]].max()))
        g.legend(['Perished', 'Survived'])
        plt.xticks(rotation=25)
        ax.set_xlabel(plot_features[i], weight='bold')


