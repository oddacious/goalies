
# coding: utf-8

# In[13]:

import collections
import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from IPython.display import display

from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble.partial_dependence import plot_partial_dependence
from sklearn.grid_search import GridSearchCV

get_ipython().magic(u'matplotlib inline')


# In[15]:

def build_model(X_train, y_train, cv=5):
    """Perform a GradientBoostingRegressor grid search and return the winner."""
    
    # Note to self: Use subsample for stochastic gradiant boosting, next time
    param_grid = {'n_estimators': [50, 100, 200],
                  'max_depth': [2, 3, 4],
                  'min_samples_split': [100, 200, 300, 500],
                  'min_samples_leaf': [5, 10, 20, 50],
                  'learning_rate': [0.01, 0.1],
                  'loss': ['huber']}

    # for testing
    param_grid2 = {'n_estimators': [50],
                  'max_depth': [2],
                  'min_samples_split': [100],
                  'min_samples_leaf': [5],
                  'learning_rate': [0.1],
                  'loss': ['huber']}
    grid_search = GridSearchCV(GradientBoostingRegressor(),
                               param_grid=param_grid,
                               cv=cv).fit(X_train, y_train)

    return grid_search.best_estimator_

def plot_improvement(model, test_score, best_tree):
    """Plot the improvement from additional trees."""
    plt.figure(figsize=(14, 7))
    plt.subplot(1, 2, 1)
    plt.title('Deviance')
    plt.plot(np.arange(model.n_estimators) + 1, model.train_score_, 'b-',
             label='Training Set Deviance')
    plt.plot(np.arange(model.n_estimators) + 1, test_score, 'r-',
             label='Test Set Deviance')
    plt.legend(loc='upper right')
    plt.xlabel('Boosting Iterations')
    plt.ylabel('Deviance')
    print "Test performance best at {} trees".format(best_tree)
    # No plt.show() here because this is meant to be followed by plot_importance()

def plot_importance(model, feature_names):
    """Plot the relative importance of the model features."""
    feature_importance = model.feature_importances_
    feature_importance = 100.0 * (feature_importance / feature_importance.max())
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5
    plt.subplot(1, 2, 2)
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, [feature_names[i] for i in sorted_idx])
    plt.xlabel('Relative Importance')
    plt.title('Variable Importance')
    plt.show()

def plot_features(model, feature_names, target, x):
    """Plot the partial dependence of the feature set."""
    plt.figure(figsize=(20, 10))
    fig, _ = plot_partial_dependence(model,
                                    x,
                                    range(len(feature_names)),
                                    feature_names=feature_names,
                                    n_jobs=-1,
                                    n_cols=4,
                                    grid_resolution=50)
    fig.suptitle('Partial dependence of features predicting'.format(target))
    plt.subplots_adjust(top=0.9)  # tight_layout causes overlap with suptitle
    plt.subplots_adjust(right=1.5)
    print('_' * 80)
    print('Custom plot via ``partial_dependence``')
    print
    #fig_size = plt.rcParams["figure.figsize"]
    plt.rcParams["figure.figsize"] = [12, 8]
    plt.show()

def get_best_tree_index(model, x, y):
    """Return the index of the best tree."""
    TreeResult = collections.namedtuple('TreeResult', ['min_score', 'best_tree', 'test_score'])

    test_score = np.zeros((model.n_estimators,), dtype=np.float64)

    min_score = None
    best_tree = None
    for i, y_pred in enumerate(model.staged_predict(x)):
        test_score[i] = model.loss_(y, y_pred)
        if min_score is None or test_score[i] < min_score:
            min_score = test_score[i]
            best_tree = i

    return TreeResult(min_score, best_tree, test_score)

def rolling_mean(x, window=10, min_periods=1):
    """Return a rolling mean"""
    return x.rolling(center=False, window=window, min_periods=min_periods).mean()

def rolling_sum(x, window=10, min_periods=1):
    """Return a rolling sum"""
    return x.rolling(center=False, window=window, min_periods=min_periods).sum()

def create_column_permutations(goalies, columns):
    """Create the columns that we'll need as features."""
    for column in columns:
        print "Adding column {}".format(column)

        labels = {'lag': column + '_lag',
                  'lag2': column + '_lag2',
                  'avg': column + '_avg',
                  'avg_weighted': column + '_wt_avg',
                  'league_avg': column + '_lg_avg',
                  'league_delta': column + '_lg_delta',
                  'league_weighted_avg': column + '_lg_wt_avg',
                  'league_weighted_delta': column + '_lg_wt_delta',
                  'total': column + '_total'}

        goalies.loc[:, labels['lag']] = goalies.groupby(['player_name'])[column].shift(1)
        goalies.loc[:, labels['lag2']] = goalies.groupby(['player_name'])[column].shift(2)
        goalies.loc[goalies[labels['lag2']].isnull(), labels['lag2']] = goalies[labels['lag']]
        goalies.loc[:, labels['total']] = (goalies.groupby(['player_name'])['GP'].shift(1) *
                                           goalies.groupby(['player_name'])[column].shift(1))
        goalies.loc[:, labels['avg']] = goalies.groupby('player_name')[labels['lag']].apply(rolling_mean)
        numer = goalies.groupby('player_name')[labels['total']].apply(rolling_sum)
        denom = goalies.groupby('player_name')['GP_lag'].apply(rolling_sum)
        goalies.loc[:, labels['avg_weighted']] = numer / denom

        tmp = goalies.groupby('year')[labels['lag']].mean().to_frame()
        tmp.columns = [labels['league_avg']]
        tmp['year'] = goalies.groupby('year')[labels['lag']].mean().keys()
        goalies = goalies.merge(tmp, how='left', on='year')
        goalies.loc[:, labels['league_delta']] = goalies[labels['lag']] - goalies[labels['league_avg']]

        tmp1 = goalies
        tmp1.loc[:, 'tmp_weight'] = tmp1[labels['lag']]*tmp1['GP_lag']

        tmp = (tmp1.groupby('year')['tmp_weight'].sum()/tmp1.groupby('year')['GP_lag'].sum()).to_frame()
        tmp.columns = [labels['league_weighted_avg']]
        tmp['year'] = goalies.groupby('year')[labels['lag']].mean().keys()
        goalies = goalies.merge(tmp, how='left', on='year')
        goalies.loc[:, labels['league_weighted_delta']] = (goalies[labels['lag']] -
                                                           goalies[labels['league_weighted_avg']])

    return goalies

def create_future_row(goalies, latest_year):
    """Create a false row in the dataset to hold our future values."""
    tmp = goalies[goalies['year'] == latest_year]
    tmp.loc[:, 'age'] += 1
    tmp.loc[:, 'year'] += 1

    return goalies.append(tmp, ignore_index=True)

def define_predictors(stat, extra_predictors):
    """Define the combined predictor set."""
    predictors = list(set(['age',
                           stat + '_lag',
                           stat + '_wt_avg',
                           stat + '_lag2',
                           stat + '_lg_wt_avg',
                           stat + '_lg_wt_delta']))
    if stat in extra_predictors:
        predictors = list(set(predictors + extra_predictors[stat]))

    return predictors

def has_all_predictors(goalies, stat, predictors):
    """Return rows that have all the necessary predictors."""
    has_value = goalies[np.isfinite(goalies[stat])]

    for item in predictors:
        has_value = has_value[np.isfinite(has_value[item])]

    return has_value

def build_data(goalies, min_games=10, min_year=1986):
    """Build out our dataset, creating feature columns and a future dataset."""
    GoalieSplit = collections.namedtuple('GoalieSplit', ['existing', 'future'])

    goalies = goalies[goalies['GP'] >= min_games]

    latest_year = max(goalies['year'])
    goalies = create_future_row(goalies, latest_year).sort_values(by=['year', 'player_name'])

    goalies.loc[:, 'age_sq'] = goalies['age']**2
    goalies.loc[:, 'W_per_GP'] = goalies['W'] / goalies['GP']
    goalies.loc[:, 'SO_per_GP'] = goalies['SO'] / goalies['GP']
    goalies.loc[:, 'SV_per_GP'] = goalies['SV'] / goalies['GP']

    goalies = create_column_permutations(goalies,
                                         ('GP', 'W', 'SO', 'SV%', 'GAA', 'W_per_GP', 'SO_per_GP', 'SV_per_GP'))

    future = goalies[goalies['year'] > latest_year]

    # Note: Save percentage not recorded until the mid-80s
    existing = goalies[np.isfinite(goalies['W_lag']) &
                       np.isfinite(goalies['W']) &
                       (goalies['year'] >= min_year) &
                       (goalies['year'] <= latest_year) &
                       (goalies['GP'] >= min_games)]

    return GoalieSplit(existing, future)

def main(input_file):
    """Main entry point for code."""
    goalie_split = build_data(pd.read_csv(input_file))

    extra_predictors = {'GP': ['W_lag', 'GAA_lag', 'SV%_lag'],
                        'W_per_GP': ['W_lag', 'GP_lag'],
                        'SO_per_GP': ['W_lag', 'GP_lag'],
                        'SV_per_GP': ['W_lag', 'GP_lag', 'GP_avg', 'SV_per_GP_wt_avg'],
                        'GAA': ['GP_lag', 'GP_avg', 'GAA_wt_avg'],
                        'SV%': ['GP_lag', 'GP_avg', 'SV%_wt_avg']}

    for stat in ['GP', 'W_per_GP', 'SO_per_GP', 'GAA', 'SV%', 'SV_per_GP']:
    #for stat in ['SV_per_GP']:
        predictors = define_predictors(stat, extra_predictors)

        print "Predicting {0} with {1}".format(stat, predictors)

        has_value = has_all_predictors(goalie_split.existing, stat, predictors)

        print "Dropped {} rows for missing values".format(len(goalie_split.existing) - len(has_value))

        x = has_value.loc[:, predictors]
        y = has_value.loc[:, stat]
        x_goalies_future = goalie_split.future.loc[:, predictors]

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

        model = build_model(X_train, y_train)

        print "Chosen model = {}".format(model)

        has_value.loc[:, 'p_' + stat] = model.predict(x)
        goalie_split.future.loc[:, 'p_' + stat] = model.predict(x_goalies_future)

        print "RSME = {}".format(math.sqrt(metrics.mean_squared_error(has_value[stat], has_value['p_' + stat])))

        spearman = scipy.stats.spearmanr(has_value[stat], has_value['p_' + stat])

        print "Spearman correlation = {}".format(spearman.correlation)

        tree_result = get_best_tree_index(model, X_test, y_test)

        plot_improvement(model, tree_result.test_score, tree_result.best_tree)
        plot_importance(model, predictors)
        plot_features(model, predictors, stat, X_train)
        
    GoalieSplit = collections.namedtuple('GoalieSplit', ['existing', 'future'])

    return GoalieSplit(goalie_split.existing, goalie_split.future)

results = main("~/code/fun/nhl/goalie_data/stats.csv")


# In[74]:

goalies2[(goalies2['player_name'] == 'Braden Holtby') & (goalies2['year'] == 2016)].loc[:,
    ['GAA_lg_avg', 'age', 'GAA_lag', 'GP_avg', 'GAA_lg_delta', 'GAA_wt_avg', 'GAA_avg', 'GAA_lag2', 'GP_lag', 'GAA_lg_wt_avg']]

#goalies_future.loc[:, ['player_name', 'year', 'GP', 'p_GP', 'p_GAA']].sort_values(by=['GP'], ascending=False)


# In[26]:


results.future.loc[:, 'p_W'] = results.future['p_GP'] * results.future['p_W_per_GP']
results.future.loc[:, 'p_SO'] = results.future['p_GP'] * results.future['p_SO_per_GP']
results.future.loc[:, 'p_SV'] = results.future['p_GP'] * results.future['p_SV_per_GP']

results.future.loc[:, ['player_name', 'year', 'p_GP', 'p_W', 'p_SO', 'p_GAA', 'p_SV%', 'p_SV', 'GP', 'W', 'SO', 'GAA', 'p_SV_per_GP']].sort_values(by=['p_SV'],
                                                                                                                                   ascending=False)
pd.set_option('display.height', 500)
pd.set_option('display.max_rows', 500)

results.future.loc[:, ['player_name', 'p_GP', 'p_SV', 'p_SV_per_GP']].sort_values(by=['p_SV'], ascending=False)


# In[9]:

pred = goalies_future.loc[:, ['player_name', 'year', 'p_GP', 'p_W', 'p_SO', 'p_GAA', 'p_SV%']].sort_values(by=['p_GP'],ascending=False)

pred.to_csv('goalies_basic_projections_201617_as_of_20160917.csv', float_format="%.3f", index=False)


# In[86]:

goalies_future.loc[:, ['player_name', 'year', 'p_GP', 'p_W', 'p_SO', 'p_GAA', 'p_SV%', 'GP', 'W', 'SO', 'GAA', 'SV%']].describe()


# In[ ]:



