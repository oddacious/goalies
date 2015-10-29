import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import a series of files from WOI. Here I just use their download functionality to get CSV files
def import_WOI_datasets(files):
    results = None
    for file in files:
        data = pd.read_csv(file)
        data['source'] = pd.Series([file] * len(data.index), index=data.index)
        if results is None:
            results = data
        else:
            results = pd.concat([results, data])
    return results

# Join in the league-wide number of goals and saves
def generate_aggregate_shots(stat_table):
    chunk_shots = stat_table.groupby(['season', 'source'], as_index=False)
    agg_stats = chunk_shots[['G.L', 'S.L', 'G.M', 'S.M', 'G.H', 'S.H']].sum()
    stats = pd.merge(stat_table, agg_stats, on=['season', 'source'], how='left', suffixes=('','_all'))
    stats['Sh.L_all'] = stats['G.L_all'] + stats['S.L_all']
    stats['Sh.M_all'] = stats['G.M_all'] + stats['S.M_all']
    stats['Sh.H_all'] = stats['G.H_all'] + stats['S.H_all']
    
    stats['adj'] = (stats['Sv%L'] * stats['Sh.L_all'] + stats['Sv%M'] * stats['Sh.M_all'] + 
                       stats['Sv%H'] * stats['Sh.H_all']) / (stats['Sh.L_all'] + stats['Sh.M_all'] + stats['Sh.H_all'])
    return stats

# Merge values across situation types into one data frame
def flatten_situations(stat_table, param):
    for source in stat_table.source.unique():
        stat_table = pd.merge(stat_table, stat_table.loc[stat_table['source'] == source, ['season', 'Name', param]], 
                    on=['season', 'Name'], how='left', suffixes=('','_' + source))
        
    return stat_table

# Include aggregate shot data and calculate adjusted save percentage
def flatten_shots(stat_table):
    chunk_shots = stat_table.groupby(['season', 'source'], as_index=False)
    agg_shots = chunk_shots[['Sh']].sum()
    num_goalies = pd.DataFrame({'count' : stat_table.groupby(['season', 'source'], as_index=False).size()}).reset_index()

    for source in stat_table.source.unique():
        stat_table = pd.merge(stat_table, agg_shots.loc[agg_shots['source'] == source, ['season', 'Sh', 'G']], 
                on=['season'], how='left', suffixes=('','_agg_' + source))
        stat_table = pd.merge(stat_table, num_goalies.loc[num_goalies['source'] == source, ['season', 'count']], 
                on=['season'], how='left', suffixes=('', '_' + source))

        # Rename
        if 'count_' + source not in stat_table.columns:
            stat_table.rename(columns={'count': 'count_' + source}, inplace=True)   
 
    return stat_table

# Perform the adjustment across multiple categories
def calculate_joint_adsv(stat_table):
    numer = [len(stat_table)*0]
    denom = [len(stat_table)*0]
    
    for source in stat_table.source.unique():
        numer = numer + stat_table['adj_' + source]*stat_table['Sh_agg_' + source]
        denom = denom + stat_table['Sh_agg_' + source]   

    stat_table['adj_agg'] = numer/denom

    return stat_table
    
# Please see http://jackman.stanford.edu/classes/BASS/ch2.pdf or 
# http://web.as.uky.edu/statistics/users/pbreheny/701/S13/notes/1-15.pdf
# Note: Equation often used with -1 and -2, as the mode of the posterior density
def calculate_priors(stat_table, weight):
    for source in stat_table.source.unique():
        for situ in ['L', 'M', 'H']:
            alpha = weight * (stat_table['G.' + situ + '_all'] + stat_table['S.' + situ + '_all'])/stat_table['count_' + source]
            beta = weight * stat_table['G.' + situ + '_all']/stat_table['count_' + source]
            stat_table['bayes_Sv%' + situ] = (alpha + stat_table['S.' + situ])/(alpha + beta + stat_table['G.' + situ] + 
            
            stat_table['numer'] = (alpha + stat_table['S.' + situ])
            stat_table['denom'] = (alpha + beta + stat_table['G.' + situ] + stat_table['S.' + situ]);

        stat_table['adj_bayes'] = 100*(stat_table['bayes_Sv%L'] * stat_table['Sh.L_all'] + 
                                   stat_table['bayes_Sv%M'] * stat_table['Sh.M_all'] + 
                                   stat_table['bayes_Sv%H'] * stat_table['Sh.H_all']) / (stat_table['Sh.L_all'] + 
                                                                stat_table['Sh.M_all'] + stat_table['Sh.H_all'])
        
    table2 = flatten_situations(stat_table, 'adj_bayes')

    numer = [len(table2) * 0]
    denom = [len(table2) * 0]
    
    for source in table2.source.unique():
        numer = numer + table2['adj_bayes_' + source]*table2['Sh_agg_' + source]
        denom = denom + table2['Sh_agg_' + source]   
        
    table2['adj_bayes'] = numer/denom
    
    return table2

# Build up the aggregate statistics
def generate_statistics(raw, weight=1):
    stat_table = generate_aggregate_shots(raw)
    stat_table2 = flatten_situations(stat_table, 'adj')
    stat_table3 = calculate_joint_adsv(flatten_shots(stat_table2))
    return calculate_priors(stat_table3, weight)

# Import our data
# For this analysis I've restricted it to goalies with 30+ minutes played.

results_all = import_WOI_datasets(['5v5.csv', '4v4.csv', 'pp.csv', 'short.csv', 'other_pulled.csv', 'leftovers.csv'])
results_most = import_WOI_datasets(['5v5.csv', '4v4.csv', 'pp.csv', 'short.csv'])
results_with_sh = import_WOI_datasets(['5v5.csv', 'short.csv'])

# Show aggregate situation type over time

chunk = results_all.groupby(['season', 'source'])
shot_plot = chunk.agg({'Sh': np.sum}).unstack().plot(kind='bar', stacked=True, title='Shots by type', figsize=(9, 7))
shot_plot.set_xlabel('Season')
shot_plot.set_ylabel('Shots')
plt.gcf().subplots_adjust(bottom=0.25)
plt.show()

# Check that my calculation of AdSv% is quite close to the War on Ice one. Mine may differ because I only pulled goalies
# with 30+ minutes played

stat_table_sh = generate_statistics(results_with_sh)

print(stat_table_sh[['AdSv%', 'adj']].corr(method='pearson'))
print(stat_table_sh[['AdSv%', 'adj']].corr(method='spearman'))

# Look at save percentage by situation-location

chunk = results_all.groupby(['source'], as_index=False)
tmp = chunk[['G.L', 'S.L', 'G.M', 'S.M', 'G.H', 'S.H']].sum()
tmp['Sv%L'] = tmp['S.L']/(tmp['S.L'] + tmp['G.L'])
tmp['Sv%M'] = tmp['S.M']/(tmp['S.M'] + tmp['G.M'])
tmp['Sv%H'] = tmp['S.H']/(tmp['S.H'] + tmp['G.H'])
tmp[['source', 'Sv%L', 'Sv%M', 'Sv%H']]

# Show the distribution of adjusted 5v5 save percentages

# stat_table_sh.head(10)
# stat_table_sh["source"].value_counts()
stat_table_sh.loc[stat_table_sh['source'] == '5v5.csv', 'adj_5v5.csv'].plot(kind='hist')
plt.show()

# Who correlation of the shorthanded version with the 5v5 version
# Note this isn't controlling for number of rows per goalie

tmp = stat_table_sh[stat_table_sh['TOI'] > 1000]

print(tmp[['adj', 'adj_agg']].corr(method='pearson'))
print(tmp[['adj', 'adj_agg']].corr(method='spearman'))

# Show correlations with Bayesian-adjusted metric

stat_table_bayes = generate_statistics(results_most)

tmp = stat_table_bayes[stat_table_bayes['TOI'] > 1000]

print(tmp[['adj', 'adj_agg']].corr(method='pearson'))
print(tmp[['adj', 'adj_agg']].corr(method='spearman'))

print(tmp[['adj_bayes', 'adj_agg']].corr(method='pearson'))
print(tmp[['adj_bayes', 'adj_agg']].corr(method='spearman'))

tmp = stat_table_bayes[stat_table_bayes['source'] == '5v5.csv']
tmp['adj'].plot(kind='hist')
plt.show()

tmp['adj_bayes'].plot(kind='hist')
plt.show()

# We can observe different tightening of the distribution by playing with weights.
# 0.5 here might give a reasonable distribution - however, I'd much prefer to
# actually give a better grounding based on a smarter prior
stat_table_bayes_weak = generate_statistics(results_most, 0.5)

#input = stat_table_bayes_weak
input = stat_table_bayes

tmp = input[(input['source'] == '5v5.csv') & (input['TOI'] > 1000) & (input['season'] == 20142015)]
tmp['adj_rank'] = tmp.adj.rank(ascending=False)
tmp['adj_agg_rank'] = tmp.adj_agg.rank(ascending=False)
tmp['adj_bayes_rank'] = tmp.adj_bayes.rank(ascending=False)
print tmp[['season', 'Name', 'TOI', 'adj', 'adj_rank', 'adj_agg', 'adj_agg_rank', 'adj_bayes', 'adj_bayes_rank']].sort(columns="adj_bayes_rank")

# Rank comparison between 5v5 and shorthanded

tmp = stat_table_sh[(stat_table_sh['source'] == '5v5.csv') & 
                       (stat_table_sh['TOI'] > 1000) & 
                       (stat_table_sh['season'] == 20142015)]
tmp['adj_rank'] = tmp.adj.rank(ascending=False)
tmp['adj_agg_rank'] = tmp.adj_agg.rank(ascending=False)
tmp['adj_bayes_rank'] = tmp.adj_bayes.rank(ascending=False)
print tmp[['season', 'Name', 'TOI', 'adj', 'adj_agg', 'adj_rank', 'adj_agg_rank']].sort(columns="adj_agg_rank")

# Plots comparing movements

tmp.plot(kind='scatter', x='adj', y='adj_bayes', s=tmp['TOI']/10)
plt.xlim(89, 95)
plt.ylim(89, 95)
plt.xlabel('AdSv%')
plt.ylabel('AdSv% with Shorthanded')
plt.show()

tmp.plot(kind='scatter', x='adj', y='adj_bayes', s=tmp['TOI']/10)
plt.xlim(89, 95)
plt.ylim(89, 95)
plt.xlabel('AdSv%')
plt.ylabel('Bayes AdSv%')
plt.show()

tmp.plot(kind='scatter', x='adj_agg', y='adj_bayes', s=tmp['TOI']/10)
plt.xlim(89, 95)
plt.ylim(89, 95)
plt.xlabel('AdSv% with Shorthanded')
plt.ylabel('Bayes AdSv%')
plt.show()

# Compare NY and Ottawa goalies

tmp2 = tmp[tmp['Name'].isin(['Henrik.Lundqvist', 'Cam.Talbot', 'Andrew.Hammond', 'Craig.Anderson', 'Robin.Lehner'])]
print tmp2[['season', 'Name', 'TOI', 'adj', 'adj_agg', 'adj_rank', 'adj_agg_rank', 'adj_5v5.csv', 'adj_short.csv']]

# Break down NY and Ottawa goalies by shot type

tmp3 = stat_table3[(stat_table3['Name'].isin(['Henrik.Lundqvist', 'Cam.Talbot', 'Andrew.Hammond', 'Craig.Anderson', 'Robin.Lehner'])) & 
                   (stat_table3['season'] == 20142015)]

print tmp3[['Name', 'source', 'Sv%L', 'Sv%M', 'Sv%H']].sort(columns=['Name', 'source'])
