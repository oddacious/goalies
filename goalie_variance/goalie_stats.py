#!/usr/bin/python

import math

import pandas as pd

class goalie_stats:
    """This class will perform statistics on the goalie data retrieved from war-on-ice.com.
    
    Key concepts:
    location/difficulty - One of L, M, or H, indicating where on the ice a shot came from.
    situation - Most commonly 5v5 or SH, indicating the status of the goalie's team at time of shot.

    The following methods are expected to be called externally:
    games_table
    situations
    streak_data
    find_threshold
    build_streaks [affects internal state]
    generate_statistics [affects internal state]
    """

    _stat_table = None
    _agg_stats = None
    _streak_data = None

    _is_flattened = 0

    def __init__(self, games):
        """A woi_games object is expected as the only parameter"""
        self._stat_table = games

    def games_table(self):
        """Accessor for the woi_games object"""

        if self._stat_table is not None:
            return self._stat_table.games_table()
        else:
            raise LookupError, "Should have been initialized with a woi_games object"

    def situations(self):
        """Accessor for the situation types present in the dataset"""

        if self._stat_table is not None:
            return self._stat_table.situations()
        else:
            raise LookupError, "Should have been initialized with a woi_games object"

    def streak_data(self):
        """Accessor for the streak data calculated by build_streaks"""

        if self._streak_data is not None:
            return self._streak_data
        else:
            raise LookupError, "Need to call build_streaks() first"

    def build_streaks(self, max_depth):
        """Calculate statistics for each number of consecutive games, up to streaks of "max_depth" length"""

        goalies_flat = self.flatten_all('5v5')
        shot_totals = self.sum_aggregate_shots()
        all_situations = self._stat_table.situations()

        # I rely on the correct ordering of these fields. Note that 'Date' should
        # exist because this only makes sense on the game-by-game data.
        if 'Date' in goalies_flat.columns:
            goalies_flat = goalies_flat.sort_values(by=['Name', 'season', 'Date'])
        else:
            goalies_flat = goalies_flat.sort_values(by=['Name', 'season'])

        results = {}

        # Every game in the data set in the start of one or more streaks
        for i in range(len(goalies_flat)):
            totals = {}

            for j in range(i, len(goalies_flat)):

                row = goalies_flat.iloc[j]

                streak_len = j - i + 1

                if (streak_len > max_depth or 
                    goalies_flat.iloc[i]['Name'] != goalies_flat.iloc[j]['Name'] or 
                    goalies_flat.iloc[i]['season'] != goalies_flat.iloc[j]['season']):
                    break

                (numer, denom, totals) = self.update_streak_counters(row, totals, shot_totals)

                # Debug this error case if observed
                if math.isnan(numer/denom):
                    print "DEBUG: {0}/{1} with {2} total shots".format(numer, denom, total_situ_shots)

                if streak_len not in results:
                    results[streak_len] = [numer/denom]
                else:
                    results[streak_len].append(numer/denom)

        self._streak_data = results

        return results  

    def find_threshold(self, num_games, value):
        """Utility to find where a statistic value of "value" would rank, within streaks of "num_games" played"""

        sample = self._streak_data

        if sample is None:
            raise LookupError, "Call build_streaks() before find_threshold()"

        if num_games not in sample:
            raise IndexError, "Streaks of {0} games not found".format(num_games)

        lower_bound = 1.0*sum(i >= value for i in sample[num_games])/len(sample[num_games])
        upper_bound = 1.0*sum(i > value for i in sample[num_games])/len(sample[num_games])

        # If there is a tie, return the average of lower and upper values
        return 1 - (lower_bound + upper_bound)/2

    def generate_statistics(self):
        """Return the twice adjusted save percentage"""

        self.calculate_adjusted_sv()

        self.flatten_all('5v5', ['adj'])

        return self.calculate_twice_adjusted_sv()

    def sum_aggregate_shots(self):
        """Get the league-wide goals, shots, and saves, aggregated"""

        # No longer merging on season - decided to use aggregate so when we compare goalies across seasons we give them
        # the same distribution of shots, keeping the adjustment uniform
        chunk_shots = self._stat_table.games_table().groupby(['situation'], as_index=False)
        agg_stats = chunk_shots[['G.L', 'S.L', 'G.M', 'S.M', 'G.H', 'S.H', 'Sh.L', 'Sh.M', 'Sh.H']].sum()

        self._agg_stats = agg_stats

        return agg_stats

    def flatten_across_situations(self, vars_to_flatten):
        """Create a new column with "vars_to_flatten" for each situation present"""

        # The "gcode" column only exists in the split-by-games WOI results
        if 'gcode' in self._stat_table.games_table().columns:
            join_index = ['season', 'Name', 'gcode']
        else:
            join_index = ['season', 'Name']

        stat_table = self._stat_table.games_table()

        for situation in self._stat_table.situations():
            stat_table = pd.merge(stat_table, 
                                stat_table.loc[stat_table['situation'] == situation, join_index + vars_to_flatten], 
                                on=join_index, 
                                how='left', 
                                suffixes=('', '_' + str(situation)))
            
        self._stat_table.replace_games_table(stat_table)

        return stat_table

    def flatten_aggregates(self):
        """For each row, include the league-aggregate shots and goals, and number of rows represented"""

        stat_table = self._stat_table.games_table()
        chunk_shots = stat_table.groupby(['season', 'situation'], as_index=False)
        agg_shots = chunk_shots[['Sh']].sum()
        num_goalies = pd.DataFrame({'count' : stat_table.groupby(['season', 'situation'], as_index=False).size()}).reset_index()

        for situation in self._stat_table.situations():
            stat_table = pd.merge(stat_table, 
                                agg_shots.loc[agg_shots['situation'] == situation, ['season', 'Sh', 'G']], 
                                on=['season'], 
                                how='left', 
                                suffixes=('','_agg_' + situation))
            stat_table = pd.merge(stat_table, 
                                num_goalies.loc[num_goalies['situation'] == situation, ['season', 'count']], 
                                on=['season'], 
                                how='left', 
                                suffixes=('', '_' + situation))

            # Rename
            if 'count_' + situation not in stat_table.columns:
                stat_table.rename(columns={'count': 'count_' + situation}, inplace=True)   

        self._stat_table.replace_games_table(stat_table)

        return stat_table

    def flatten_all(self, situation, stat_set=None):
        """Flatten across all situations and also incorporate aggregates
        
        Returns the rows where situation matches the provided situation.
        stat_set is which variables to flatten over. If None, a default set is used
        """
        
        if stat_set == None:
            stat_set = ['G.L', 'Sh.L', 'G.M', 'Sh.M', 'G.H', 'Sh.H', 'G.L_all', 
                'Sh.L_all', 'G.M_all', 'Sh.M_all', 'G.H_all', 'Sh.H_all']

        if self._is_flattened == 0:
            self.flatten_across_situations(stat_set) 

            self.flatten_aggregates()

            self._is_flattened = 1

        return self._stat_table.games_table()[self._stat_table.games_table()['situation'] == situation]

    def update_streak_counters(self, row, totals, shot_totals):
        """Add another row to the ongoing stat count, returning a numerator, denominator, and updated counts"""

        SvPerc = {}
        total_shots = {}
        numer = 0
        denom = 0

        for situ in self.situations():
            for loc in ['L', 'M', 'H']:

                goal_key = 'G.' + loc + '_' + situ
                shot_key = 'Sh.' + loc + '_' + situ

                if goal_key not in totals:
                    if math.isnan(row[goal_key]):
                        totals[goal_key] = 0
                    else:
                        totals[goal_key] = row[goal_key]
                    if math.isnan(row[shot_key]):
                        totals[shot_key] = 0
                    else:
                        totals[shot_key] = row[shot_key]
                else:
                    if not math.isnan(row[goal_key]):
                        totals[goal_key] = (totals[goal_key] + row[goal_key])
                    if not math.isnan(row[shot_key]):
                        totals[shot_key] = (totals[shot_key] + row[shot_key])

                # I could generate shot_totals in this function, but since it will be called many times,
                # and always internally, I chose to let it be passed in
                total_shots[loc] = shot_totals.loc[shot_totals.situation == situ, 'Sh.' + loc].item()

                if totals[shot_key] > 0:
                    SvPerc[loc + '_' + situ] = 1 - 1.0 * totals[goal_key] / totals[shot_key]
                else:
                    # Use the league average if the goalie hasn't faced any shots
                    total_goals = shot_totals.loc[shot_totals.situation == situ, 'G.' + loc].item() 
                    SvPerc[loc + '_' + situ] = 1 - (1.0 * total_goals / total_shots[loc])

            total_situ_shots = total_shots['L'] + total_shots['M'] + total_shots['H']

            adj = ((SvPerc['L_' + situ] * total_shots['L'] + 
                    SvPerc['M_' + situ] * total_shots['M'] + 
                    SvPerc['H_' + situ] * total_shots['H']) /
                    total_situ_shots)

            numer = numer + adj * total_situ_shots
            denom = denom + total_situ_shots

        return (numer, denom, totals)

    def calculate_adjusted_sv(self):
        """Calculate adjusted save percentage"""

        # This will generate self._agg_stats used below
        self.sum_aggregate_shots()

        # No longer merging on season - decided to use aggregate so when we compare goalies across seasons we give them
        # the same distribution of shots, keeping the adjustment uniform
        stats = pd.merge(self._stat_table.games_table(), 
                self._agg_stats, 
                on=['situation'], 
                how='left', 
                suffixes=('','_all'))
        
        # Calculate unadjusted save percentage, by location
        for loc in ['L', 'M', 'H']:
            stats.ix[pd.isnull(stats['Sv%' + loc]), 'Sv%' + loc] = stats['G.' + loc + '_all'] / stats['Sh.' + loc + '_all']  
        
        # Calculate adjusted save percentage by weighing the three locations by their league-wide averages
        stats['adj'] = (stats['Sv%L'] * stats['Sh.L_all'] + stats['Sv%M'] * stats['Sh.M_all'] + 
                        stats['Sv%H'] * stats['Sh.H_all']) / (stats['Sh.L_all'] + stats['Sh.M_all'] + stats['Sh.H_all'])

        return self._stat_table.replace_games_table(stats)

    def calculate_twice_adjusted_sv(self):
        """Calculate twice adjusted save percentage"""

        stat_table = self._stat_table.games_table()
        shot_totals = self.sum_aggregate_shots()

        #print stat_table

        numer = [len(stat_table)*0]
        denom = [len(stat_table)*0]

        # TODO: Rewrite this to let people do the shot adjustment on a per-season basis, if they so choose

        # Much like how adjusted save percentage is weighed by location, here we weigh again by situation
        for situ in self.situations():
            x = shot_totals.loc[shot_totals.situation == situ, 'Sh.H']
            y = stat_table['Sh_agg_' + situ]
            numer = numer + (stat_table['adj_' + situ] * stat_table['Sh_agg_' + situ]).fillna(
                100*(1 - ((shot_totals.loc[shot_totals.situation == situ, 'G.L'] + 
                shot_totals.loc[shot_totals.situation == situ, 'G.M'] +
                shot_totals.loc[shot_totals.situation == situ, 'G.H']) /
                (shot_totals.loc[shot_totals.situation == situ, 'Sh.L'] + 
                shot_totals.loc[shot_totals.situation == situ, 'Sh.M'] +
                shot_totals.loc[shot_totals.situation == situ, 'Sh.H'])).item()) * stat_table['Sh_agg_' + situ])
            denom = denom + stat_table['Sh_agg_' + situ]   

        stat_table['adj_agg'] = numer/denom

        return stat_table
