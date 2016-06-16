#!/usr/bin/python

import os
import re

import pandas as pd

class woi_games():
    """This class can read and hold data from war-on-ice.com files. Essentially it is a CSV importer""" 

    # This describes which files will be read when searching a directory. However all of the files must be in CSV format
    _types_to_read = ['csv']

    _season_index = 'season'
    _games_table = None
    _latest_season = None

    # This is important. WOI CSV outputs do not contain metadata about the query used to generate them. So for my
    # personal ease, I put that into the filename. E.g. aniemi 5v5.csv
    _5v5_naming = '5v5'

    _SH_naming = 'SH'
    _folder_delim = '/'

    def __init__(self, entries):
        """Create an object to hold games results as output by the war-on-ice.com interface"""

        # If provided a string, consider that a file and add it
        if isinstance(entries, basestring):
            self.add_dir_or_file(entries)
        else:
            # If provided a list, add each entry
            for item in entries:
                self.add_dir_or_file(item)

    def add_dir_or_file(self, item):
        """The item should be a file created by war-on-ice.com, or a folder containing such items.
           If item is a directory, each file within it that is within our attempted types (e.g. csv)
           will be imported
        """

        # If it's a directory, add all CSV files in it (but do not traverse the tree)
        if os.path.isdir(item):
            for filename in os.listdir(item):

                # Only attempt to read from a set of filetypes
                for filetype in self._types_to_read:
                    if filename.lower().endswith('.' + filetype.lower()):
                        self.add_dir_or_file(item + self._folder_delim + filename)
        else:
            # If the user wants to import certain CSV files but not others, she needs to specify them all explicitly
            try:
                data = pd.read_csv(item)

                # Save the filename
                data['source'] = pd.Series([item] * len(data.index), index=data.index)

                # Save the situation type, calculated by filename. Shaky, ain't it?
                data['situation'] = pd.Series(map(lambda x: 
                                                            #self._5v5_naming if re.search(self._5v5_naming + '\.[^\.]+$', x) 
                                                            self._5v5_naming if x.find(self._5v5_naming) != -1 
                                                            else self._SH_naming, 
                                                        data['source']), 
                                                    index=data.index)

                # WOI does not have a shots column, but I find it handy
                data['Sh.L'] = data['G.L'] + data['S.L']
                data['Sh.M'] = data['G.M'] + data['S.M']
                data['Sh.H'] = data['G.H'] + data['S.H'] 

                if self._games_table is None:
                    self._games_table = data
                else:
                    self._games_table = pd.concat([self._games_table, data])
            except IOError:
                print "Could not import file \"{0}\"".format(item)

    def replace_games_table(self, results):
        """Replace the data with a provided version. I felt this was the most flexible way to manage this"""

        self._games_table = results
        return results

    def games_table(self):
        """Accessor to get the results. There is no separation from the different files"""

        return self._games_table

    def latest_season(self):
        """Get the results from the latest season"""

        # Confirm that the data is well-formed (note: if one file wasn't, then you're in trouble...)
        if not self._season_index in self._games_table:
            raise LookupError, "Input data did not contain column \"{0}\"".format(self._season_index)

        latest_season = max(self._games_table[self._season_index])
        return self._games_table[self._games_table[self._season_index] == latest_season]

    def latest_season_instance(self):
        """Return a subset of this object with only the latest season""" 

        # Beware: Not deep copying here!
        item = self

        latest_season = max(self._games_table[self._season_index])
        item.replace_games_table(self._games_table[self._games_table[self._season_index] == latest_season])

        return item

    def situations(self):
        """Get a list of unique situations represented in this sample"""

        return self._games_table.situation.unique()
