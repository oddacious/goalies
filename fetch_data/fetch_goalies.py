#!/usr/bin/python

"""
Download all basic goalie stats, possibly ever.

Usage:
    fetch_goalies.py [--min_year=<1918>] [--max_year=<2017>] [--csv] [--personal]

Options:
    -h --help          Show this screen.
    --min_year=YEAR    Earliest season to download (year it ended in). [default: 1918]
    --max_year=YEAR    Latest season to download (default: ending next year).
    --csv              Create CSV output and write to STDOUT.
    --personal         Download the personal pages for each goalie.
"""

import sys
import datetime
import urlparse
import csv
import requests
import docopt
from bs4 import BeautifulSoup

from cacheblob import Cacheblob

EARLIEST_YEAR = 1917
BASE_URL = "http://www.hockey-reference.com/leagues/NHL_{}_goalies.html"
SUPPORTED_STATISTICS = {"age": "age",
                        "games_goalie": "GP",
                        "starts_goalie": "GS",
                        "wins_goalie": "W",
                        "losses_goalie": "L",
                        "ties_goalie": "T/O",
                        "goals_against": "GA",
                        "shots_against": "SA",
                        "saves": "SV",
                        "save_pct": "SV%",
                        "goals_against_avg": "GAA",
                        "shutouts": "SO",
                        "min_goalie": "MIN",
                        "quality_starts_goalie": "QS",
                        "quality_start_goalie_pct": "QS%",
                        "really_bad_starts_goalie": "RBS",
                        "ga_pct_minus": "GA%-",
                        "gs_above_avg": "GSAA",
                        "goals": "G",
                        "assists": "A",
                        "points": "PTS",
                        "pen_min": "PIM"}

CSV_ORDER = ["player_name", "player_url", "year", "team_name", "team_url", "age", "GP",
             "GS", "W", "L", "T/O", "GA", "SA", "SV", "SV%", "GAA", "SO", "MIN", "QS",
             "QS%", "RBS", "GA%-", "GSAA", "G", "A", "PTS", "PIM"]

def is_number(value):
    """Return if value is numeric, based on attempting to coerce to float.

    :param value: The value in question.
    :returns: True if the float(value) does not throw an exception, otherwise False.
    """
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

def fetch_page_if_needed(cache, url, duration=datetime.timedelta(days=1)):
    """Fetch a page from `url` and store it in the `cache`, unless it is already cached.

    :param cache: The cacheblob object to use for storage.
    :param url: The URL to fetch, which will be the key in the cache.
    :param duration: How long to store the value in the cache [default: 1 day].

    :returns: The page text, or None if it was not status 200.
    """
    res = cache.fetch(url)
    if res:
        return res.value

    page = requests.get(url)
    if page.status_code != 200:
        print("Error fetching page \"{0}\": {1}".format(url, page.status_code))
        return None
    else:
        cache.store(index=url, value=page.text, duration=duration)
        return page.text

def parse_stats(html):
    """Parse a HTML block that contains player stats.

    :param html: The HTML code containing the stats.

    :returns: None if not a sufficient HTML block, a dictionary otherwise.
    """
    result = {}

    if not html.find('th', attrs={"data-stat": "ranker"}):
        return None

    for link in html.findAll("a"):
        if link["href"].find("player") >= 0:
            result["player_url"] = link["href"]
            result["player_name"] = link.contents[0]
        if link["href"].find("teams") >= 0:
            result["team_url"] = link["href"]
            result["team_name"] = link.contents[0]

    # Header rows
    if "player_name" not in result:
        return None

    for code, stat in SUPPORTED_STATISTICS.items():
        for link in html.findAll("td", attrs={"data-stat": code}):
            if len(link.contents):
                result[stat] = link.contents[0]
        if stat not in result:
            result[stat] = None

    return result

def fetch_data(min_year, max_year, personal):
    """Fetch all the data required.

    :param min_year: First year to fetch for (calendar year the season ended in).
    :param max_year: Last year to fetch for (calendar year the season ended in).
    :param personal: Whether or not to also fetch the personal stat pages per goalie.

    :returns: A list of dictionaries that contain one item per goalie.
    """
    cache = Cacheblob.cache(handler="mongo", opts={"table_name": "goalie_stats_html"})

    parsed_uri = urlparse.urlparse(BASE_URL)
    base = '{uri.scheme}://{uri.netloc}/'.format(uri=parsed_uri)

    all_goalies = []

    for year in range(min_year, max_year + 1):
        url = BASE_URL.format(year)
        page = fetch_page_if_needed(cache, url)

        if not page:
            continue

        tree = BeautifulSoup(page, "lxml")

        for row in tree.findAll('tr'):
            result = parse_stats(row)

            if not result:
                continue

            result["year"] = year

            if personal and result["player_url"]:
                player_url = base + result["player_url"]
                fetch_page_if_needed(cache,
                                     player_url,
                                     duration=datetime.timedelta(days=365))

            all_goalies.append(result)

    return all_goalies

def output_csv(data, header_order=CSV_ORDER):
    """Write output in CSV format to STDOUT.

    :param data: A list of dictionaries of CSV data to output.
    :param header_order: The order to put the entries in.
    """
    if not len(data):
        return

    writer = csv.DictWriter(sys.stdout, fieldnames=header_order)
    writer.writeheader()
    for row in data[1:]:
        writer.writerow(row)

def main():
    """Entry point of the code. This parses the options and calls `fetch_data`.
    """
    args = docopt.docopt(__doc__)

    min_year = args["--min_year"]
    max_year = args["--max_year"]

    if not is_number(min_year):
        raise ValueError("--min_year requires a numeric value")
    elif float(min_year) < EARLIEST_YEAR:
        raise ValueError("--min_year must be {} or greater".format(EARLIEST_YEAR))

    season_ending_next_year = datetime.date.today().year + 1

    if max_year is None:
        max_year = season_ending_next_year
    elif not is_number(max_year):
        raise ValueError("--max_year requires a numeric value")
    elif float(max_year) > season_ending_next_year:
        raise ValueError("--max_year cannot be greater than the current season")

    all_data = fetch_data(min_year=int(min_year),
                          max_year=int(max_year),
                          personal=args["--personal"])

    if args["--csv"]:
        output_csv(all_data)

if __name__ == "__main__":
    main()
