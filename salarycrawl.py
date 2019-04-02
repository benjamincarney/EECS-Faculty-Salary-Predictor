import lxml
import requests
import urllib
from collections import deque
from bs4 import BeautifulSoup
import sys


def parseURL(url):

    # send a GET request to the specified URL
    try:
        page = urllib.request.urlopen(url)
    except:
        print("Couldn't open webpage")
        return []

    # read that content in and parse it with BS4
    html = page.read()
    soup = BeautifulSoup(html, 'html.parser')

    rows = []

    # this line returns a BS object containing the main table on the page
    index = soup.find_all("table", {"class": "index"})
    print(index)
    # print(soup.get_text())
    # print(type(rows))


def main(argv):

    global visitedPages

    if len(argv) != 1:
        print("Must specify proper command line arguments")

    parseURL('http://www.umsalary.info/deptsearch.php?Dept=EECS+-+CSE+Division&Year=0&Campus=0')


if __name__ == "__main__":
    main(sys.argv)
