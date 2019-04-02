import lxml
import requests
import urllib
from collections import deque
from bs4 import BeautifulSoup
import sys


def parseURL(url):

    try:
        page = urllib.request.urlopen(url)
    except:
        print("Couldn't open webpage")
        return []

    html = page.read()
    soup = BeautifulSoup(html, 'html.parser')

    rows = []

    rows = soup.find_all('tr')

    print(rows)


def main(argv):

    global visitedPages

    if len(argv) != 1:
        print("Must specify proper command line arguments")

    parseURL('http://www.umsalary.info/deptsearch.php?Dept=EECS+-+CSE+Division&Year=0&Campus=0')


if __name__ == "__main__":
    main(sys.argv)
