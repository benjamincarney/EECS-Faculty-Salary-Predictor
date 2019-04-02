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

    links = deque([])

    for link in soup.find_all('a'):
        links.append(link.get('href'))

    return links


def main(argv):

    global visitedPages

    if len(argv) != 2:
        print("Must specify proper command line arguments")


if __name__ == "__main__":
    main(sys.argv)
