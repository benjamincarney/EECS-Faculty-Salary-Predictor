"""
    Jai Padalkar
    jaipad
    EECS 486
    """
import sys
import os
import re
from bs4 import BeautifulSoup
import requests
import string
import time
import scrapy


def findUsers():
    domain = "https://scholar.google.com"
    seedq = []
    seedq.append( "https://scholar.google.com/citations?view_op=view_org&hl=en&org=4770128543809686866")
    outfile = open("userLinks.txt", "w")
    urls = set()
    while len(seedq) > 0:
        r = requests.get(seedq[0])
        if not r.ok:
            continue
        soup = BeautifulSoup(r.text, 'html.parser')
        for link in soup.find_all('a'):
            userURL = link.get('href')
            if "user=" in userURL:
                urls.add(userURL)
        for link in soup.find_all('button'):
            if link.attrs['aria-label'] == "Next":
                if "onclick" not in link.attrs:
                    continue
                nextPage = link.attrs['onclick']
                nextPage = cleanURL(nextPage)
                print(nextPage)
                seedq.append(domain + nextPage)
        seedq.pop(0)
    #Once all profile links have been extracted
    for url in urls:
        outfile.write(url + "\n")


def cleanURL(url):
    ret = url.replace("window.location=", "")
    ret = ret.replace("'", "")
    ret = ret.replace("\\x3d", "=")
    ret = ret.replace("\\x26", "&")
    return(ret)


def profileDict():
    print('profiling')
    users = open("userLinks.txt", "r")
    profiles = {}
    domain = "https://scholar.google.com"
    count = 0
    for line in users.readlines():
        r = requests.get(domain + line.replace("\n", ""))
        if not r.ok:
            print('continuing')
            continue
        soup = BeautifulSoup(r.text, 'html.parser')
        tag = soup.find(id="gsc_prf_in")
        print(tag.string)
        profiles[tag.string] = {}
        profiles[tag.string]["url"] = domain + line
        table = soup.find(id="gsc_rsb_st")
        count += 1
        print(count)
    return(profiles)


def main():
    print('started')
    start = time.time()
    #findUsers()
    profiles = profileDict()
    stop = time.time()
    runtime = (stop-start)/60
    #print('Runtime: ', runtime)


if __name__ == '__main__':
    main()



