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
import json
from multiprocessing import Process


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
    try:
        os.makedirs("profiles")
    except FileExistsError:
        print("Profiles Directory Already Exists")
    users = open("userLinks.txt", "r")
    domain = "https://scholar.google.com"
    links = []
    for line in users.readlines():
        links.append(line.replace("\n", ""))
    chunk = int(len(links)/10)
    for i in range(11):
        start = i*chunk
        end = min( ((i+1)*chunk), len(links) )
        if start >= len(links):
            break
        Process(target=processProfiles, args=(domain, links[start:end], end)).start()


def processProfiles(domain, links, counter):
    profiles = {}
    for link in links:
        time.sleep(2)
        r = requests.get(domain + link)
        if not r.ok:
            print("Error:", link)
            continue
        soup = BeautifulSoup(r.text, 'html.parser')
        tag = soup.find(id="gsc_prf_in")
        #print(tag.string)
        profiles[tag.string] = {}
        profiles[tag.string]["url"] = domain + link
        table = soup.find(id="gsc_rsb_st")
        nums = find_stats(table)
        profiles[tag.string]["citations"] = nums[0]
        profiles[tag.string]["citations-2014"] = nums[1]
        profiles[tag.string]["h-index"] = nums[2]
        profiles[tag.string]["h-index-2014"] = nums[3]
        profiles[tag.string]["i10-index"] = nums[4]
        profiles[tag.string]["i10-index-2014"] = nums[5]
    ofile = "profiles/profiles" + str(counter) + ".json"
    with open(ofile, "w") as outfile:
        json.dump(profiles, outfile)
    print("Finished Process:", counter)


def find_stats(table):
    nums = []
    if table is None:
        nums = [0,0,0,0,0,0]
        return(nums)
    body = table.find('tbody')
    for row in body.find_all('tr'):
        for cell in row.find_all('td'):
            temp = cell.string
            if temp.isdigit():
                nums.append(cell.string)
    return(nums)


def concatProfiles():
    results = []
    #load JSON files
    for file in os.listdir(path="profiles/"):
        path = "profiles/" + file
        with open(path, "r") as infile:
            results.append(json.load(infile))
    #write single JSON file
    with open("GoogleScholar_profiles.json", "w") as outfile:
        json.dump(results, outfile)


def main():
    print('started')
    start = time.time()
    findUsers()
    profiles = profileDict()
    concatProfiles()
    stop = time.time()
    runtime = (stop-start)/60
    print('Runtime: ', runtime)


if __name__ == '__main__':
    main()



