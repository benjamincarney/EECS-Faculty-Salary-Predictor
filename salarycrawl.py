import lxml
import requests
import urllib
from bs4 import BeautifulSoup
import re
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

    # remove all javascript and stylesheet code, makes things a bit easier
    for script in soup(["script", "style"]):
        script.extract()

    # finds all table on page and returns the third table of those pages
    table = soup.find_all('table')[2]

    # add each row within this table to a list
    rows = []
    for row in table.find_all('tr'):
        rows.append(str(row))

    # faculty member information starts at row 2
    for faculty in rows:

        # obtain first name of faculty member within this row
        firstname = re.search(re.escape('FName=')+"(.*?)"+re.escape('&amp'), faculty)
        firstNameStr = ''
        if firstname:
            firstNameStr = str(firstname.group(1))
            if '+' in firstNameStr:
                firstNameStr = firstNameStr.replace('+', ' ')
            print(firstNameStr  + " ", end="")

        # obtain last name of faculty member within this row
        lastname = re.search(re.escape('LName=')+"(.*?)"+re.escape('&amp'), faculty)
        lastNameStr = ''
        if lastname:
            lastNameStr = str(lastname.group(1))
            if '+' in lastNameStr:
                lastNameStr = lastNameStr.replace('+', ' ')
            elif '%27' in lastNameStr:
                lastNameStr = lastNameStr.replace('%27', ' ')
            print(lastNameStr + ' ', end='')

        # obtain title of faculty member within this row
        title = re.search(re.escape('Title=')+"(.*?)"+re.escape('&amp'), faculty)
        titleStr = ''
        if title:
            titleStr = str(title.group(1))
            if '+' in titleStr:
                titleStr = titleStr.replace('+', ' ')
            elif '%27' in titleStr:
                titleStr = titleStr.replace('%27', ' ')
            elif '%2' in titleStr:
                titleStr = titleStr.replace('%2', ' ')
            print(titleStr + ' ',end='')

        # obtain salary of current faculty member
        salary = re.search(re.escape('"right">')+"(.*?)"+re.escape('</td>'), faculty)
        salaryStr = ''
        if salary:
            salaryStr = str(salary.group(1))
            if '+' in salaryStr:
                salaryStr = salaryStr.replace('+', ' ')
            elif '%27' in salaryStr:
                salaryStr = salaryStr.replace('%27', ' ')
            elif '%2' in salaryStr:
                salaryStr = salaryStr.replace('%2', ' ')
            print(salaryStr)

        if not firstNameStr or not lastNameStr or not titleStr or not salaryStr:
            continue

        f = open("goldstandard.csv", "a")
        facultyString = firstNameStr + " " + lastNameStr + ',' + titleStr + ',' + salaryStr + "\n"
        f.write(facultyString)
        f.close()




def main(argv):

    global visitedPages

    if len(argv) != 1:
        print("Must specify proper command line arguments")

    url = 'http://www.umsalary.info/deptsearch.php?Dept=EECS+-+CSE+Division&Year=0&Campus=0'

    parseURL(url)


if __name__ == "__main__":
    main(sys.argv)
