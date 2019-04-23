import sys
import os
import csv
import random
import string
import parameters
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from parsel.selector import Selector

degrees = ['phd', 'masters', 'bachelors', 'associates']
tokenizer = RegexpTokenizer(r'\w+')

def main(argv):

	csv_name_file = open(argv[1], 'r')
	reader = csv.reader(csv_name_file)
	next(reader)

	# build professor dictionary containing lastName-firstName key-value pairs
	profMap = buildProfMap(reader)
	print(profMap)

	file_list = os.listdir('htmlDownloads/htmlPages/')
	writer = csv.writer(open(argv[2], 'w', newline=''))
	writer.writerow(['LastName',
					'FirstName',
					'Degree',
					'FOS',
					'YearStarted',
					'YearEarned'])

	# Add .html page names to a list
	filtered_list = []
	for file_name in file_list[:]:
		if file_name.endswith("LinkedIn.html"):
			if "_ Search _" not in file_name:
				filtered_list.append(file_name)
	file_list = filtered_list

	for lastName in profMap:
		file_name = None
		# find html page according to last name
		file_name = findUsersHtmlPage(file_list, lastName)
		if not file_name:
			print(lastName + " has no page!")
			continue
		htmlDoc = readHtmlDoc(file_name)
		sel = Selector(text=htmlDoc)
		fullName = getFullName(sel)
		degreeInfo = getDegreeInfo(sel)
		dateInfo = getDateInfo(sel)
		print(fullName)
		print(degreeInfo)
		print(dateInfo)
		flushOutput(writer, fullName, degreeInfo, dateInfo)

	return


'''
Requires: driver
Modifies: driver
Effects: Logs the user into their LinkedIn account if log-in succeeds.
		Attempts to handle case where "https://www.linkedin.com" sends
		user to the login request page.
'''


def readNameInformation(fileName):
	global nameInfoDirectory
	names = []
	fullPath = nameInfoDirectory
	if (not fileName[0].isupper() or fileName[1] != ":"):
		# if not already a full path
		fullPath += fileName
	INFILE = open(fullPath, "r")
	for i, line in enumerate(INFILE):
		if i < 2:
			continue
		elif i % 2 == 0:
			continue
		else:
			name = line[23:63]
			name = name.rstrip()
			split_name = name.split(',')
			last_name = split_name[0]
			first_name = split_name[1].split()[0]
			# middle_name = ""
			# if (len(split_name[1].split()) > 1):
			# 	middle_name = split_name[1].split()[1]
			names.append(first_name + " " + last_name)
	INFILE.close()
	# sanity check
	if not names:
		print('Something went wrong with reading in data from {0}.'.format(fileName))
		print('Aborting...')
		exit()
	return names


def getProfileURLs(driver):
	linked_in_urls = driver.find_elements_by_class_name('iUh30')
	linked_in_urls = [url.text for url in linked_in_urls]
	return linked_in_urls


def getFullName(sel):
	# xpath to extract the first h1 text (to extract first and last name)
	fullName = sel.xpath('//h1/text()').extract_first()
	if not fullName:
		fullName = sel.xpath('//*[starts-with(@class, "name")]/text()').extract_first()
	if fullName:
		# removes any newline characters
		fullName = fullName.strip()
	return fullName


def getDegreeInfo(sel):

	profileViewPath = '//*[starts-with(@class, "pv-entity__comma-item")]/text()'
	degreeStartsWith = '//*[starts-with(@class, "education-item__degree-info")]/text()'
	degreeContains = '//*[contains(@class, "degree")]/text()'

	degreeInfo = sel.xpath(profileViewPath).getall()
	if not degreeInfo:
		degreeInfo = sel.xpath(degreeStartsWith).getall()

	if not degreeInfo:
		degreeInfo = sel.xpath(degreeContains).getall()

	if degreeInfo:
		for i in range(0, len(degreeInfo)):
			degreeInfo[i] = degreeInfo[i].strip()

	return degreeInfo


def getDateInfo(sel):

	educationInfo = '//*[contains(@class, "education-item__content")]'
	yearStartsWith = '/span/*[starts-with(@class, "date-range")]/text()'
	yearContains = '/span/*[contains(@class, "date")]/text()'
	# profileViewPath = '//*[starts-with(@class, "pv-entity__dates")]/span/text()'
	profileViewPath = '//*[starts-with(@class, "pv-entity__dates")]/*/*/text()'

	dateInfo = sel.xpath(profileViewPath).getall()
	if not dateInfo:
		dateInfo = sel.xpath(educationInfo + yearStartsWith).getall()

	if not dateInfo:
		dateInfo = sel.xpath(educationInfo + yearContains).getall()

	if dateInfo:
		for i in range(0, len(dateInfo)):
			dateInfo[i] = dateInfo[i].strip()

	return dateInfo


def getJobInfo(sel):
	xPathJobTitle = '//*[contains(@class, "pv-top-card-section")]/text()'
	job_title = sel.xpath(xPathJobTitle).extract_first()
	return job_title


def flushOutput(writer, fullName, degreeInfo, dateInfo):
	last_name = fullName.split()[1]
	first_name = fullName.split()[0]
	degree = "N/A"
	field_of_study = "N/A"
	startDate = "N/A"
	endDate = "N/A"
	if degreeInfo:
		rawString = degreeInfo[0]
		# print(rawString)
		degree = getNormalizedDegree(rawString)
		if (len(degreeInfo) > 1):
			field_of_study = degreeInfo[1]
	if dateInfo:
		startDate = dateInfo[0]
		if (len(dateInfo) > 1):
			endDate = dateInfo[1]
	writer.writerow([first_name, last_name, degree, field_of_study, startDate, endDate])
	return


# def buildPrevScrapeMap():
# 	scrapedUsers = {}
# 	csv_file = open('alreadyScrapedUsers.csv', 'r')
# 	sanity_reader = csv.reader(csv_file)
# 	next(sanity_reader)
# 	for row	in sanity_reader:
# 		print(row)
# 		scrapedUsers[row[1] + " " + row[0]] = None
# 	return scrapedUsers


def readHtmlDoc(file_name):
	INFILE = open('htmlDownloads/htmlPages/' + file_name, errors='ignore')
	doc = ""
	for line in INFILE:
		doc += line
	return doc


def buildProfMap(reader):
	profMap = {}
	for row in reader:
		uid = row[0]
		revOrdName = row[1]
		campus = row[2]
		apptTitle = row[3]
		apptDept = row[4]
		apptAnnFtr = row[5]
		apptFtrBasis = row[6]
		apptFrac = row[7]
		amtSal = row[8]

		name_list = revOrdName.split(",")
		last_name = name_list[0]
		first_name_group = name_list[1].split()
		first_name = first_name_group[0]
		if (profMap.get(last_name, "N") == "N"):
			profMap[last_name] = [first_name]
		else:
			profMap[last_name].append(first_name)
	return profMap

def getNormalizedDegree(rawString):
	global degrees, tokenizer
	# normalize by lowercasing
	normalizedStr = rawString.lower()
	words = normalizedStr.split()
	# print(words)
	# remove punctuation
	table = str.maketrans('', '', string.punctuation)
	stripped = [w.translate(table) for w in words]
	# print(stripped)
	for cand in degrees:
		if cand in stripped:
			return cand
	return 'N/A'

def findUsersHtmlPage(file_list, lastName):
	# print(lastName)
	for file_name in file_list[:]:
		# print(file_name)
		if (lastName in file_name):
			return file_name
	return None

if __name__ == "__main__":
	main(sys.argv)
