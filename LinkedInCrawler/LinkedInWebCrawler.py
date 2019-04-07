import sys
import os
import csv
import parameters
from time import sleep
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from parsel.selector import Selector

# The directory containing the file "EECS_Dept_Salary.txt"
nameInfoDirectory = "C:/Users/Aaron/Desktop/EECS-486-Final-Project/SalaryReleaseData/"


def main(argv):

	# read names from EECS_Dept_Salary.txt. Store in list
	# by first and last name
	names = readNameInformation(argv[1])
	profMap = dict(zip(names, [None]*len(names)))

	writer = csv.writer(open(parameters.file_name, 'w', newline=''))
	writer.writerow(['LastName',
		'FirstName',
		'Degree',
		'FOS',
		'YearStarted',
		'YearEarned'])

	# instantiates the chrome driver 
	driver = webdriver.Chrome(parameters.driverDirectory)

	# driver.get method() will navigate to a page given by the URL address
	driver.get('https://www.linkedin.com')

	# logs the user into their LinkedIn account
	driverLogIn(driver)

	driver.get('https://www.google.com')

	# locate search form by_name
	search_query = driver.find_element_by_name('q')

	# send_keys() to simulate the search text key strokes
	search_query.send_keys(parameters.query)

	# send_keys() to simulate the return key 
	search_query.send_keys(Keys.RETURN)

	# wait for page to load
	sleep(3.0)

	i = 0
	while(True):
		google_list_page = driver.current_url
		linked_in_urls = getProfileURLs(driver)
		for linked_in_url in linked_in_urls[:]:
			# get the profile URL
			driver.get(linked_in_url)
			# add a 5 second pause loading each URL
			sleep(5)
			# assigning the source code for the webpage to variable sel
			sel = Selector(text=driver.page_source)
			# get full name (first + last name) of user
			fullName = getFullName(sel)
			# returns a list of degree name followed by
			# field of study in alternating sequence
			degreeInfo = getDegreeInfo(sel)
			# returns a list of year started and year completed for each degree.
			# only need the first element to determine how long primary
			# degree has been held
			dateInfo = getDateInfo(sel)
			# write to csv file
			flushOutput(writer, fullName, degreeInfo, dateInfo)

		try:
			driver.get(google_list_page)
			next_result_element = driver.find_element_by_id('pnnext')
			next_page = next_result_element.get_attribute('href')
			driver.get(next_page)
		except NoSuchElementException:
			print('No next page found ')
			break
		i += 1
		if (i > 30):
			break
	# terminates the application
	driver.quit()

	return


'''
Requires: driver
Modifies: driver
Effects: Logs the user into their LinkedIn account if log-in succeeds.
		Attempts to handle case where "https://www.linkedin.com" sends
		user to the login request page.
'''


def driverLogIn(driver):

	myUserEmail = parameters.linked_in_email
	myUserPassword = parameters.linked_in_password

	try:
		login_button = driver.find_element_by_class_name('nav__button-secondary')
		login_button.click()
		sleep(5.0)
	except NoSuchElementException:
		pass

	# locate email form by_id
	username = driver.find_element_by_id('login-email')

	# send_keys() to simulate key strokes
	username.send_keys(myUserEmail)

	# sleep for 0.5 seconds
	sleep(0.5)

	# locate password form by_id
	password = driver.find_element_by_id('login-password')

	# send_keys() to simulate key strokes
	password.send_keys(myUserPassword)
	sleep(0.5)

	login_button = driver.find_element_by_id('login-submit')

	# click event on login button
	login_button.click()
	sleep(0.5)

	# Manage account login check may appear
	if ("/check/manage-account" in driver.current_url):
		confirmButton = driver.find_element_by_class_name('primary-action')
		confirmButton.click()
		sleep(0.5)

	return


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
	if fullName:
		# removes any newline characters
		fullName = fullName.strip()
	print(fullName)
	return fullName


def getDegreeInfo(sel):
	xPathDegree = '//*[starts-with(@class, "pv-entity__comma-item")]/text()'
	degreeInfo = sel.xpath(xPathDegree).getall()
	if degreeInfo:
		for i in range(0, len(degreeInfo)):
			degreeInfo[i] = degreeInfo[i].strip()
	else:
		degreeInfo = sel.xpath(xPathDegree).extract_first()

	print(degreeInfo)
	return degreeInfo


def getDateInfo(sel):
	xPathDate = './/*[starts-with(@class, "pv-entity__dates")]/span/time/text()'
	dateInfo = sel.xpath(xPathDate).getall()
	if dateInfo:
		for i in range(0, len(dateInfo)):
			dateInfo[i] = dateInfo[i].strip()
	else:
		dateInfo = sel.xpath(xPathDate).extract_first()

	print(dateInfo)
	return dateInfo


def getJobInfo(sel):
	xPathJobTitle = '//*[starts-with(@class, "pv-top-card-section__headline")]/text()'
	job_title = sel.xpath(xPathJobTitle).extract_first()
	return job_title


def flushOutput(writer, fullName, degreeInfo, dateInfo):
	last_name = fullName.split()[1]
	first_name = fullName.split()[0]
	degree = ""
	field_of_study = ""
	startDate = ""
	endDate = ""
	if degreeInfo:
		degree = degreeInfo[0]
		if (len(degreeInfo) > 1):
			field_of_study = degreeInfo[1]
	if dateInfo:
		startDate = dateInfo[0]
		if (len(dateInfo) > 1):
			endDate = dateInfo[1]
	writer.writerow([last_name, first_name, degree, field_of_study, startDate, endDate])
	return

def buildProfDictionary(profMap):

	return

if __name__ == "__main__":
	main(sys.argv)
