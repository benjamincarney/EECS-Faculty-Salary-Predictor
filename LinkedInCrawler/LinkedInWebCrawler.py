import sys
import os
import csv
import random
import parameters
from time import sleep
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.keys import Keys
from parsel.selector import Selector


def main(argv):

	writer = csv.writer(open(argv[2], 'w', newline=''))
	writer.writerow(['FirstName'
		'LastName',
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
	delayRequest()

	profMap = buildProfessorMap(argv)

	# user-targeted google search approach
	for user in profMap:
		google_query = parameters.query
		google_query += ' AND "' + user + '"'
		driver.get('https://www.google.com')
		# locate search form by_name
		search_query = driver.find_element_by_name('q')
		# send_keys() to simulate the search text key strokes
		search_query.send_keys(google_query)
		delayRequest()
		# send_keys() to simulate the return key 
		search_query.send_keys(Keys.RETURN)
		# add a 5 second pause loading each URL
		linked_in_urls = getProfileURLs(driver)
		# go to the page contained in the first result of the query
		delayRequest()
		if len(linked_in_urls) > 0:
			driver.get(linked_in_urls[0])
		if ('authwall' in driver.current_url):
			driverLogIn(driver) 
		delayRequest()
		# assign the source code for the webpage to variable sel
		sel = Selector(text=driver.page_source)
		# get full name (first + last name) of user
		fullName = getFullName(sel)
		# does the page belong to the user undergoing search?
		if fullName:
			if (user not in fullName):
				print('Name \"' + user + '\" in file does not match \"' + fullName + '\"')
			else:
				print(user + " found!")
		# returns a list of degree name followed by
		# field of study in alternating sequence
		degreeInfo = getDegreeInfo(sel)
		# returns a list of year started and year completed for each degree.
		# only need the first element to determine how long primary
		# degree has been held
		dateInfo = getDateInfo(sel)
		# write to csv file
		flushOutput(writer, user, degreeInfo, dateInfo)
		# wait before redirecting to google search page

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
		print("Element not found")
		pass

	# locate email form by_id
	redirectPath = '//*[starts-with(@class, "form__input--floating")]/*[starts-with(@id, "username")]'
	abrPath = '//*[starts-with(@id, "username")]'
	username = driver.find_element_by_id('login-email')
	if not username:
		username = driver.find_element_by_xpath(redirectPath)
	if not username:
		username = driver.find_element_by_xpath(abrPath)
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


'''
Requires: fileName
Modifies: nothing
Effects: returns a list of professor names with a 
		format adopted by first and last name conventions by reading in
		the "EECS_Dept_Salary.txt" file
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


'''
Requires: driver
Modifies: nothing
Effects: returns a list of profile urls ranked within the Google search results page. 
'''
def getProfileURLs(driver):
	linked_in_urls = driver.find_elements_by_class_name('iUh30')
	linked_in_urls = [url.text for url in linked_in_urls]
	return linked_in_urls


'''
Requires: sel
Modifies: nothing
Effects: extracts and returns the name associated 
		with a given LinkedIn profile page.
'''
def getFullName(sel):
	# xpath to extract the first h1 text (to extract first and last name)
	fullName = sel.xpath('//h1/text()').extract_first()
	if fullName:
		# removes any newline characters
		fullName = fullName.strip()
	print(fullName)
	return fullName


'''
Requires: sel
Modifies: nothing
Effects: extracts degree information including degree and field of study (FOS)
		from the LinkedIn html page provided in sel and
		returns it as a list formatted as [degree, FOS]
'''
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

	print(degreeInfo)
	return degreeInfo


'''
Requires:sel
Modifies: nothing
Effects: extracts date information regarding degree
		from the LinkedIn html page provided in sel and
		returns it as a list formatted as [YearStarted, YearEarned]
'''
def getDateInfo(sel):

	educationInfo = '//*[contains(@class, "education-item__content")]'
	yearStartsWith = '/span/*[starts-with(@class, "date-range")]/text()'
	yearContains = '/span/*[contains(@class, "date")]/text()'
	profileViewPath = '//*[starts-with(@class, "pv-entity__dates")]/span/text()'

	dateInfo = sel.xpath(profileViewPath).getall()
	if not dateInfo:
		dateInfo = sel.xpath(educationInfo + yearStartsWith).getall()

	if not dateInfo:
		dateInfo = sel.xpath(educationInfo + yearContains).getall()

	if dateInfo:
		for i in range(0, len(dateInfo)):
			dateInfo[i] = dateInfo[i].strip()

	print(dateInfo)
	return dateInfo


'''
Requires: sel
Modifies: nothing
Effects: extracts and returns job title from the LinkedIn html page provided in sel.
		NOTE: not needed for relevant data points
'''
def getJobInfo(sel):
	xPathJobTitle = '//*[contains(@class, "pv-top-card-section")]/text()'
	job_title = sel.xpath(xPathJobTitle).extract_first()
	return job_title


'''
Requires: writer, fullName, degreeInfo, dateInfo
Modifies: writer
Effects: prints the target information gather from LinkedIn
		to file.
'''
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


'''
Requires: nothing
Modifies: nothing
Effects: puts the program to sleep for randomly chosen interval of time. 
		Set debugInfo to false to stop debug print statements. 
'''
def delayRequest(debugInfo=True):
	sleepTime = random.randint(1, 100)
	if debugInfo:
		print('Sleeping for ' + str(sleepTime) + ' sec...')
	sleep(sleepTime)
	return


'''
Requires: driver, writer
Modifies: driver, writer
Effects: Simulates the in-sequence web page loading of the top 10
		search results provided by Google. All commands issued affect
		the state of the google chrome driver named 'driver'.

		NOTE: this method was deprecated after LinkedIn throttled our
		automated search after a cap of 60 URL requests.
'''
def top10Pass(driver, writer):
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
	return


'''
Requires: command line arguments from argv
Modifies: nothing
Effects: returns a dictionary professor names for efficient lookup
'''
def buildProfessorMap(argv):

	# Read names from EECS_Dept_Salary.txt
	# and store by first name followed by last name
	names = readNameInformation(argv[1])
	profMap = dict(zip(names, [None] * len(names)))

	# output for LinkedInParser.py
	out_file = open("EECS_Dept_Names.txt", "w")
	for key in profMap:
		out_file.write(key + ' University of Michigan' + '\n')

	return profMap


if __name__ == "__main__":
	main(sys.argv)
