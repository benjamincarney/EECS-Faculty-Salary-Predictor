# EECS 486 Final Project: EECS Faculty Salary Predictor

*Contributers: Jai Padalkar, Aaron Balestrero, Noah Erikson, Spencer Vagg, and Benjamin Carney*

<a href="rel"><img src="/NeuralNetwork.png" align="right" height="200" width="233" ></a>



## Introduction

This repository contains code that our team wrote for our EECS 486: Information Retrieval final project during the Winter of
2019. Our goal was to predict University of Michigan EECS faculty salaries based on a variety of data points including:
official title with the university, research involvement, years of experience, and a variety of other unique indicators. To
generate a salary prediction for each professor, our team trained a neural network on datasets pooled from several sources.

---

## Directory Structure and Software Descriptions

In an attempt organize the various files found within our project, we've broken our repository into a number of different
directories. The naming conventions that we have chosen to use should be fairly self explanatory, but in case not, we've 
chosen to provide  descriptions for the contents of each directory. The first five listed headings/directories below contain
actual pieces of software and descriptions of how to run each one, while "Presentation" and "Data" simply contain various
auxillary files.

### GoldStandardCrawler

This directory contains a Python script that was used to collect the names, official titles, and salaries of each faculty
member within the University of Michigan EECS department, however it technically is capable of being run on any URL that
falls under the http://www.umsalary.info/deptsearch.php?Dept= domain. To run the program, simply type the following:

```python

python3 goldstandardcrawl.py 

```
You will be prompted to enter a URL that must follow the http://www.umsalary.info/deptsearch.php?Dept= naming convention.
After you have entered a URL, you will be prompted to enter the name of the department in which you would like to collect
data from. After doing this, the program will run and produce a .csv containing the names of the faculty members within 
that department followed by their official title, annual salary and newline in the format "department"_goldstandard.csv.


### LinkedInCrawler

This directory contains a python script that we deprecated in place of LinkedInParser.py. To run the deprecated program, type the following:

```python

python LinkedInWebCrawler.py EECS_Dept_Salary.txt crawler_output.csv

```
The program will simulate a browser session using Google chrome. The key difference between this and other
crawlers has to do with the necessity of logging-in before search. Per LinkedIn's policy, found at
https://www.linkedin.com/legal/user-agreement, before search can be performed one must have a valid LinkedIn profile and be logged-in to access other user profiles. 

NOTE: Set the LinkedIn user name and password fields in parameters.py

Since this approach led to blacklisting and throttling of our search capabilities, we dropped it in favor of LinkedInParser.py. To run the program, simply type the following:

```python
python LinkedInParser.py EECS_Dept_Salary.csv parser_output.csv

```

This program will extract the degree information from a static source of html pages, which are included
in the htmlDownloads/ folder. 
Although an in-place solution would be ideal and require no memory footprint, we had to improvise given
the circumstances. 

### RateMyProfessorCrawler

This directory contains a Python script that was used to gather data on EECS professors from the University of Michigan. To run the program, type the following:

```python

python3 rmpcrawler.py EECS_Dept_Salary.txt

```
This program uses the names of professors in the given file and searches them on RateMyProfessor. The program's way of extracting the names to search on RateMyProfessor makes the assumption that the given file is EECS_Dept_Salary.txt. However, this could be reworked to use a different file. The search process itself produces three output files containing the data found on professors, with default values for any that weren't found. The resulting data is separated into three files. One contains the data for all professors, found or not. One contains data for only those found. The last contains the names of those that weren't found, along with default placeholder data. The program also prints a summary of the search process upon its conclusion.


### Google Scholar Crawler
This directory contains a python script to pull data from Google Scholar on all professors and researchers associated with the University of Michigan. To run the file type the following:

```python
python3 scholar_crawler.py
```

The script will start by extracting the names and profile links of all researchers associated with the University of Michigan. That file will be saved in the same directory as userLinks.txt. The script will use this file to request the Google Scholar profile page of each researchers, parallelized over 10 subprocesses. Each subprocess will save in JSON format the researcher's name, url, number of citations, h-index and i-10 index from their profile. Finally, these JSON files will be merged into one file named GoogleScholar_profiles.json. The key for each object in the final JSON file is the researcher's name. 

### NeuralNetwork
This directory contains the models used to load and run the dataset. There is a jupyter notebook file and a python file that contain the same code. To run the python file, type the following:

```python
python NeuralNet486.py
```
This will run all three models on the specified dataset. To change which dataset is used, since we have three (entire dataset, top 4 predictors, and only Rate My Professor), you have to do two things. First, change the boolean values in the loader function to have it load the desired dataset. Second, make sure the correct value is in the nn.Batchnorm1d layers under the Neural Net class. 4 is used for the top 4 and RMP datasets and 18 is used when you run the entire dataset.

### Presentation

We used version control for making edits to our poster presentation.  We also included a few data visualizations that Spencer created for our presentation.

### Data

This directory contains a series of .csv files that represent the combined datasets across all four of the different
platforms that we used to extract data from. 

---
