# EECS 486 Final Project: EECS Faculty Salary Predictor

Contributers: Jai Padalkar, Aaron Balestrero, Noah Erikson, Spencer Vagg, and Benjamin Carney
<a href="rel"><img src="/NeuralNetwork.png" align="right" height="200" width="233" ></a>



## Abstract

This repository contains code that our team wrote for our EECS 486: Information Retrieval final project during the Winter of
2019. Our goal was to predict University of Michigan EECS faculty salaries based on a variety of data points including:
official title with the university, research involvement, years of experience, and a variety of other unique indicators. To
generate a salary prediction for each professor, our team trained a neural network on datasets pooled from several sources.
In this repository you will find the software that we used to collect this data, as well as the software that we used to
construct our neural network. Descriptions of these different pieces of software, as well as the content found within the
directories that they are nested in, can all be found below.

---

## Layout/File Structure

In an attempt to add some organization to this repository, we've broken our project up into various directories. The naming
conventions that we have chosen to use should be fairly self explanatory, but in case not, we've chosen to provide the
following descriptions for each directory below:

#### GoldStandardCrawler

#### LinkedInCrawler

#### RateMyProfessorCrawler

#### SalaryReleaseData

#### NeuralNetwork

#### Presentation

We used version control for making edits to our poster presentation.  We also included a few data visualizations that Spencer created for our presentation.

#### Data

This directory contains a series of .csv files that represent the combined datasets across all four of the different
platforms that we used to extract data from.

---

## Software

Below you will find information on how to run the various Python scripts that our team wrote for data collection and creating our neural network.

#### GoldStandardCrawler

Found inside of: /GoldStandardCrawler

Run with:

```python

python3 goldstandardcrawl.py 

```

#### GoogleScholarCrawler

#### RateMyProfessorCrawler

#### LinkedInCrawler

#### NeuralNetwork


