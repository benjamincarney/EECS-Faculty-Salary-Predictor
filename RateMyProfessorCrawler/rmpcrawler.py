"""Web crawler for RateMyProfessor."""

from bs4 import BeautifulSoup
import requests
import os
import click

def extract_prof_page_url(relative_link):
    """Return absolute professor page url using relative_link."""
    # professor pages all start with this base, then just add the relative link to it to get to prof page
    prof_page_base = "https://www.ratemyprofessors.com"
    prof_page_url = prof_page_base
    prof_page_url += relative_link

    return prof_page_url

def get_search_url_ann_arbor(first_name, last_name):
    """Return a url for a RateMyProfessor search on UM Ann Arbor for first_name and last_name."""
    # search base is for UM Ann Arbor
    # use search base and add first and last names to it as a query to complete the search
    search_url_base = "https://www.ratemyprofessors.com/search.jsp?queryoption=HEADER&queryBy=teacherName&schoolName=University+of+Michigan&schoolID=1258&query="
    search_url = search_url_base
    search_url += first_name
    search_url += "+"
    search_url += last_name

    return search_url

def get_search_url_dearborn(first_name, last_name):
    """Return a url for a RateMyProfessor search on UM Dearborn for first_name and last_name."""
    # search base is for UM Dearborn
    # use search base and add first and last names to it as a query to complete the search
    search_url_base = "https://www.ratemyprofessors.com/search.jsp?queryoption=HEADER&queryBy=teacherName&schoolName=University+of+Michigan+-+Dearborn&schoolID=1534&query="
    search_url = search_url_base
    search_url += first_name
    search_url += "+"
    search_url += last_name

    return search_url

def get_overall_quality_rating(soup):
    """Return the professor's overall quality rating found in prof page soup."""
    # after examining prof page page source, this isolates the tag containing overall quality rating
    highest_container = soup.find("div", class_="breakdown-container quality")
    mid_container = highest_container.findChildren()[0]
    exact_container = mid_container.findChildren()[0]
    # extract text from tag and remove whitespace
    rating_as_string = exact_container.get_text()
    rating_as_string = rating_as_string.strip()

    return rating_as_string

def get_level_of_difficulty(soup):
    """Return the professor's level of difficulty rating found in prof page soup."""
    # after examining prof page page source, this isolates the tag containing level of difficulty rating
    highest_container = soup.find_all("div", class_="breakdown-header")[1]
    mid_container = highest_container.findChildren()[4]
    exact_container = mid_container.findChildren()[0]
    # extract text from tag and remove whitespace
    level_of_difficulty_as_string = exact_container.get_text()
    level_of_difficulty_as_string = level_of_difficulty_as_string.strip()

    return level_of_difficulty_as_string

def get_would_take_again_percentage(soup):
    """Return the professor's would take again percentage found in prof page soup."""
    # after examining prof page page source, this isolates the tag containing wta percentage
    highest_container = soup.find_all("div", class_="breakdown-header")[1]
    mid_container = highest_container.findChildren()[0]
    exact_container = mid_container.findChildren()[2]
    # extract text from tag and remove whitespace
    percentage_as_string = exact_container.get_text()
    percentage_as_string = percentage_as_string.strip()
    # wta percentage is either a number followed by % or N/A
    # if N/A, no processing needed
    # if number followed by %, remove % and convert number to a decimal
    if percentage_as_string[0] == 'N':
        return percentage_as_string
    processed_percentage = ""
    for char in percentage_as_string:
        if char == "%":
            break
        processed_percentage += char
    processed_percentage = float(processed_percentage) / 100
    processed_percentage = str(processed_percentage)

    return processed_percentage

def get_quantity_of_reviews(soup):
    """Return the amount of reviews this professor has, found in prof page soup."""
    # after examining prof page page source, this isolates the tag containing quantity of reviews
    container = soup.find("div", class_="table-toggle rating-count active")
    # extract text from tag and remove whitespace
    quantity_as_string = container.get_text()
    quantity_as_string = quantity_as_string.strip()
    # isolate just the number and not the words that follow
    processed_quantity = ""
    for char in quantity_as_string:
        if char.isspace():
            break
        processed_quantity += char

    return processed_quantity

def get_num_professor_hits(soup):
    """Return the number of professor results in search page soup."""
    # after examining search result page, this isolates the tag containing number of results
    result_count = soup.find_all("div", class_="result-count")[1]
    result_count_text = result_count.get_text()
    result_count_text = result_count_text.strip()
    # if no results, the text will be "Your search didn't return any results."
    if result_count_text[0] == "Y":
        return 0
    # else, it will be "Showing x-x of y results", and y is the number of results
    split_str = result_count_text.split()
    count = split_str[3]

    return int(count)

def prof_has_been_reviewed(soup):
    """Return true if professor has been reviewed, false otherwise."""
    # if professor has no page, will be redirected to the add page url
    # title of this page will start with "Add A Review for"
    title = soup.find("title")
    title_text = title.get_text()
    title_text = title_text.strip()
    if title_text[0] == 'A':
        return False
    return True

def get_relative_prof_link(soup):
    """Return the relative professor page link from search page soup."""
    # after examining search result page, this isolates tag of first search result
    results = soup.find_all("li", class_="listing PROFESSOR")[0]
    container = results.findChildren()[0]
    # get the link to prof page from the tag
    relative_link = container.get("href")

    return relative_link

def get_soup(url):
    """Return a BeautifulSoup soup object constructed from url."""
    r = requests.get(url)
    soup = BeautifulSoup(r.text, features='lxml')

    return soup

def get_professor_data(first_name, last_name):
    """Crawl RateMyProfessor to get revelant professor data as formatted string."""
    # default data values
    data = dict()
    data['first_name'] = first_name
    data['last_name'] = last_name
    data['found'] = False
    data['quality'] = "N/A"
    data['difficulty'] = "N/A"
    data['wtapercent'] = "N/A"
    data['reviews'] = "N/A"
    data['found_at'] = "N/A"
    # attempt to find professor at ann arbor first
    aa_search = get_search_url_ann_arbor(first_name, last_name)
    aa_search_soup = get_soup(aa_search)
    aa_num_hits = get_num_professor_hits(aa_search_soup)
    # if there is a hit, attempt to get professor data from ann arbor
    if aa_num_hits:
        relative_link = get_relative_prof_link(aa_search_soup)
        aa_result = extract_prof_page_url(relative_link)
        aa_result_soup = get_soup(aa_result)
        # prof might be in RateMyProfessor but not reviewed
        # get relevant data if reviewed, else default data remains
        if prof_has_been_reviewed(aa_result_soup):
            data['found_at'] = aa_result
            data['found'] = True
            data['quality'] = get_overall_quality_rating(aa_result_soup)
            data['difficulty'] = get_level_of_difficulty(aa_result_soup)
            data['wtapercent'] = get_would_take_again_percentage(aa_result_soup)
            data['reviews'] = get_quantity_of_reviews(aa_result_soup)
    # if no ann arbor hits, attempt to find professor at dearborn
    else:
        db_search = get_search_url_dearborn(first_name, last_name)
        db_search_soup = get_soup(db_search)
        db_num_hits = get_num_professor_hits(db_search_soup)
        # if there is a hit, attempt to get professor data from dearborn
        if db_num_hits:
            relative_link = get_relative_prof_link(db_search_soup)
            db_result = extract_prof_page_url(relative_link)
            db_result_soup = get_soup(db_result)
            # prof might be in RateMyProfessor but not reviewed
            # get relevant data if reviewed, else default data remains
            if prof_has_been_reviewed(db_result_soup):
                data['found_at'] = db_result
                data['found'] = True
                data['quality'] = get_overall_quality_rating(db_result_soup)
                data['difficulty'] = get_level_of_difficulty(db_result_soup)
                data['wtapercent'] = get_would_take_again_percentage(db_result_soup)
                data['reviews'] = get_quantity_of_reviews(db_result_soup)
    # return state of data after attempted searches
    return data

def get_output_string(data):
    """Return a string of professor data formatted as follows:

    first_name,last_name found:<True/False> quality:<rating> difficulty:<rating> wtapercent:<decimal> reviews:<quantity>
    """
    output_string = data['first_name'] + "," + data['last_name']
    output_string += " "
    output_string += "found:{0}".format(data['found'])
    output_string += " "
    output_string += "quality:{0}".format(data['quality'])
    output_string += " "
    output_string += "difficulty:{0}".format(data['difficulty'])
    output_string += " "
    output_string += "wtapercent:{0}".format(data['wtapercent'])
    output_string += " "
    output_string += "reviews:{0}".format(data['reviews'])
    output_string += " "
    output_string += "at:{0}\n".format(data['found_at'])

    return output_string

@click.command()
@click.argument("filename", nargs=1)
def main(filename):
    # opening output
    print('RateMyProfessor crawler started.')
    print('Opening {0}...'.format(filename))
    # attempt to open filename, in this case, EECS_Dept_Salary.txt and read in name data
    opened_file = open(filename, 'r')
    list_of_names = []
    for i, line in enumerate(opened_file):
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
            list_of_names.append(first_name + " " + last_name)
    opened_file.close()
    # sanity check
    if not list_of_names:
        print('Something went wrong with reading in data from {0}.'.format(filename))
        print('Aborting...')
        return
    print('Data successfully read in.')
    # set up output files and data tracking vars
    all_output = open("rmpoutput_all.txt", "w")
    found_output = open("rmpoutput_found.txt", "w")
    not_found_output = open("rmpoutput_not_found.txt", "w")
    num_found = 0
    num_total = 0
    num_not_found = 0
    # crawl RateMyProfessor using each name
    print('Initiating crawling...')
    for name in list_of_names:
        first_name = name.split()[0]
        last_name = name.split()[1]
        data = get_professor_data(first_name, last_name)
        output = get_output_string(data)
        if data['found']:
            num_found += 1
            found_output.write(output)
        else:
            num_not_found += 1
            not_found_output.write(output)
        all_output.write(output)
        num_total += 1
    all_output.close()
    found_output.close()
    not_found_output.close()
    # closing output
    print('Crawling finished.')
    print('Number of faculty searched: {0}'.format(num_total))
    print('Number of faculty with RateMyProfessor pages: {0}'.format(num_found))
    print('Number of faculty without RateMyProfessor pages: {0}'.format(num_not_found))
    print('Output files: ')
    print('rmpoutput_all.txt - contains data from all faculty (those not found have default data that cannot be used)')
    print('rmpoutput_found.txt - contains RMP data from faculty that have RateMyProfessor pages')
    print('rmpoutput_not_found.txt - contains default data for faculty that do not have RateMyProfessor pages')
    print('Though the default data cannot be used, these faculty are still present as a record.')

if __name__ == '__main__':
    main()
