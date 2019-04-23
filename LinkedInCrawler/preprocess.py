# Name: Aaron Balestrero
# Uniqname: balestaa

import sys
import os
import re
import stemmer
import operator

encliticMappings = {
	"d": "had",
	"m": "am",
	"s": "is",
	"ve": "have",
	"re": "are",
	"ll": "will",
	"t": "not"
}

stopwords = [
	"a",
	"all",
	"an",
	"and",
	"any",
	"are",
	"as",
	"at",
	"be",
	"been",
	"but",
	"by",
	"few",
	"from",
	"for",
	"have",
	"he",
	"her",
	"here",
	"him",
	"his",
	"how",
	"i",
	"in",
	"is",
	"it",
	"its",
	"many",
	"me",
	"my",
	"none",
	"of",
	"on",
	"or",
	"our",
	"she",
	"some",
	"the",
	"their",
	"them",
	"there",
	"they",
	"that",
	"this",
	"to",
	"us",
	"was",
	"what",
	"when",
	"where",
	"which",
	"who",
	"why",
	"will",
	"with",
	"you",
	"your"
]

'''
Removes SGML tags from the input string 'raw_text.'
Returns a string containing no SGML tags.
'''


def removeSGML(raw_text):
	text = re.sub('([\b<A-Z/>\b])', '', raw_text)
	return text


'''
Builds a list of tokens from the input string 'text.'
Returns a list of tokens.
'''


def tokenizeText(text):
	# print(text)
	# tokenList = re.findall('[^\r\n\t\f\v\s\.\,]+|(?:[a-zA-Z0-9\.]){2,}', text)
	# tokenList = re.findall('[^\r\n\t\f\v\s\.\,]{4,}|[a-zA-Z0-9\.]{2,}', text)
	tokenList = re.findall('[^\r\n\t\f\v\s\.\,]{4,}|[a-zA-Z0-9\.]{2,}|[A-Za-z]', text)

	# print(len(tokenList))
	# print(tokenList)

	apostrophize(tokenList)

	# print(len(tokenList))
	# print(tokenList)
	return tokenList


'''
Removes stopwords from the input list 'tokens.'
Returns a list of tokens.
'''


def removeStopwords(tokens):
	filteredTokenList = []
	for token in tokens[:]:
		if (token not in stopwords):
			filteredTokenList.append(token)

	return filteredTokenList


'''
Stems the tokens provided in the input list 'tokens.'
Returns a list of tokens.
'''


def stemWords(tokens):
	stemmedTokens = []
	pStemmer = stemmer.PorterStemmer()
	for token in tokens:
		stemmedTokens.append(pStemmer.stem(token, 0, len(token) - 1))

	return stemmedTokens


'''
Breaks conjuctions into their distinct parts.
Returns an updated list of tokens.
'''


def apostrophize(tokens):
	segments = []
	for token in tokens[:]:
		if '\'' in token:
			segments = token.split('\'')
			if (token in stopwords):
				# enclitic expansion
				if (len(segments) >= 2):
					clitic = segments[1]
					if (segments[1] != ''):
						clitic = encliticMappings[segments[1]]
					if (clitic == "not"):
						segments[0].replace('n', '')
					tokens.remove(token)
					tokens.append(segments[0])
					tokens.append(clitic)
			else:
				# possessive pronoun conjunction
				if (len(segments) >= 2):
					if (token in tokens):
						tokens.remove(token)
						tokens.append(segments[0])
						tokens.append("\'" + segments[1])
	return tokens


def encliticExpansion(word):
	segments = []
	return segments


'''
(1) Open the folder containing the data collection, provided as
	the first argument on the command line.

(2) For each file, apply, in order: removeSGML , tokenizeText , removeStopwords , stemWords.

(3) In addition, write code to determine (this is after step 2 above):
	- the total number of words in the collection (numbers should be counted as words)
	- vocabulary size (i.e., number of unique terms)
	- most frequent 50 words in the collection, along with their frequencies (list in reverse order
	of their frequency)
'''


def main(argv):
	OUTFILE = open("preprocess.output", "w")
	masterWords = [] # contains the number of words in the collection
	frequencyMap = {} # cotains key-value pairs of word-frequency map

	file_list = os.listdir(argv)
	directory = argv
	if (not argv.endswith('/')):
		directory = argv + '/'
	for file_name in file_list[:]:
		print(file_name)
		INFILE = open(directory + file_name, "r")
		text = INFILE.readline()
		for line in INFILE:
			text += line
		scrubbedText = removeSGML(text)
		tokens = tokenizeText(scrubbedText)
		filteredTokens = removeStopwords(tokens)
		stemmedTokens = stemWords(filteredTokens)
		masterWords = masterWords + stemmedTokens
		for word in stemmedTokens[:]:
			if (frequencyMap.get(word, 'N') == 'N'):
				frequencyMap[word] = 1
			else:
				frequencyMap[word] += 1

	# print resuls to preprocess.output
	OUTFILE.write('Words ' + str(len(masterWords)) + '\n')
	OUTFILE.write('Vocabulary ' + str(len(frequencyMap)) + '\n')
	OUTFILE.write('Top 50 words\n')

	rankedVocabulary = sorted(frequencyMap.items(), key=operator.itemgetter(1), reverse=True)

	for wordFreqPair in rankedVocabulary[:50]:
		OUTFILE.write(wordFreqPair[0] + " " + str(wordFreqPair[1]) + '\n')


if __name__ == '__main__':
    main(sys.argv[1])