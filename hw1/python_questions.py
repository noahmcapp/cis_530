#!/usr/local/bin/python3.9

'''Homework 1 Python Questions

This is an individual homework
Implement the following functions.

Do not add any more import lines to this file than the ones
already here without asking for permission on Piazza.
Use the regular expression tools built into Python; do NOT use bash.
'''

import re

# TODO: should be foo_and_bar???
def check_for_foo_or_bar(text):
   '''Checks whether the input string meets the following condition.

   The string must have both the word 'foo' and the word 'bar' in it,
   whitespace- or punctuation-delimited from other words.
   (not, e.g., words like 'foobar' or 'bart' that merely contain
    the word 'bar');

   See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#match-objects

   Return:
     True if the condition is met, false otherwise.
   '''

   # start, end, whitespace, new line, punctation
   delim = '(^|$|[\s\\n.?!-_\\\/\[\]\{\}])'

   # delim string to match on either side
   s = '%s%s%s' % (delim, '%s', delim)

   # check both foo and bar
   return (not re.search(s % 'foo', text) is None and not re.search(s % 'bar', text) is None)
#
# end of check_for_foo_or_bar

def replace_rgb(text):
   '''Replaces all RGB or hex colors with the word 'COLOR'
   
   Possible formats for a color string:
   #0f0
   #0b013a
   #37EfaA
   rgb(1, 1, 1)
   rgb(255,19,32)
   rgb(00,01, 18)
   rgb(0.1, 0.5,1.0)

   There is no need to try to recognize rgba or other formats not listed 
   above. There is also no need to validate the ranges of the rgb values.

   However, you should make sure all numbers are indeed valid numbers.
   For example, '#xyzxyz' should return false as these are not valid hex digits.
   Similarly, 'rgb(c00l, 255, 255)' should return false.

   Only replace matching colors which are at the beginning or end of the line,
   or are space separated from the text around them. For example, due to the 
   trailing period:

   'I like rgb(1, 2, 3) and rgb(2, 3, 4).' becomes 'I like COLOR and rgb(2, 3, 4).'

   # See the Python regular expression documentation:
   https://docs.python.org/3.4/library/re.html#re.sub

   Returns:
     The text with all RGB or hex colors replaces with the word 'COLOR'
   '''

   # int and float strings, any whitespace on either side
   arg_int = '\s*[0-9]{1,}\s*'
   arg_float = '\s*[0-9]{1,}\.[0-9]?\s*'

   # define the hex and decimal rgb formats
   formats = [
      '#[0-9A-f]{3}', # 3 byte hex      
      '#[0-9A-f]{6}', # 6 byte hex      
      'rgb\(%s,%s,%s\)' % ((arg_int,)*3), # decimal int
      'rgb\(%s,%s,%s\)' % ((arg_float,)*3), # decimal float
   ]

   # add delimiter on either side of formats
   formats = ['(?<!\S)(%s)(?!\S)' % (f) for f in formats]
   for f in formats:
      text = re.sub(f, 'COLOR', text)
   return text
#
# end of replace_rgb

def edit_distance(str1, str2, debug=False):
   '''Computes the minimum edit distance between the two strings.

   Use a cost of 1 for all operations.

   See Section 2.4 in Jurafsky and Martin for algorithm details.
   Do NOT use recursion.

   Returns:
    An integer representing the string edit distance
    between str1 and str2
   '''

   # remove whitespace from strings
   str1, str2 = str1.strip(), str2.strip()
   
   # initialize the distance matrix
   n, m = len(str1), len(str2)
   D = []
   for i in range(n+1):
      D.append([])
      for j in range(m+1):
         D[-1].append(0)
   
   # initialize the first row and column -- distance from an empty string
   for i in range(1, n+1):
      D[i][0] = D[i-1][0] + del_cost(str1[i-1])
   for j in range(1, m+1):
      D[0][j] = D[0][j-1] + ins_cost(str2[j-1])
   
   if debug:
      print('Initialization:')
      display(D)

   # perform the recurrence relation
   for i in range(1, n+1):
      for j in range(1, m+1):
         D[i][j] = min([
            D[i-1][j] + del_cost(str1[i-1]),
            D[i-1][j-1] + sub_cost(str1[i-1], str2[j-1]),
            D[i][j-1] + ins_cost(str2[j-1]),
         ])

   if debug:
      print('Final:')
      display(D)
   
   return D[n][m]
#
# end of edit_distance

def display(x):
   print('')
   for row in x:
      print(row)
   print('')   
# NOTE: Levenshtein distance uses 1 as an assumption
def del_cost(x):
   return 1
# NOTE: Levenshtein distance uses 1 as an assumption
def ins_cost(x):
   return 1
# NOTE: Levenshtein distance uses 2 as an assumption
def sub_cost(x, y):
   if x == y:
      return 0
   else:
      return 1

def wine_text_processing(wine_file_path, stopwords_file_path):
   '''Process the two files to answer the following questions and output results to stdout.
   
   1. What is the distribution over star ratings?
   2. What are the 10 most common words used across all of the reviews, and how many times
      is each used?
   3. How many times does the word 'a' appear?
   4. How many times does the word 'fruit' appear?
   5. How many times does the word 'mineral' appear?
   6. Common words (like 'a') are not as interesting as uncommon words (like 'mineral').
     In natural language processing, we call these common words "stop words" and often
     remove them before we process text. stopwords.txt gives you a list of some very
     common words. Remove these stopwords from your reviews. Also, try converting all the
     words to lower case (since we probably don't want to count 'fruit' and 'Fruit' as two
     different words). Now what are the 10 most common words across all of the reviews,
     and how many times is each used?
   7. You should continue to use the preprocessed reviews for the following questions
     (lower-cased, no stopwords).  What are the 10 most used words among the 5 star
     reviews, and how many times is each used? 
   8. What are the 10 most used words among the 1 star reviews, and how many times is
     each used? 
   9. Gather two sets of reviews: 1) Those that use the word "red" and 2) those that use the word
     "white". What are the 10 most frequent words in the "red" reviews which do NOT appear in the
     "white" reviews?
   10. What are the 10 most frequent words in the "white" reviews which do NOT appear in the "red"
      reviews?

   No return value.
   '''

   # load the tab delimited data, ignore unreadable bytes
   reviews, ratings = zip(*[l.rstrip().split('\t') for l in open(wine_file_path, 'r', errors='ignore')])

   # get the unique ratings, sorted from longest to shortest
   uniq_ratings = sorted(list(set(ratings)), key=lambda x: len(x), reverse=True)

   # 1) get the frequency of each rating
   for uniq_rating in uniq_ratings:
      print('%s\t%s' % (uniq_rating, ratings.count(uniq_rating)))
   print()
   
   # load the word frequencies using a bad word finding method
   word_freqs = get_word_freqs(reviews, method='bad')

   # 2) get the top 10 most frequent words
   get_top_freqs(word_freqs)
   print()
   
   # 3) get the frequency of "a"
   print(word_freqs["a"])
   print()
   
   # 4) get the frequency of "fruit"
   print(word_freqs["fruit"])
   print()
   
   # 5) get the frequency of "mineral"
   print(word_freqs["mineral"])
   print()

   # load the stopwords
   stopwords = [l.rstrip() for l in open(stopwords_file_path, 'r')]

   # get the word frequencies using a better word finding method
   word_freqs = get_word_freqs(reviews, method='good', stopwords=stopwords)

   # 6) get the 10 most frequent words
   get_top_freqs(word_freqs)
   print()

   # 7) get the 10 most frequent words in 5 star reviews
   five_stars = [reviews[i] for i in range(len(ratings)) if ratings[i] == '*****']
   word_freqs = get_word_freqs(five_stars, method='good', stopwords=stopwords)
   get_top_freqs(word_freqs)
   print()

   # 8) get the 10 most frequent words in 1 star reviews
   one_stars = [reviews[i] for i in range(len(ratings)) if ratings[i] == '*']
   word_freqs = get_word_freqs(one_stars, method='good', stopwords=stopwords)
   get_top_freqs(word_freqs)
   print()

   # get the word frequencies of the "red" and "white" reviews
   # NOTE: this is inefficient since we're calling get_words twice per review :(
   red_reviews = [x for x in reviews if "red" in get_words(x, method='good', stopwords=stopwords)]
   white_reviews = [x for x in reviews if "white" in get_words(x, method='good', stopwords=stopwords)]
   red_word_freqs = get_word_freqs(red_reviews, method='good', stopwords=stopwords)   
   white_word_freqs = get_word_freqs(white_reviews, method='good', stopwords=stopwords)      

   # keep only the unique words between reviews
   del_red_keys = [x for x in red_word_freqs if x in white_word_freqs]
   del_white_keys = [x for x in white_word_freqs if x in red_word_freqs]
   for key in del_red_keys:
      del red_word_freqs[key]
   for key in del_white_keys:
      del white_word_freqs[key]

   # 9) get the 10 most frequent words in red reviews that are not in white reviews
   get_top_freqs(red_word_freqs)
   print()

   # 10) get the 10 most frequent words in white reviews that are not in red reviews
   get_top_freqs(white_word_freqs)
   print()
   
def get_words(review, method, stopwords=None):
   if method == 'bad':
      words = review.split()
   else:
      if stopwords is None:
         print("***> ERROR: Cannot use good word finding method without stopwords")
         exit()

      # get case insensitive words
      words = review.lower().split()

      # remove stopwords
      words = [w for w in words if w not in stopwords]

   return words
#
# end of get_words

def get_word_freqs(reviews, method='bad', stopwords=None):
   
   word_freqs = {}
   for review in reviews:
      words = get_words(review, method, stopwords)
      for word in words:
         if word not in word_freqs:
            word_freqs[word] = 1
         else:
            word_freqs[word] += 1
   return word_freqs
#
# end of get_word_freqs

def get_top_freqs(word_freqs, n=10):
   words, freqs = zip(*sorted(word_freqs.items(), key=lambda x: (-x[1], x[0])))
   for i in range(n):
      print('%s\t%s' % (words[i], freqs[i]))
#
# end of get_top_freqs
