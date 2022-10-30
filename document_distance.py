# Purpose: Check for similarity between two texts by comparing different kinds of word statistics.

import string
import math


### DO NOT MODIFY THIS FUNCTION
def load_file(filename):
    """
    Args:
        filename: string, name of file to read
    Returns:
        string, contains file contents
    """
    # print("Loading file %s" % filename)
    inFile = open(filename, 'r')
    line = inFile.read().strip()
    for char in string.punctuation:
        line = line.replace(char, "")
    inFile.close()
    return line.lower()


### Prep Data ###
def text_to_list(input_text):
    """
    Args:
        input_text: string representation of text from file.
                    assume the string is made of lowercase characters
    Returns:
        list representation of input_text, where each word is a different element in the list
    """
    input_list = input_text.split()
    return input_list


### Get Frequency ###
def get_frequencies(input_iterable):
    """
    Args:
        input_iterable: a string or a list of strings, all are made of lowercase characters
    Returns:
        dictionary that maps string:int where each string
        is a letter or word in input_iterable and the corresponding int
        is the frequency of the letter or word in input_iterable
    Note: 
        You can assume that the only kinds of white space in the text documents we provide will be new lines or space(s) between words (i.e. there are no tabs)
    """
    freq_dict = {}
    unique_elements = set(input_iterable)
    for n in unique_elements:
        freq_dict[n] = input_iterable.count(n)
    return freq_dict


### Letter Frequencies ###
def get_letter_frequencies(word):
    """
    Args:
        word: word as a string
    Returns:
        dictionary that maps string:int where each string
        is a letter in word and the corresponding int
        is the frequency of the letter in word
    """
    letter_dict = {}
    letter_list = list(word)
    unique_letters = set(word)
    for letter in unique_letters:
        letter_dict[letter] = letter_list.count(letter)
        
    return letter_dict


### Similarity ###
def calculate_similarity_score(freq_dict1, freq_dict2):
    """
    The keys of dict1 and dict2 are all lowercase,
    you will NOT need to worry about case sensitivity.

    Args:
        freq_dict1: frequency dictionary of letters of word1 or words of text1
        freq_dict2: frequency dictionary of letters of word2 or words of text2
    Returns:
        float, a number between 0 and 1, inclusive
        representing how similar the words/texts are to each other

        The difference in words/text frequencies = DIFF sums words
        from these three scenarios:
        * If an element occurs in dict1 and dict2 then
          get the difference in frequencies
        * If an element occurs only in dict1 then take the
          frequency from dict1
        * If an element occurs only in dict2 then take the
          frequency from dict2
         The total frequencies = ALL is calculated by summing
         all frequencies in both dict1 and dict2.
        Return 1-(DIFF/ALL) rounded to 2 decimal places
    """
    diff = 0
    total = 0
    for key in freq_dict1:
        if key in freq_dict2:
            diff += (abs(freq_dict1[key] - freq_dict2[key]))
            total += freq_dict1[key] + freq_dict2[key]
        else:
            diff += freq_dict1[key] 
            total += freq_dict1[key]
    for key in freq_dict2:
        if key not in freq_dict1:
            diff += freq_dict2[key]
            total += freq_dict2[key]      
    return round(1 - (diff/total), 2)


### Most Frequent Word(s) ###
def get_most_frequent_words(freq_dict1, freq_dict2):
    """
    The keys of dict1 and dict2 are all lowercase,
    you will NOT need to worry about case sensitivity.

    Args:
        freq_dict1: frequency dictionary for one text
        freq_dict2: frequency dictionary for another text
    Returns:
        list of the most frequent word(s) in the input dictionaries

    The most frequent word:
        * is based on the combined word frequencies across both dictionaries.
          If a word occurs in both dictionaries, consider the sum the
          freqencies as the combined word frequency.
        * need not be in both dictionaries, i.e it can be exclusively in
          dict1, dict2, or shared by dict1 and dict2.
    If multiple words are tied (i.e. share the same highest frequency),
    return an alphabetically ordered list of all these words.
    """
    all_keys = set(list(freq_dict1.keys()) + list(freq_dict2.keys()))
    combined_dict = {}
    for key in all_keys:
        key_freq = 0
        if key in freq_dict1:
            key_freq += freq_dict1[key]
        if key in freq_dict2:
            key_freq += freq_dict2[key]
        combined_dict[key] = key_freq
    
    values_list = combined_dict.values()
    highest_freq = max(values_list)
    highest_freq_words = []
    
    for key in combined_dict:
        if combined_dict[key] == highest_freq:
            highest_freq_words.append(key)
        
    return sorted(highest_freq_words)
    


### Finding TF-IDF ###
def get_tf(file_path):
    """
    Args:
        file_path: name of file in the form of a string
    Returns:
        a dictionary mapping each word to its TF

    * TF is calculatd as TF(i) = (number times word *i* appears
        in the document) / (total number of words in the document)
    * Think about how we can use get_frequencies from earlier
    """
    text = load_file(file_path)
    text_list = text_to_list(text)
    text_dict = get_frequencies(text_list)
    
    tf_dict = {}
    for key in text_dict:
        tf = text_dict[key]/len(text_list)
        tf_dict.update({key: tf})
    return tf_dict
    

def get_idf(file_paths):
    """
    Args:
        file_paths: list of names of files, where each file name is a string
    Returns:
       a dictionary mapping each word to its IDF

    * IDF is calculated as IDF(i) = log_10(total number of documents / number of
    documents with word *i* in it), where log_10 is log base 10 and can be called
    with math.log10()

    """
    all_words = []
    idf_dict = {}
    for file in file_paths:
        text_list = text_to_list(load_file(file))
        all_words += text_list
    
    for word in set(all_words):
        count = 0
        for file in file_paths:
            if word in text_to_list(load_file(file)):
                count += 1
        idf_dict.update({word: (math.log10(len(file_paths)/count))})
    return idf_dict
       

def get_tfidf(tf_file_path, idf_file_paths):
    """
        Args:
            tf_file_path: name of file in the form of a string (used to calculate TF)
            idf_file_paths: list of names of files, where each file name is a string
            (used to calculate IDF)
        Returns:
           a sorted list of tuples (in increasing TF-IDF score), where each tuple is
           of the form (word, TF-IDF). In case of words with the same TF-IDF, the
           words should be sorted in increasing alphabetical order.

        * TF-IDF(i) = TF(i) * IDF(i)
        """
    tf_dict = get_tf(tf_file_path)
    idf_dict = get_idf(idf_file_paths)
    tfidf_dict = {}
    for word in tf_dict:
        tfidf_dict.update({word: (tf_dict[word] * idf_dict[word])})
    sorted_tfidf_dict = sorted(tfidf_dict.items(), key=lambda x: x[1])   
    return sorted_tfidf_dict
    

