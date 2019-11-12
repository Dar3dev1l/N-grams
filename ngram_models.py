import nltk  
from nltk.corpus import udhr  
from nltk.probability import FreqDist
import re
from math import *
from fractions import Fraction

# Generate n-grams from given string. Pass string and n as parameters
def generate_ngram(string, n):
    string = string.lower()                                        #Convert to lower case.
    string = re.sub(r'^#|#$','',string)                            #Remove existing '#' characters. We will use # only as start and end char.
    string = re.sub(r'[^A-za-z0-9]+',' ',string)                   #Remove all non-alphanumeric characters like ', / . etc.'
    string = '#'+string+'#'                                        #Add '#' as start and end character of string
    ngrams = [string[index:index+n] for index in range(len(string)-(n-1))]     #Split string into n-character parts. len(string)-(n-1) is the max range value as starting from 0.
    return ngrams,string
    
#UnigramModel Class. Method calc_unigram_probability() to calculate and store probabilities in dictionary
class UnigramModel:
    def __init__(self,text):
        self.unigrams,self.string = generate_ngram(text,1)                 #Initialize list of unigrams by calling generate ngram method with n=1
        self.unigram_probabilities = {}                                    #Dictionary to store unigram and corresponding probability
        self.vocab_size = 0;
        self.unigram_dist = FreqDist()

    def calc_unigram_probability(self,unigrams):                            
        self.unigram_dist = FreqDist(unigrams)                              #Create Unigram distribution using FreqDist() function
        total_count =  len(unigrams)                                        #Total count of unigrams in training set
        self.vocab_size = len(set(unigrams))                                #Count of distinct unigrams (characters)
        for unigram in set(unigrams):           
            unigram_count = self.unigram_dist.get(unigram)                  #get count of occurrence for particular unigram
            
            #Add one to numerator and V to denominator for Add-one smoothing
            self.unigram_probabilities.update({unigram:Fraction(unigram_count+1,total_count+self.vocab_size)})    #Create dictionary of unigram and its corresponding probability
        
        
        
#BigramModel Class. Method calc_bigram_probability() to calculate and store probabilities in a 2 level dictionary
class BigramModel:
    def __init__(self,text):
        self.bigrams,self.string = generate_ngram(text,2)                          #Initialize list of bigrams by calling generate ngram method with n = 2
        self.bigram_probabilities = {}                                             #Dictionary to store bigram and corresponding probability
        self.vocab_size = 0;
        self.unigram_dist = FreqDist()
    def calc_bigram_probability(self,string,bigrams):
        bigram_dist = FreqDist(bigrams)                                            #Create Bigram distribution using FreqDist() function
        unigrams,string = generate_ngram(string, 1)                                  
        self.unigram_dist = FreqDist(unigrams)                                      #Create Unigram distribution using FreqDist() function 
        self.vocab_size = len(set(unigrams))                                         #vocab size is the toal distinct characters in dataset
        for bigram in set(bigrams):
            bigram_count = bigram_dist.get(bigram)                                   #get count of bigram 
            unigram_count = self.unigram_dist.get(bigram[:-1])                       #get count of unigram
            probability = Fraction(bigram_count+1,unigram_count+self.vocab_size)     #Calculate probability with add-one smoothing
            temp = {bigram[-1]:probability}                                          #Storing in temp dictionary - last char in bigram and probability
            if bigram[0] in self.bigram_probabilities:
                self.bigram_probabilities[bigram[0]].update(temp)                    #update main dictionary with temp- {bigram[0]: {bigram[1]:probability}}
            else:
                self.bigram_probabilities.setdefault(bigram[0],{})                   #If encountered a character first time, add it to main dictionary as a dict object
                self.bigram_probabilities[bigram[0]].update(temp)
                
                
                
#TrigramModel Class. Method calc_trigram_probability() to calculate and store probabilities in a 3 level dictionary
class TrigramModel:
    def __init__(self,text):
        self.trigrams,self.string = generate_ngram(text, 3)                      #Initialize list of trigrams by calling generate ngram method
                                                                                 #self.string will be used to generate bigram and unigram distributions for later use
        self.trigram_probabilities = {}                                          #Dictionary to store trigram and corresponding probability
        self.vocab_size = 0; 
        self.bigram_dist = FreqDist()
        
    def calc_trigram_probability(self,string,trigrams):
        trigram_dist = FreqDist(trigrams)                                         #Create Trigram distribution using FreqDist() function
        bigrams,string = generate_ngram(string, 2)
        self.bigram_dist = FreqDist(bigrams)                                      #Create Bigram distribution using FreqDist() function
        unigrams,string = generate_ngram(string, 1)
        unigram_dist = FreqDist(unigrams)                                        #Create Unigram distribution using FreqDist() function to calculate vocab size of dataset.
        self.vocab_size = len(set(unigrams))
               
        for trigram in set(trigrams):
            trigram_count = trigram_dist.get(trigram)                            #get count of occurrence of trigram
            bigram_count = self.bigram_dist.get(trigram[:-1])                    #get count of occurrence of preceding bigram
            probability = Fraction(trigram_count+1,bigram_count+self.vocab_size) #Calculate probability with add-one smoothing   
            temp = {trigram[2]:probability}                                      #Store in temp ditionary
            if trigram[0] in self.trigram_probabilities:
                if trigram[1] in self.trigram_probabilities.get(trigram[0]):
                    self.trigram_probabilities.get(trigram[0])[trigram[1]].update(temp)   #Storing in 3 level dictionary as -  
                else:                                                                     #{trigram[0]:{trigram[1]:{trigram[2]:probability}}}
                    self.trigram_probabilities.get(trigram[0]).setdefault(trigram[1],{})
                    self.trigram_probabilities.get(trigram[0])[trigram[1]].update(temp)
            else:
                self.trigram_probabilities.setdefault(trigram[0],{}) 
                self.trigram_probabilities.get(trigram[0]).setdefault(trigram[1],{})
                self.trigram_probabilities.get(trigram[0])[trigram[1]].update(temp)
