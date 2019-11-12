import nltk  
from nltk.corpus import udhr  
from nltk.probability import FreqDist
import re
from math import *
from fractions import Fraction
from ngram_models import *

#Pass word, model type and training dataset to find probability for word
def calc_word_probability(word,model,training_set):
    word = word.lower()
    probability = 1                             #Initializing probability to 1
    base = 10                                   #Initialize base as 10 to calculate log of probability to the base 10
    if model.lower()=="unigram":
        unigram_obj = UnigramModel(training_set)           
        unigram_obj.calc_unigram_probability(unigram_obj.unigrams)         #Create unigram model. Store in dictionary
        ngrams = [word[index:index+1] for index in range(len(word))]       #Split word into unigrams
        for ngram in ngrams:                                               #for every unigram in word retreive its probability and multiply
            if (unigram_obj.unigram_probabilities.get(ngram))==None:
                #probability for unseen unigram
                log_probability = log(probability,base) + log(Fraction(1,len(unigram_obj.unigrams)+unigram_obj.vocab_size),base) #using addition of logs for multiplication
            else:
                log_probability = log(probability,base) + log(unigram_obj.unigram_probabilities.get(ngram),base)
    elif model.lower()=="bigram":
        bigram_obj = BigramModel(training_set)
        bigram_obj.calc_bigram_probability(bigram_obj.string,bigram_obj.bigrams)    #Create bigram model
       
        # Add spaces to words.Space will be useful in bigram and trigrams as it will tell us which chracters have a higher probability of occuring at the start or end of a word
        if word[0]=="#":         
            word = word+" "
        elif word[-1]=="#":
            word = " "+word
        else:
            word = " "+word+" "
        ngrams = [word[index:index+2] for index in range(len(word)-1)]           #Split word into bigrams
        for ngram in ngrams:
            if (bigram_obj.bigram_probabilities.get(ngram[0]))==None: 
                #probability for unseen bigram[0]
                log_probability = log(probability,base) + log(Fraction(1,bigram_obj.vocab_size),base)
            elif bigram_obj.bigram_probabilities.get(ngram[0]).get(ngram[1])==None:
                #probability for unseen bigram[1]
                log_probability = log(probability,base) + log(Fraction(1,bigram_obj.unigram_dist.get(ngram[0])+bigram_obj.vocab_size),base)
            else:
                log_probability = log(probability,base) + log(bigram_obj.bigram_probabilities.get(ngram[0]).get(ngram[1]),base)
    elif model.lower()=="trigram":
        trigram_obj = TrigramModel(training_set)
        trigram_obj.calc_trigram_probability(trigram_obj.string,trigram_obj.trigrams)
        if word[0]=="#":
            word = word+" "
        elif word[-1]=="#":
            word = " "+word
        else:
            word = " "+word+" "
        ngrams = [word[index:index+3] for index in range(len(word)-2)]                   #Split word into trigrams
        for ngram in ngrams:
            if (trigram_obj.trigram_probabilities.get(ngram[0]))==None:   
                #probability for unseen trigram[0]
                log_probability = log(probability,base) + log(Fraction(1,trigram_obj.vocab_size),base)
            elif trigram_obj.trigram_probabilities.get(ngram[0]).get(ngram[1])==None:
                #probability for unseen trigram[1]
                log_probability = log(probability,base) + log(Fraction(1,trigram_obj.vocab_size),base)
            elif trigram_obj.trigram_probabilities.get(ngram[0]).get(ngram[1]).get(ngram[2])==None:
                #probability for unseen trigram[2]
                log_probability = log(probability,base) + log(Fraction(1,trigram_obj.bigram_dist.get(ngram[:2])+trigram_obj.vocab_size),base)
            else:
                log_probability = log(probability,base) + log(trigram_obj.trigram_probabilities.get(ngram[0]).get(ngram[1]).get(ngram[2]),base)
    probability = round(base**log_probability,5)    #Convert logarithm back to decimal format and round off to precision 5
    return probability
    
#Compare two language models with language1 test set and return the accuracy of language1 model.
def compare_model(language1_test, language1_train, language2_train, model):
    language1 = []
    language2 = []
    language1_test = list(language1_test)                                       #Convert to list.
    #Add '#' as start and end character
    language1_test[0] = "#"+language1_test[0]
    language1_test[-1] = language1_test[0]+"#"
    for word in language1_test:
        #Call calc_word_probability function for each word in langauge1 test set
        language1.append(calc_word_probability(word,model,language1_train))  
        language2.append(calc_word_probability(word,model,language2_train))
    count = 0
    for i in range(len(language1_test)-1):
        if language1[i] > language2[i]:                                         #If word has higher probability for language1 increase count.
            count += 1
    print("Number of words correctly identified as language 1: ",count)
    print("Total number of words in Test set: ",len(language1_test)) 
    accuracy = (count*100)/len(language1_test)                                  #Correctly identfied words*100/total words
    print("Accuracy of ",model," Model: ",accuracy,"%")
    return accuracy 

#Compare English vs French Language Models
def english_vs_french():
    print("English (Language1) vs French (Language2) Comparison Model- \n")
    uni_accuracy = compare_model(english_test, english_train, french_train, "unigram")
    bi_accuracy = compare_model(english_test, english_train, french_train, "bigram")
    tri_accuracy = compare_model(english_test, english_train, french_train, "trigram")
    
    #Average of unigram, bigram an trigram model accuracies
    print("Average Accuracy of the 3 models: ",(uni_accuracy+bi_accuracy+tri_accuracy)/3)
    
#Compare Spanish vs Italian Language Models
def spanish_vs_italian():
    print("\nSpanish (Language1) vs Italian (Language2) Comparison Model - \n")
    uni_accuracy = compare_model(spanish_test , spanish_train, italian_train, "unigram")
    bi_accuracy = compare_model(spanish_test , spanish_train, italian_train, "bigram")
    tri_accuracy = compare_model(spanish_test , spanish_train, italian_train, "trigram")
    
    #Average of unigram, bigram an trigram model accuracies
    print("Average Accuracy of the 3 models: ",(uni_accuracy+bi_accuracy+tri_accuracy)/3,"%")

if __name__ == "__main__":

    #Loading my Dataset. Using Universal Declaration of Human Rights from nltk.corpus
    english = udhr.raw('English-Latin1')
    french = udhr.raw('French_Francais-Latin1')
    italian = udhr.raw('Italian_Italiano-Latin1')
    spanish = udhr.raw('Spanish_Espanol-Latin1')

    #Splitting dataset into train, dev and Test
    english_train, english_dev = english[0:1000], english[1000:1100]  
    french_train, french_dev = french[0:1000], french[1000:1100] 
    italian_train, italian_dev = italian[0:1000], italian[1000:1100]  
    spanish_train, spanish_dev = spanish[0:1000], spanish[1000:1100]   

    english_test = udhr.words('English-Latin1')[0:1000]  
    french_test = udhr.words('French_Francais-Latin1')[0:1000] 
    italian_test = udhr.words('Italian_Italiano-Latin1')[0:1000]
    spanish_test = udhr.words('Spanish_Espanol-Latin1')[0:1000]

    english_vs_french()

    spanish_vs_italian()
