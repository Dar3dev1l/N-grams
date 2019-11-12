# N-grams
Unigram, Bigram and Trigram Character Language model implementation with Laplace (Add-One Smoothing)

This program will train the 3 n-gram models for two languages (say English and French) on a part of the Universal Declaration of Human Rights Corpus from nltk (Pretty small corpus), and test it on another part of same corpus. This is a character model, so while testing, probabilities of words will be calculated by multiplying the character probabilities. Result will be the accuracy of the model correctly identifying English words as english i.e Probability of any english test word should be higher for the English n-gram model compared to another language model ( say French).

Usage:
python main.py

Note:  You can train your own corpus by using only the ngram_models.py file and training it your own way.


N-gram Probability MLE (Without Laplace Smoothing):

P = Probability
C = Count

Unigram-
P(Wi) = C(Wi) / N           where N = No. of total tokens(characters) in corpus

Bigram-
P(Wi|Wi-1) = C(Wi-1, Wi) / C (Wi-1) 
i.e. Count of both tokens occuring together / Count of first token

Trigram-
P(Wi|Wi-1,Wi-2) = C(Wi-2, Wi-1, Wi) / C (Wi-1, Wi-2) 
i.e. Count of all three tokens occuring together / Count of first two tokens appearing together
