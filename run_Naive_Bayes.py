# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:57:31 2019

@author: ntorba605
"""

import Naive_Bayes as nb

def read_corpus(corpus_file):
     out = []
     with open(corpus_file,encoding='utf') as f:
          for line in f:
              tokens = line.strip().split()
              out.append( (tokens[1], tokens[3:]) )
     return out
 
if __name__ == '__main__':
    data = read_corpus('all_sentiment_shuffled.txt')
    num_train = int(len(data)*0.8)
    train_data = data[:num_train]
    test_data = data[num_train:]
    
    # alpha can be set in the model initialization, if done, no validation will be performed
    model = nb.Naive_Bayes(alpha=None)
    
    # alpha_range defaults to 0->19 if not set, make plot_validation=True if want to see validation plots 
    model.train_nb(train_data, alpha_range=list(range(20)), plot_validation=False)
    
    #evaluate_nb returns the accuracy along with the incorrect_indices to allow for quick evaluation
    acc, inccorect_indices = model.evaluate_nb(test_data)
    print(f'test accuracy = {acc}')
    negative_sorted, positive_sorted = model.print_top_nb(10)