# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 14:45:12 2019

@author: ntorba605
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 18:15:21 2019

@author: ntorba605
"""
import numpy as np 

class Naive_Bayes: 
    
    def __init__(self, alpha=None):
        self.alpha=alpha
        self.classifier_data = None
        
        
    #Train data is the corpus with labels
    # alpha range is a list of ints to choose alpha from through validation
    # plot_validation=True if you want a plot of validations
    #return classifier_dict data_struct and sets as a class attribute 
    def train_nb(self, train_data, alpha_range=None, plot_validation=False):
        """
        Build the classifier_data structure to hold counts for each word in each type of review 
        word_dict_stucture:
        'word': [negaive_count, positive_count]
        Probabilities are calculated later with these counts 
        """
        if self.alpha is None:
            self.validate__(train_data, plot_validation, alpha_range=alpha_range)
            print(f'validated alpha_val = {self.alpha}')
        
        word_dict = {}
        for label, sentence in train_data:
            dict_index = 0
            if label == 'pos':
                dict_index = 1
            for word in sentence:
                if word in word_dict:
                    word_dict[word][dict_index] += 1 
                elif dict_index == 1:
                    # word_dict[word] = [0,1]
                    word_dict[word] = [self.alpha,self.alpha+1]
                else:
                    # word_dict[word] = [1,0]
                    word_dict[word] = [self.alpha+1,self.alpha]
                    
        self.classifier_data = word_dict
        return word_dict
    
    # use self.classifier_dict to classify the document
    # my classifier_dict holds counts, not probabilities, so the 
    # probabilities are calculated here 
    def classify_nb(self, document):
        neg_prob = 1
        pos_prob = 1
        # sentence = document[1]
        # doc_half_len = int(len(document)/2)
        for word in document:
            if word not in self.classifier_data:
                pass
            else: 
                i_pos_prob = self.classifier_data[word][1]/(self.classifier_data[word][0]+self.classifier_data[word][1])
                i_neg_prob = self.classifier_data[word][0]/(self.classifier_data[word][0]+self.classifier_data[word][1])
                pos_prob *= i_pos_prob
                neg_prob *= i_neg_prob
#                if i_pos_prob != 0:
#                    pos_prob+=np.log(i_pos_prob)
#                if i_neg_prob !=0:
#                    neg_prob+=np.log(i_neg_prob)
        if neg_prob > pos_prob:
            return 'neg'
        else:
            return 'pos'
    
    def evaluate_nb(self, test_data):
        test_labels = [label for label, document in test_data]
        test_labels_predicted = []
        correct_count = 0
        
        for label, document in test_data:
            predicted_label = self.classify_nb(document)
            test_labels_predicted.append(predicted_label)
        
        index=0
        # index, correct, predicted 
        incorrect_indices = []    
        for correct, predicted in zip(test_labels, test_labels_predicted):
            if correct == predicted:
                correct_count+=1
            else:
                incorrect_indices.append((index, correct, predicted))
            index+=1
        return correct_count/len(test_labels), incorrect_indices
    
    # print_top_nb prints the top-N features that are most indicative of the positive and the negative categories.
    def print_top_nb(self, n_highest):
        
        neg_sorted = sorted([(k,(v[0]/(v[0]+v[1]))) for k,v in self.classifier_data.items()], key=lambda x: x[1], reverse=True)
        pos_sorted = sorted([(k,v[1]/(v[0]+v[1])) for k,v in self.classifier_data.items()], key=lambda x: x[1], reverse=True)
        
        print(f'Top {n_highest} indicative words of postive reviews:')
        for i in range(n_highest):
            print(f'{i+1}. "{pos_sorted[i][0]}" with probability {pos_sorted[i][1]}')
        print('---')
        print(f'Top {n_highest} indicative words of negative reviews: ')
        for i in range(n_highest):
            print(f'{i+1}. "{neg_sorted[i][0]}" with probability {neg_sorted[i][1]}')
        
        return neg_sorted, pos_sorted
    
    #train_data is text corpus with labels 
    # plot_validation is boolean, True if want a plot 
    # if alpha_range is not specified is defaults to 0->19 
    def validate__(self, train_data, plot_validation, alpha_range=None):
        # print('validate called!')
        if self.alpha != None:
            raise Exception('the models alpha value is already defined, if you wish to change it, set it to none')
        
        if alpha_range is None:
            alpha_range = [i for i in range(20)]

        val_alpha_accs = []
        for alpha in alpha_range:
            val_train = train_data[int(len(train_data)*0.2):]
            val_test = train_data[:int(len(train_data)*0.2)]
            
            self.alpha = alpha
            classifier_data = self.train_nb(val_train)
            self.classifier_data = classifier_data
            acc, incorrect_indices = self.validation_eval__(classifier_data, val_test)
            val_alpha_accs.append((alpha, acc))
        
        if plot_validation: 
            import matplotlib.pyplot as plt 
            plt.bar(*zip(*val_alpha_accs))
            plt.ylabel('validation accuracy')
            plt.xlabel('alpha_val')
            plt.title('validation accuracy for alpha_val selection')
            plt.xticks(np.arange(0,20,1.0))
            plt.savefig('alpha_validation_accuracies.png')
            plt.show()
    
        # print(val_alpha_accs)
        self.alpha = max(val_alpha_accs, key=lambda x: x[1])[0]    
        return self.alpha
    
    def validation_eval__(self, classifier_data, test_data):
        test_labels = [label for label, document in test_data]
        test_labels_predicted = []
        correct_count = 0
        
        for label, document in test_data:
            predicted_label = self.classify_nb(document)
            test_labels_predicted.append(predicted_label)
        
        index=0
        # index, correct, predicted 
        incorrect_indices = []    
        for correct, predicted in zip(test_labels, test_labels_predicted):
            if correct == predicted:
                correct_count+=1
            else:
                incorrect_indices.append((index, correct, predicted))
            index+=1
        return correct_count/len(test_labels), incorrect_indices
