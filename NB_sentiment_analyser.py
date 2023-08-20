# -*- coding: utf-8 -*-
"""
NB sentiment analyser. 

Start code.
"""
import argparse
import pandas as pd
import numpy as np
from nltk.corpus import wordnet as wn
import matplotlib.pyplot as plt
import seaborn as sns


USER_ID = "test" 

def parse_args():
    parser=argparse.ArgumentParser(description="A Naive Bayes Sentiment Analyser for the Rotten Tomatoes Movie Reviews dataset")
    parser.add_argument("training")
    parser.add_argument("dev")
    parser.add_argument("test")
    parser.add_argument("-classes", type=int)
    parser.add_argument('-features', type=str, default="all_words", choices=["all_words", "features"])
    parser.add_argument('-output_files', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('-confusion_matrix', action=argparse.BooleanOptionalAction, default=False)
    args=parser.parse_args()
    return args

def value_scale(original_label):
    if original_label == 0 or original_label == 1:
        new_label = 0
    elif original_label == 2:
        new_label = 2
    elif original_label == 3 or original_label == 4:
        new_label = 4
    else:
        print('label err occurs')
    return new_label

def get_prior_prob(df, class_num, total_count):
    #count for each class
    neg_sentence = df[df["Sentiment"] == 0]
    neu_sentence = df[df["Sentiment"] == 2]
    pos_sentence = df[df["Sentiment"] == 4]
    #calculate prior probability
    prior_neg = neg_sentence.shape[0]/total_count
    prior_neu = neu_sentence.shape[0]/total_count
    prior_pos = pos_sentence.shape[0]/total_count
    if class_num == 3:
        return [prior_neg, prior_neu, prior_pos]

    if class_num == 5:
        sw_neg_sentence = df[df["Sentiment"] == 1]
        sw_pos_sentence = df[df["Sentiment"] == 3]
        prior_sw_neg = sw_neg_sentence.shape[0]/total_count
        prior_sw_pos = sw_pos_sentence.shape[0]/total_count
        return [prior_neg, prior_sw_neg, prior_neu, prior_sw_pos ,prior_pos]   

def sentiment_count(word_dict, word):
    if word in word_dict:
        word_dict[word] += 1
    else:
        word_dict[word] = 1

def posterior_prob(prior_prob, word_dict, phrase, features):
    words = phrase.strip().split()
    if features == 'features':
        words = extract_feature(words)
    total_words = sum(word_dict.values())
    prob = prior_prob
    for word in words:
        if word in word_dict:
            prob = prob*(word_dict[word]+1/total_words + len(word_dict)) #Likelihood applied Laplace smoothing
    return prob

def get_f1(cm, sentiment):
        tp = cm[sentiment][sentiment]
        fn_cm = cm.drop([sentiment], axis=0)
        fn = fn_cm.iloc[:, sentiment].sum()
        fp_cm = cm.drop([sentiment], axis=1)
        fp = fp_cm.iloc[sentiment, :].sum()
        f1 = 2*tp/(2*tp+fp+fn)
        return f1

def extract_feature(words):
    features = []
    for word in words:
        if wn.synsets(word):
            pos = wn.synsets(word)[0].pos()
            if (pos == 'a') | (pos == 'n'):
                features.append(word)
    return features


def main():
    
    inputs=parse_args()
    
    #input files
    training = inputs.training
    dev = inputs.dev
    test = inputs.test
    
    #number of classes
    number_classes = inputs.classes
    
    #accepted values "features" to use your features or "all_words" to use all words (default = all_words)
    features = inputs.features
    
    #whether to save the predictions for dev and test on files (default = no files)
    output_files = inputs.output_files
     
    
    #whether to print confusion matrix (default = no confusion matrix)
    confusion_matrix = inputs.confusion_matrix
    
    #tsv to df
    train_df = pd.read_csv(training, sep='\t')
    dev_df = pd.read_csv(dev, sep='\t')
    test_df = pd.read_csv(test, sep='\t')

    #prepocessing
    #map 5 values to 3 values
    if number_classes == 3:
        train_df['Sentiment'] = train_df['Sentiment'].map(value_scale)

    #get prior probability
    #total count
    total_count = train_df.shape[0] #6529
    
    if number_classes == 3:
        prior_probs = get_prior_prob(train_df, 3, total_count)
    elif number_classes == 5:
        prior_probs = get_prior_prob(train_df, 5, total_count)

 
    #get likelihood
    neg_words = {}
    neu_words = {}
    pos_words = {}
    sw_neg_words={}
    sw_pos_words={}
    
    #count word frequency in the specific sentimental sentences
    for id, phrase, sentiment in train_df.itertuples(index=False):
        words = phrase.strip().split()
        #if features are used, it only use adjectives and nouns
        if features=='features':
            words = extract_feature(words)

        #sentiment_count(word_dict, word)
        for word in words:
            if sentiment == 0:
                sentiment_count(neg_words, word)
            elif sentiment == 2:
                sentiment_count(neu_words, word)
            elif sentiment == 4:
                sentiment_count(pos_words, word)
            elif (sentiment == 1) & (number_classes == 5):
                sentiment_count(sw_neg_words, word)
            elif (sentiment == 3) & (number_classes == 5):
                sentiment_count(sw_pos_words, word)
            else:
                print(sentiment)
                print('err to get likelihood')
    
    if number_classes == 3:
        word_dicts = [neg_words, neu_words, pos_words]
    elif number_classes == 5:
        word_dicts = [neg_words, sw_neg_words, neu_words, sw_pos_words, pos_words]
    
    #classify the sentence
    #create confusion matrix
    cm = np.zeros((5,5))
    dev_predictions = []
    for id, phrase, sentiment in dev_df.itertuples(index=False):
        posterior_probs = []
        for prior_prob, word_dict in zip(prior_probs, word_dicts):
            post_prob = posterior_prob(prior_prob, word_dict, phrase, features)
            posterior_probs.append(post_prob)
        sentiment_pred = np.argmax(posterior_probs)
        cm[sentiment][sentiment_pred] += 1
        dev_predictions.append([id, sentiment_pred])
    
    #delete empty rows and cols in confusion matrix
    if number_classes == 3:
        cm = np.delete(cm, [3,4], 1)
        cm = np.delete(cm, [1,3], 0)

    #macro F-1 score
    cm = pd.DataFrame(cm)
    f1_list = []
    #get f1 score for each class
    for label in range(number_classes):
        f1 = get_f1(cm, label)
        f1_list.append(f1)
    #get macro F-1 score
    macro_f1 = np.mean(f1_list)


    #test the model
    test_predictions = []
    for id, phrase in test_df.itertuples(index=False):
        posterior_probs = []
        for prior_prob, word_dict in zip(prior_probs, word_dicts):
            post_prob = posterior_prob(prior_prob, word_dict, phrase, features)
            posterior_probs.append(post_prob)
        sentiment_pred = np.argmax(posterior_probs)
        test_predictions.append([id, sentiment_pred]) 

    #output file
    if output_files:
        with open('outputfiles/dev_predictions_acd19sb.tsv', 'w') as out:
            for (id, sentiment) in dev_predictions:
                print(id, sentiment, file=out)
        with open('outputfiles/test_predictions_acd19sb.tsv', 'w') as out:
            for (id, sentiment) in test_predictions:
                print(id, sentiment, file=out)

    #present confusion matrix 
    if confusion_matrix:
        ax = sns.heatmap(cm, annot=True, fmt='.2f', cbar=False, cmap='Blues')
        plt.title('Confusion Matrix(3_classes_features)')
        plt.xlabel('Predicted label')
        plt.ylabel('Actual label')
        plt.show()
    
    #return macro-F1 score for the dev set
    f1_score = macro_f1


    #print("User\tNumber of classes\tFeatures\tmacro-F1(dev)\tAccuracy(dev)")
    print("%s\t%d\t%s\t%f" % (USER_ID, number_classes, features, f1_score))

if __name__ == "__main__":
    main()