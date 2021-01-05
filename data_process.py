#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import re
import string
import nltk 
import pandas as pd 
from nltk.corpus import sentiwordnet as swn 
from pandas.core.frame import DataFrame

'''

In Part A, extract verbs from the concodance. Following rules are obeyed:
1. After automatical extraction, manual extraction is also needed.
2. We manually deleted following elements:
    1) auxilary verbs, like be, have and do
    2) non verbs. The existence of non-verbs should be the mistakes of corpus 
       tools which mistook them as verbs. For example, fast-spreading is an adjective. 
3. For phrasal verbs composed purely of verbs, like "help treat", the whole phrases
   are remained. For phrasal verbs composed of verbs and prepostions/adverbs, 
   like "look at", only the verbs are remained.

'''
# define file names
file_names = ['cleaned_data/2020_01.csv', 'cleaned_data/2020_02.csv', 
              'cleaned_data/2020_03.csv', 'cleaned_data/2020_04.csv',
              'cleaned_data/2020_05.csv', 'cleaned_data/2020_06.csv',
              'cleaned_data/2020_07.csv', 'cleaned_data/2020_08.csv',
              'cleaned_data/2020_09.csv', 'cleaned_data/2020_10.csv']

manual_names = []
for file_name in file_names:
    str_list = list(file_name)
    str_list.insert(13, 'manual/')
    manual_name = ''.join(str_list)
    manual_names.append(manual_name)

final_names = []
for file_name in file_names:
    str_list = list(file_name)
    del str_list[:13]
    str_list.insert(0, 'final_data/')
    final_name = ''.join(str_list)
    final_names.append(final_name)

label_names = []
for file_name in file_names:
    str_list = list(file_name)
    del str_list[:13]
    str_list.insert(0, 'labelled_data/')
    label_name = ''.join(str_list)
    label_names.append(label_name)

# define functions 
def get_verb(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.encode('UTF-8', 'ignore').decode()
    text = text.strip()
    tokens = text.lower().split()
    text = [token for token in tokens if token not in remove_words]
    text = ' '.join(text)
    return text

def get_word_number(verb):
    tokens = verb.split()
    verb = [token for token in tokens if token not in remove_words]
    number = len(verb)
    return number

def get_predicative_verb(file_name):
    original = pd.read_csv(file_name, error_bad_lines = False)
    original['Verb'] = original['Concordance'].apply(get_verb)
    original['Count'] = original['Verb'].apply(get_word_number)

    save = original
    save = save.to_csv(file_name, index = False)
    
    return original, save

def get_pivot(manual_name, final_name):
    manual = pd.read_csv(manual_name, error_bad_lines = False)
    pivot = pd.pivot_table(manual, 
                           index = 'Verb', 
                           values = ['Pmil', 'Frequency'], 
                           aggfunc = 'sum')
    pivot = pivot.reset_index('Verb')
    save = pivot
    save = save.to_csv(final_name)
    
    return pivot, save    

# 202001
remove_words = ['coronavirus', 'COVID-19', 'for', 'with', 'the', 'a', 'by', 
                'with', 'from', 'novel', 'about', 'of', 'against', 'china', 
                'wuhan', 'new', 'number', 'another', 'first', 'amid', 'that', 
                'this', 'deadly', 'to', 'chinese', 'two', 'over', 'out', 'as', 
                'in', 'on', 'if', 'sars', 'associated', 'cov', 'like', 'canine', 
                'suspected', 'paris', 'after', 'company', 'potential', 'more', 
                'british', 'human', 'daily', 'into', 'patented', 'what', 'killer', 
                'student', 'and', 'off', 'how', 'through',  'hubei', 'uk', 'its',
                'airport', 'u', 's', 'five', 'case', 'no', 'any', 'whether', 
                'infectious', 'her', 'you', 'possible', 'past', 'your', 'up', 'not',
                'global', 'australian', 'positive', 'negative', 'some', 'during', 
                'four', 'second', 'first', 'school', 'other', 'one', 'fatal']
clean_01, save_01 = get_predicative_verb(file_names[0])
pivot_01, save_01 = get_pivot(manual_names[0], final_names[0])


# 202002
remove_words = remove_words + ['covid', 'despite', 'while', 'their', 'down']
clean_02, save_02 = get_predicative_verb(file_names[1])
pivot_02, save_02 = get_pivot(manual_names[1], final_names[1])

# 202003
remove_words = remove_words + ['free', 'before', 'fake']
clean_03, save_03 = get_predicative_verb(file_names[2])
pivot_03, save_03 = get_pivot(manual_names[2], final_names[2])

# 202004
remove_words = remove_words + ['at']
clean_04, save_04 = get_predicative_verb(file_names[3])
pivot_04, save_04 = get_pivot(manual_names[3], final_names[3])

# 202005
remove_words = remove_words + ['severe']
clean_05, save_0 = get_predicative_verb(file_names[4])
pivot_05, save_05 = get_pivot(manual_names[4], final_names[4])

# 202006
remove_words = remove_words + ['still', 'all']
clean_06, save_06 = get_predicative_verb(file_names[5])
pivot_06, save_06 = get_pivot(manual_names[5], final_names[5])

# 202007
remove_words = remove_words + ['high', 'federal']
clean_07, save_07 = get_predicative_verb(file_names[6])
pivot_07, save_07 = get_pivot(manual_names[6], final_names[6])

# 202008
remove_words = remove_words + ['active']
clean_08, save_08 = get_predicative_verb(file_names[7])
pivot_08, save_08 = get_pivot(manual_names[7], final_names[7])

# 202009
remove_words = remove_words + ['color']
clean_09, save_09 = get_predicative_verb(file_names[8])
pivot_09, save_09 = get_pivot(manual_names[8], final_names[8])

# 202010
remove_words = remove_words + ['strict', 'his', 'away', 'trump', 'rapid']
clean_10, save_10= get_predicative_verb(file_names[9])
pivot_10, save_10 = get_pivot(manual_names[9], final_names[9])

'''

In Part B, we assign sentiment score to each verb based on sentiwordnet.
1. As a verb can have more than one synset, we use weighted avearge as the score
   for the verb, as is shown in the following formula:
   n = 0, score = 0
                   n                 n
   n > 0, score =  ∑ (score_k / k) / ∑ (1 / k)
                  k=1               k=1
2. For phrasal verb containing more than one verb (often two), use arithmetic 
   mean of each verb's core as the sentiment score of the whole phrasal verb.
3. The final sentiment score of each verb is multiplied with its relative 
   frequency, i.e. the Pmil. The sentiment value of that month is the arithmetic 
   mean of the score of all verbs.

'''   
# single verb score
def get_verb_score(verb):
    senti = list(swn.senti_synsets(verb, 'v'))
    score = 0 
    place = 0
    if len(senti) > 0:
        for j in range(len(senti)):
            score += (senti[j].pos_score() - senti[j].neg_score()) / (j + 1)
            place += 1 / (j + 1) 
    else:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        
        score = 0
        place = 1 # to avoid error caused by 0/0
    
    return score/place

# deal with phrasal verbs
def get_verbs_score(verbs):
    if get_word_number(verbs) > 1:
        verb_tokens = verbs.split()
        score_sum = 0
        for i in range(len(verb_tokens)):
            score_sum += get_verb_score(verb_tokens[i])
        
        verbs_score = score_sum/len(verb_tokens)
    else:
        verbs_score = get_verb_score(verbs)
        
    return verbs_score

def get_mean(num):
    label = pd.read_csv(final_names[num], error_bad_lines = False)
    label['Score'] = label['Verb'].apply(get_verbs_score)
    label.eval('senti_score = Pmil * Score', inplace = True)
    mean = label['senti_score'].sum() / label['Pmil'].sum() 

    save = label
    save = save.to_csv(label_names[num], index = False)
    
    return label, save, mean

# write into file
label_01, save_01, mean_01 = get_mean(0)
label_02, save_02, mean_02 = get_mean(1)
label_03, save_03, mean_03 = get_mean(2)
label_04, save_04, mean_04 = get_mean(3)
label_05, save_05, mean_05 = get_mean(4)
label_06, save_06, mean_06 = get_mean(5)
label_07, save_07, mean_07 = get_mean(6)
label_08, save_08, mean_08 = get_mean(7)
label_09, save_09, mean_09 = get_mean(8)
label_10, save_10, mean_10 = get_mean(9)

mean = [mean_01, mean_02, mean_03, mean_04, mean_05, mean_06, mean_07, mean_08,
        mean_09, mean_10]

# get the first 20 verbs by Pmil
def get_frequent_verb(label):
    label = label.sort_values(by = ['Pmil'], ascending = False)
    frequent = label[:10]
    verb = list(frequent['Verb'] + '(' + 
                frequent['Pmil'].map('{0:.03f}'.format).map(str) + ', ' +
                frequent['Score'].map('{0:.03f}'.format).map(str) + ')')
    return frequent, verb

frequent_01, verb_01 = get_frequent_verb(label_01)
frequent_02, verb_02 = get_frequent_verb(label_02)
frequent_03, verb_03 = get_frequent_verb(label_03)
frequent_04, verb_04 = get_frequent_verb(label_04)
frequent_05, verb_05 = get_frequent_verb(label_05)
frequent_06, verb_06 = get_frequent_verb(label_06)
frequent_07, verb_07 = get_frequent_verb(label_07)
frequent_08, verb_08 = get_frequent_verb(label_08)
frequent_09, verb_09 = get_frequent_verb(label_09)
frequent_10, verb_10 = get_frequent_verb(label_10)

#print(verb_01, '\n', verb_02, '\n', verb_03, '\n', verb_04, '\n', verb_05, '\n', 
#      verb_06, '\n', verb_07, '\n', verb_08, '\n', verb_09, '\n', verb_10)
    
'''
In Part C, we process the COVID-19 data from WHO and create a table with WHO data
and the means of sentiment score.

'''
who_data = pd.read_csv('who_data/WHO_COVID_19_global_data.csv')

def aggregate(df):
    columns = df.columns
    accum_case = dict(df.groupby(columns[0]).sum()[columns[5]])
    accum_death = dict(df.groupby(columns[0]).sum()[columns[-1]])
    dates = ['2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30', '2020-05-31',
             '2020-06-30', '2020-07-31', '2020-08-31', '2020-09-30', '2020-10-31']

    accum_case_month = {}
    accum_death_month = {}
    for date in dates:
        accum_case_month[date] = accum_case[date]
        accum_death_month[date] = accum_death[date]
    
    return accum_case_month, accum_death_month
#
# global data
case_global, death_global = aggregate(who_data)   

# create datafrram from dict
dates = ['2020-01-31', '2020-02-29', '2020-03-31', '2020-04-30', '2020-05-31',
         '2020-06-30', '2020-07-31', '2020-08-31', '2020-09-30', '2020-10-31']

month1 = pd.DataFrame.from_dict(case_global, 
                                orient = 'index', 
                                columns = ['Case_global'])
month2 = pd.DataFrame.from_dict(death_global, 
                                orient = 'index', 
                                columns = ['Death_global'])

month3 = {'Date': dates, 'Mean': mean}
month3 = DataFrame(month3)
month3 = month3.set_index('Date')

month = pd.concat([month1, month2, month3], axis = 1)

death = list(month ['Death_global'])
case = list(month ['Case_global'])
new_death = []
new_case = []
for i in range(len(death)):
    if i == 0:
        new_d = death[i]
        new_c = case[i]
    else:
        new_d = death[i] - death[i-1]
        new_c = case[i] - case[i-1]
    new_death.append(new_d)
    new_case.append(new_c)

month['new_death_global'] = new_death
month['new_case_global'] = new_case

save = month
save = save.to_csv('data_by_month.csv')
       
'''

Reference
Pan, C. (2020). Text Sentiment Analysis wiht Sentiwordnet (A Brief Introduction). 
    Retrieved 10 December 2020, from https://blog.csdn.net/weixin_44592631/
    article/details/104808236        

'''     




