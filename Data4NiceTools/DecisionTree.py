#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import collections
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import Imputer
from sklearn import tree

field_list = [
'FVMARSTAT',
'ED',
'PATHISP',
'FVJOBLOC__4',
'PREVVAGINF',
'MR1SPRCADQ',
'PERFUMEFF',
'LIQSOAPFF',
'CLEANER',
'MEATFAT',
'CHICKFAT',
'FISH',
'COLDCUTSFAT'] 


def Run(raw_data_file):
  data = pd.read_csv(raw_data_file)
  raw_label = data['PPTERM'].tolist()
  data.drop('PPTERM', axis=1, inplace=True)
  results = collections.OrderedDict()
  #num_features = len(data.columns)
  num_samples = len(data.index)
  num_features = len(field_list)
  samples = []
  labels = []
  for i in data.index:
    sample = []
    #for column in data.columns:
    for column in field_list:
      #if pd.isnull(data[column][i]):
      #  sample = []
      #  break
      sample.append(data[column][i])
    if len(sample) > 0:
      samples += sample
      labels.append(raw_label[i])
  num_samples_final = len(samples) / num_features 
  print "Number of samples: " + str(num_samples_final)
  print "Number of features: " + str(num_features)
  x = np.array(samples)
  x = np.reshape(x, (num_samples_final, num_features))
  y = np.array(labels)
  num_test_set = num_samples_final - 50
  print x

  imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
  imp.fit(x)
  imputed_x = imp.transform(x)
  clf = tree.DecisionTreeClassifier()
  #clf = clf.fit(x[:num_test_set], y[:num_test_set])
  clf = clf.fit(imputed_x[:num_test_set], y[:num_test_set])
  #result = clf.predict(x[num_test_set:])
  result = clf.predict(imputed_x[num_test_set:])
  ref = y[num_test_set:]
  count = 0
  false_negative_count = 0
  false_positive_count = 0
  preterm_count = 0
  term_count = 0
  for i in range(len(result)):
    if result[i] == ref[i]:
      count += 1
    if ref[i] == 2:
      preterm_count += 1
    if ref[i] == 1:
      term_count += 1
    # False negative
    if ref[i] == 2 and result[i] == 1:
      false_negative_count += 1
    # False positive
    if ref[i] == 1 and result[i] == 2:
      false_positive_count += 1

  accuracy = float(count) / float(len(result))
  false_negative_rate = float(false_negative_count) / float(preterm_count)
  false_positive_rate = float(false_positive_count) / float(term_count)
      
  print result
  print ref
  print "Accuracy: " + str(accuracy)
  print "False negative rate: " + str(false_negative_rate)
  print "False positive rate: " + str(false_positive_rate)

  tree.export_graphviz(clf,
                       #feature_names=data.columns,  
                       feature_names=field_list,  
                       class_names=['Term', 'Preterm'],
                       filled=True, rounded=True,  
                       special_characters=True,
                       out_file='tree.dot')

def main():
  print ("Start program.")

  if len(sys.argv) < 2:
    print "Too few arguments"
    print "Please specify the raw data file."
    sys.exit()

  filename = sys.argv[1]
  Run(filename)

  print ("End program.")
  return 0

if __name__ == '__main__':
  main()


