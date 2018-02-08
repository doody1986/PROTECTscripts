#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import collections
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import metrics


field_list = ['WTPREPREG', 'FVPREGSTWT', 'FVCURRWT', 'MR1WGHTLBR', 'MR1RBC', 'MR1HCT', \
              'MR1PLTS', 'MR1WBC', 'MR1NEUTRPH', 'MR1LYMPHS', 'MR1EOSINPHS']
#field_list = ['PARTJOBS', 'VOLJOBS']

def Run(raw_data_file):
  data = pd.read_csv(raw_data_file)
  raw_label = data['PPTERM'].tolist()
  data.drop('PPTERM', axis=1, inplace=True)
  results = collections.OrderedDict()
  num_features = len(field_list)
  num_samples = len(data.index)
  samples = []
  labels = []
  for i in data.index:
    sample = []
    for column in field_list:
      if pd.isnull(data[column][i]):
        sample = []
        break
      sample.append(data[column][i])
    if len(sample) > 0:
      samples += sample
      labels.append(raw_label[i])
  num_samples_final = len(samples) / num_features
  x = np.array(samples)
  x = np.reshape(x, (num_samples_final, num_features))
  y = np.array(labels)
  print "Number of samples: " + str(num_samples_final)
  print "Number of features: " + str(num_features)

  final_x = np.empty((0, num_features))
  final_y = np.empty((1, 0))
  preterm_idx = []
  num_preterm = 0
  for i in range(num_samples_final):
    if y[i] == 2:
      final_x = np.vstack((final_x, x[i]))
      preterm_idx.append(i)
      final_y = np.append(final_y, y[i])
      num_preterm += 1
  count = 0
  for i in preterm_idx:
    temp_x = np.delete(x, i-count, 0)
    temp_y = np.delete(y, i-count, 0)
    count += 1

  print "Number of preterm: " + str(num_preterm)
  num_rounds = 10
  n_folds = 10
  final_accuracy = 0.0
  final_false_positive_rate = 0.0
  final_false_negative_rate = 0.0
  final_true_positive_rate = 0.0
  for n_round in range(num_rounds):
    for i in range(num_preterm):
      idx = np.random.random_integers(0, temp_x.shape[0]-1)
      final_x = np.vstack((final_x, temp_x[idx]))
      final_y = np.append(final_y, temp_y[i])
    indices = np.arange(final_x.shape[0])
    np.random.seed()
    np.random.shuffle(indices)
    final_x = final_x[indices]
    final_y = final_y[indices]

    # Shuffle the whole samples
    #indices = np.arange(x.shape[0])
    #np.random.seed()
    #np.random.shuffle(indices)
    #x_train = x[indices]
    #y_train = y[indices]
    accuracy = 0.0
    false_negative_rate = 0.0
    false_positive_rate = 0.0
    true_positive_rate = 0.0
    cv_arg = KFold(n_folds, shuffle=True)
    fold_num = 0
    for train_idx, test_idx in cv_arg.split(final_x):
      train_set = final_x[train_idx]
      train_label = final_y[train_idx]
      test_set = final_x[test_idx]
      ref = final_y[test_idx]

      
      logreg = linear_model.LogisticRegression()
      #clf = clf.fit(x[:num_test_set], y[:num_test_set])
      #clf = clf.fit(imputed_x[:num_test_set], y[:num_test_set])
      #result = clf.predict(x[num_test_set:])
      logreg.fit(train_set, train_label)
      result = logreg.predict(test_set)
      count = 0
      true_positive_count = 0
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
        # True positive
        if ref[i] == 2 and result[i] == 2:
          true_positive_count += 1
      if preterm_count == 0:
        continue
      accuracy += float(count) / float(len(result))
      false_negative_rate += float(false_negative_count) / float(preterm_count)
      false_positive_rate += float(false_positive_count) / float(term_count)
      true_positive_rate += float(true_positive_count) / float(preterm_count)

    accuracy /= n_folds
    false_negative_rate /= n_folds
    false_positive_rate /= n_folds
    true_positive_rate /= n_folds
    final_accuracy += accuracy
    final_false_negative_rate += false_negative_rate
    final_false_positive_rate += false_positive_rate
    final_true_positive_rate += true_positive_rate
        
    #print str(num) +" X preterm samples"
    print "Accuracy: " + str(accuracy)
    print "False negative rate: " + str(false_negative_rate)
    print "False positive rate: " + str(false_positive_rate)

    #f1_scores = cross_val_score(logreg, final_x, final_y, cv=10, scoring='f1')
    #precision = cross_val_score(logreg, final_x, final_y, cv=10, scoring='precision')
    #recall = cross_val_score(logreg, final_x, final_y, cv=10, scoring='recall')
    #auc = cross_val_score(logreg, final_x, final_y, cv=10, scoring='roc_auc')
    #print str(num) +" X preterm samples"
    ##print "Averaged F1 score: " + str(sum(f1_scores) / 10)
    ##print "Averaged precision: " + str(sum(precision) / 10)
    ##print "Averaged recall: " + str(sum(recall) / 10)
    #print auc
  final_accuracy /= num_rounds
  final_false_negative_rate /= num_rounds 
  final_false_positive_rate /= num_rounds
  final_true_positive_rate /= num_rounds

  print "Averaged metrics"
  print final_accuracy
  print final_false_negative_rate
  print final_false_positive_rate
  print "Final AUC: ", metrics.auc([0.0, final_false_positive_rate, 1.0], [0.0, final_true_positive_rate, 1.0])

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


