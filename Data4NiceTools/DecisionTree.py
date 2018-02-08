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
from sklearn.model_selection import KFold
from sklearn import metrics

label_name = 'PPTERM'
#label_name = 'Class'
def Run(raw_data_file):
  data = pd.read_csv(raw_data_file)
  raw_label = data[label_name].tolist()
  data.drop(label_name, axis=1, inplace=True)
  if label_name == 'Class':
    data = data.replace('?', np.NaN)
  results = collections.OrderedDict()
  num_features = len(data.columns)
  num_samples = len(data.index)
  samples = []
  labels = []
  for i in data.index:
    sample = []
    for column in data.columns:
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
  num_preterm = 0
  for i in labels:
    if i == 2:
      num_preterm += 1
  print "Number of Preterm samples: " + str(num_preterm)

  imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
  
  preterm_idx = []
  term_idx = []
  for i in range(num_samples_final):
    if y[i] == 2:
      preterm_idx.append(i)
    elif y[i] == 1:
      term_idx.append(i)

  preterm_x = x[preterm_idx]
  term_x = x[term_idx]
  
  imp.fit(preterm_x)
  imputed_preterm_x = imp.transform(preterm_x)
  imp.fit(term_x)
  imputed_term_x = imp.transform(term_x)

  imputed_x = np.vstack((imputed_preterm_x, imputed_term_x))

  num_rounds = 10
  n_folds = 10
  #num = 9
  #final_x = np.empty((0, num_features))
  #final_y = np.empty((1, 0))
  #preterm_idx = []
  #for i in range(num_samples_final):
  #  if y[i] == 2:
  #    final_x = np.vstack((final_x, imputed_x[i]))
  #    preterm_idx.append(i)
  #    final_y = np.append(final_y, y[i])
  #temp_x = np.delete(imputed_x, preterm_idx, 0)
  #temp_y = np.delete(y, preterm_idx, 0)

  #for i in range(num_preterm * num):
  #  final_x = np.vstack((final_x, temp_x[i]))
  #  final_y = np.append(final_y, temp_y[i])
  #indices = np.arange(final_x.shape[0])
  indices = np.arange(imputed_x.shape[0])
  np.random.seed()
  np.random.shuffle(indices)
  #final_x = final_x[indices]
  #final_y = final_y[indices]
  final_x = imputed_x[indices]
  final_y = y[indices]
  final_accuracy = 0.0
  final_false_positive_rate = 0.0
  final_false_negative_rate = 0.0
  final_true_positive_rate = 0.0
  for num_temp in range(num_rounds):
    #final_x = np.empty((0, num_features))
    #final_y = np.empty((1, 0))
    #preterm_idx = []
    #for i in range(num_samples_final):
    #  if y[i] == 2:
    #    final_x = np.vstack((final_x, imputed_x[i]))
    #    preterm_idx.append(i)
    #    final_y = np.append(final_y, y[i])
    #temp_x = np.delete(imputed_x, preterm_idx, 0)
    #temp_y = np.delete(y, preterm_idx, 0)

    #for i in range(num_preterm * num):
    #  final_x = np.vstack((final_x, temp_x[i]))
    #  final_y = np.append(final_y, temp_y[i])
    #indices = np.arange(final_x.shape[0])
    #np.random.seed()
    #np.random.shuffle(indices)
    #final_x = final_x[indices]
    #final_y = final_y[indices]
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

      clf = tree.DecisionTreeClassifier(criterion="entropy")
      #clf = clf.fit(x[:num_test_set], y[:num_test_set])
      #clf = clf.fit(imputed_x[:num_test_set], y[:num_test_set])
      #result = clf.predict(x[num_test_set:])
      clf.fit(train_set, train_label)
      result = clf.predict(test_set)
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
      #print float(count) / float(len(result))
      #print float(false_negative_count) / float(preterm_count)
      #print float(false_positive_count) / float(term_count)
      dot_file = 'tree_'+str(fold_num)+'.dot'
      tree.export_graphviz(clf,
                           feature_names=data.columns,  
                           #feature_names=field_list,  
                           class_names=['Term', 'Preterm'],
                           filled=True, rounded=True,  
                           special_characters=True,
                           out_file=dot_file)
      fold_num += 1
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

    #dot_file = 'tree_'+str(num)+'.dot'
    #tree.export_graphviz(clf,
    #                     feature_names=data.columns,  
    #                     #feature_names=field_list,  
    #                     class_names=['Term', 'Preterm'],
    #                     filled=True, rounded=True,  
    #                     special_characters=True,
    #                     out_file=dot_file)
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


