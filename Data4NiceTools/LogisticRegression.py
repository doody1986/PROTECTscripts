#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import collections
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


field_list = ['WTPREPREG', 'FVPREGSTWT', 'FVCURRWT', 'MR1WGHTLBR', 'MR1RBC', 'MR1HCT', \
              'MR1PLTS', 'MR1WBC', 'MR1NEUTRPH', 'MR1LYMPHS', 'MR1EOSINPHS']
other_field_list = ['PARTJOBS', 'VOLJOBS']

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

  # Shuffle the whole samples
  indices = np.arange(x.shape[0])
  np.random.seed()
  np.random.shuffle(indices)
  x_train = x[indices]
  y_train = y[indices]

  logreg = linear_model.LogisticRegression()
  scores = cross_val_score(logreg, x_train, y_train, cv=10, scoring='f1')
  print "F1 score: "
  print scores

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


