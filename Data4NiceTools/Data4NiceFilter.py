#! /usr/bin/env python

import numpy as np
import pandas as pd
import sys
import csv
import time

# According the data dictionary
field_index = 0
form_index = 1
type_index = 3
choice_index = 5
text_type_index = 7

text_fields = []
categorical_fields = []
checkbox_fields = []
numerical_fields = []
mixed_fields = []
def Extract(dd_file, data_type):
  readfile = csv.reader(open(dd_file, "r"))

  # Human subject data
  for row in readfile:
    if row[type_index] == "radio" or row[type_index] == 'dropdown':
      categorical_fields.append(row[field_index].upper())
      mixed_fields.append(row[field_index].upper())
    elif row[text_type_index] == "number" or row[text_type_index] == "integer":
      numerical_fields.append(row[field_index].upper())
      mixed_fields.append(row[field_index].upper())
    elif row[type_index] == 'checkbox':
      field_choices = row[choice_index]
      sepintlist = field_choices.split('|')
      for item in sepintlist:
        found_int = re.search("\d+", item)
        checkbox_fields.append(row[field_index].upper()+"__"+str(found_int.group()))
        mixed_fields.append(row[field_index].upper()+"__"+str(found_int.group()))
    else:
      text_fields.append(row[field_index].upper())

def main():
  filename = "human_subjects_dd.csv"
  Extract(filename, data_type)

if __name__ == '__main__':
  main()

def Filter(chosen_fields, raw_data_file):
  # Get chosen fields
  readfile = csv.reader(open(chosen_fields_file, "r"))
  chosen_fields = []
  for row in readfile:
    chosen_fields.append(row[0])

  data = pd.read_csv(raw_data_file)
  no_ambiguous_data = True
  if no_ambiguous_data:
    df_dict[file_name].replace('888', np.nan, inplace=True, regex=True)
    df_dict[file_name].replace('999', np.nan, inplace=True, regex=True)
  print "The column number of raw data BEFORE filtered is: " + str(len(data.columns))
  for column in data.columns:
    if column == "STUDY_ID":
      #data.drop(column, axis=1, inplace=True)
      continue
    if column == "PPTERM":
      continue

    # Remove unchosen fields
    if column not in chosen_fields:
      data.drop(column, axis=1, inplace=True)
      continue
    
    null_flags = data[column].isnull()
    remove = True
    null_count = 0
    for flag in null_flags:
      if flag == False:
        remove = False
      if flag == True:
        null_count += 1

    proportion_by_samples = float(null_count) / float(len(null_flags))

    if remove == True or proportion_by_samples > 0.5:
      data.drop(column, axis=1, inplace=True)

  print "The column number of raw data AFTER filtered is: " + str(len(data.columns))

  # Remove some samples with too little valid features
  print "The row number of raw data BEFORE filtered is: " + str(len(data.index))
  remove_idx = []
  for i in data.index:
    data_row = data.iloc[[i]]
    num_features = float(len(data.columns) - 2)
    null_count = 0
    for column in data.columns:
      if column == "STUDY_ID" or column == "PPTERM":
        continue
      if pd.isnull(data_row[column][i]):
        null_count += 1

    proportion_by_features = float(null_count) / num_features
    if proportion_by_features > 0.5:
      remove_idx.append(i)
    #  data = data.drop(data.index[i])
  print len(remove_idx)
  for i in range(len(remove_idx)):
    data = data.drop(data.index[remove_idx[i] - i])
    
    
  print "The row number of raw data AFTER filtered is: " + str(len(data.index))

  #data.fillna(0, inplace=True)
  data.to_csv("filtered_" + chosen_fields_file[:-4] + "_" + raw_data_file[:-4] + "_" + time.strftime("%m%d%Y") + ".csv")


def main():
  print ("Start program.")

  if len(sys.argv) < 3:
    print "Too few arguments"
    print "Please specify the choisen fields file and csv file."
    sys.exit()

  choisen_fields_file = sys.argv[1]
  filename = sys.argv[2]
  Filter(choisen_fields_file, filename)

  print ("End program.")
  return 0

if __name__ == '__main__':
  main()
