#! /usr/bin/env python

import sys
import csv
import re
from enum import Enum

class DataType(Enum):
  TEXT = 0,
  RADIO = 1,
  NUMBER = 2

extract_all = False

# According the data dictionary
field_index = 0
form_index = 1
type_index = 3
choice_index = 5
text_type_index = 7

human_subject_form_list = ["first_visit", "med_rec_v1", "product_use"]

def Extract(csv_file, data_type):
  readfile = csv.reader(open(csv_file, "r"))
  if data_type == DataType.RADIO:
    prefix = "categorical_"
  elif data_type == DataType.NUMBER:
    prefix = "numerical_"
  else:
    prefix = "text_"
  writefile = csv.writer(open(prefix+"chosenfields.csv", "w"))

  # write header
  #header = next(readfile)

  # Human subject data
  for row in readfile:
    new_row = []
    if row[type_index] == "radio" or row[type_index] == 'dropdown':
      if data_type == DataType.RADIO:
        new_row.append(row[field_index].upper())
        writefile.writerow(new_row)
      else:
        continue
    elif row[text_type_index] == "number" or row[text_type_index] == "integer":
      if data_type == DataType.NUMBER:
        new_row.append(row[field_index].upper())
        writefile.writerow(new_row)
      else:
        continue
    elif row[type_index] == 'checkbox':
      if data_type == DataType.RADIO:
        field_choices = row[choice_index]
        sepintlist = field_choices.split('|')
        for item in sepintlist:
          new_row = []
          found_int = re.search("\d+", item)
          new_row.append(row[field_index].upper()+"__"+str(found_int.group()))
          writefile.writerow(new_row)
    else:
      if data_type == DataType.TEXT:
        new_row.append(row[field_index].upper())
        writefile.writerow(new_row)
      else:
        continue

  # Biological data
  #key_field = "CONC"
  #row = [key_field]
  #writefile.writerow(row)

  # Postpartum label
  birth_label = "PPTERM"
  row = [birth_label]
  writefile.writerow(row)

def main():
  print ("Start program.")

  if len(sys.argv) < 2:
    print "Too few arguments"
    print "Please specify the data type."
    sys.exit()
  
  data_type_str = sys.argv[1]
  if data_type_str == "NUM":
    data_type = DataType.NUMBER
  elif data_type_str == "CATE":
    data_type = DataType.RADIO
  elif data_type_str == "TEXT":
    data_type = DataType.TEXT
  else:
    print "Un-known type"
    exit()

  filename = "human_subjects_dd.csv"
  Extract(filename, data_type)

  print ("End program.")
  return 0

if __name__ == '__main__':
  main()
