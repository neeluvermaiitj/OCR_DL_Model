import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import re
import nltk
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

df = pd.read_csv("A3.csv")

bartype = 'hbar'

path = ""
ver = "v3_HBAR2"
filename = f"RCNN_{ver}"
# filename = f"final_extraction_2"

ocrdf = pd.read_csv(path + filename+".csv")
img_nums = ocrdf['Image'].unique().tolist()

from difflib import SequenceMatcher

def clean(df):
  cleaned_string = list()
  for i in df:
    if isinstance(i, float):
      i = 'none'
    new = re.sub(r'[^\w\s](?!(?<=\w\+)\d)(?!(?<=\d\.)\d)', '', i.strip())
    neww = re.sub(r'\s+(?=\s)', '', new)
    cleaned_string.append(neww.lower())
  return cleaned_string


def clean_ticks(df):
  tmp = df.tolist()
  clean_list = []
  for item in tmp:
    res = []
    for x in item.split(' '):
      try:
        if x.lower() in ['oo', 'o0', '0o']:
          x = '0'

        if x.replace('.','',1).replace('e-','',1).replace('e+','',1).isdigit() and float(x).is_integer():
            res.append(str(int(eval(x))))
        else:
          res.append(str(x))
          
      except:
        res.append(str(x))
    # print(res)
    res = ' '.join(res)
    clean_list.append(res)
  return clean_list

bar_ticks = {
  'hbar' : 'X-tick',
  'vbar' : 'Y-tick'
}

cols_name = ['Title', 'X-label','Y-label', 'X-tick', 'Y-tick', 'Legend']


img_nums = ocrdf['Image'].unique().tolist()

# ------------------------------------------------



for col in cols_name:
  df[col] = clean(df[col])
  ocrdf[col] = clean(ocrdf[col])
  if col == bar_ticks[bartype]:
    df[col] = clean_ticks(df[col])
    ocrdf[col] = clean_ticks(ocrdf[col])


def accuracy_column(col, accuracy_table):
  ground_list = []
  ocr_list = []
  for nn in img_nums:
    ground_list.append(df[df['Image'] == nn][col].iloc[0])
    ocr_value = ocrdf[ocrdf['Image'] == nn][col].iloc[0]
    
    if pd.isnull(ocr_value) or ocr_value == ' ':
      ocr_list.append("")
    else:

      ocr_list.append(ocr_value)

  accuracy_dict = {}
  accuracy_list = []
  for i in range(len(ocr_list)):
    ground_value = ground_list[i]
    ocr_value = ocr_list[i]
    sm = SequenceMatcher(None, ocr_value, ground_value)
    true_positive_char_num = 0
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
      if tag== 'equal':
        true_positive_char_num += (j2 - j1)
      else:
        pass
    accur = true_positive_char_num/len(ground_value)
    
    accuracy_dict[img_nums[i]] = {"Ground value" : ground_value, "OCR value" : ocr_value,'Accuarcy' : accur}
    accuracy_list.append(accur)

  temp = pd.DataFrame(accuracy_dict).T

  accuracy_table[col] = {'mae' : sum(accuracy_list)/len(accuracy_list)}
  
  return accuracy_table


accuracy_table = {}


for col in cols_name:
  accuracy_table = accuracy_column(col, accuracy_table)

from difflib import SequenceMatcher

def metrics(colu, table):

  total_acc = 0
  total_recall = 0
  total_precision = 0
  total_f1 = 0

  acc = 0
  true_pos = 0
  pre_len= 0
  accuracy_dict = {}
  ground_list = []
  ocr_list = []
  for nn in img_nums:
    ground_list.append(df[df['Image'] == nn][colu].iloc[0])
    ocr_value = ocrdf[ocrdf['Image'] == nn][colu].iloc[0]
    if pd.isnull(ocr_value) or ocr_value == ' ':
      ocr_list.append("")
    else:
      ocr_list.append(ocr_value)
  actual = ground_list

  for i in range(len(ocr_list)):
    ground_value = ground_list[i]
    ocr_value = ocr_list[i]
    g_list = list(ground_value.split(" "))
    o_list = list(ocr_value.split(" "))

    found = 0
    actual_words = len(g_list)
    predicted_words = len(o_list)
    temp_list = g_list.copy()
    for word in o_list:
      if word in temp_list:
        found += 1
        temp_list.remove(word)

    # print()

    recall = found/actual_words
    precision = found/predicted_words
    accuracy_dict[img_nums[i]] = {"Ground value" : ground_value, "OCR value" : ocr_value,'Recall' : recall, 'Precision': precision}
    # accuracy_list.append(accur)
    total_recall += recall
    total_precision += precision

  temp = pd.DataFrame(accuracy_dict).T
  avg_recall = total_recall/len(ground_list)
  avg_precision = total_precision/len(ground_list)
  f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)
  table[colu]['f1score'] = f1
  table[colu]['recall'] = avg_recall
  table[colu]['precision'] = avg_precision

  print(f"------{colu}-------\n")
  print(len(ground_list))

  return table

for col in cols_name:
  accuracy_table = metrics(col, accuracy_table)

print("-"*50)
print(filename)
print("-"*50)
df = pd.DataFrame(accuracy_table).T
df.to_csv(path + f"{filename}_{ver}.csv", index = False)
print(df)