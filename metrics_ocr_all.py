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
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("A3.csv")

path = ""

from difflib import SequenceMatcher

def clean(df):
  cleaned_string = list()
  for i in df:
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
        # print(x)
        if x.lower() in ['oo', 'o0', '0o']:
          x = '0'
        # Null items:
        null_items = ['°', ',']
        for ni in null_items:
          x = x.replace(ni, '', 1)
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


# ----------------------------------------------
ver = 'ALL'
bartype = ['hbar','vbar']

ver1 = "v3_" + bartype[0].upper()
filename1 = f"Robo_extraction_"
ocrdf1 = pd.read_csv(path + filename1+ver1+".csv")
img1 = ocrdf1['Image'].unique().tolist()
df1 = df[df['Image'].isin(img1)]

print(df1.shape)

ver2 = "v3_" + bartype[1].upper()
filename2 = f"Robo_extraction_"
ocrdf2 = pd.read_csv(path + filename2+ver2+".csv")
img2 = ocrdf2['Image'].unique().tolist()
df2 = df[df['Image'].isin(img2)]
print(df2.shape)

for col in cols_name:
  ocrdf1[col] = clean(ocrdf1[col])
  ocrdf2[col] = clean(ocrdf2[col])
  df1[col] = clean(df1[col])
  df2[col] = clean(df2[col])

  if col == bar_ticks[bartype[0]]:
    ocrdf1[col] = clean_ticks(ocrdf1[col])
    df1[col] = clean_ticks(df1[col])

  if col == bar_ticks[bartype[1]]:
    ocrdf2[col] = clean_ticks(ocrdf2[col])
    df2[col] = clean_ticks(df2[col])



ocrdf = pd.concat([ocrdf1, ocrdf2], ignore_index = True) 
df = pd.concat([df1, df2], ignore_index = True) 

img_nums = ocrdf['Image'].unique().tolist()

# ------------------------------------------------


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
  temp.to_csv(path + f"{col}_accuracy_{ver}.csv", index = False)

  accuracy_table[col] = {'mae' : sum(accuracy_list)/len(accuracy_list)}
  
  return accuracy_table


accuracy_table = {}

for col in cols_name:
  accuracy_table = accuracy_column(col, accuracy_table)


from difflib import SequenceMatcher
# accuracy_list = []
def metrics(colu, table):

  total_acc = 0
  total_recall = 0
  total_precision = 0
  total_f1 = 0

  acc = 0
  true_pos = 0
  pre_len= 0

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
  accuracy_dict = {}
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

    recall = found/actual_words
    precision = found/predicted_words
    accuracy_dict[img_nums[i]] = {"Ground value" : ground_value, "OCR value" : ocr_value,'Recall' : recall, 'Precision': precision}
    total_recall += recall
    total_precision += precision

  temp = pd.DataFrame(accuracy_dict).T
  temp.to_csv(path + f"{col}_RP_robo_{ver}.csv", index = False)

  avg_recall = total_recall/len(ground_list)
  avg_precision = total_precision/len(ground_list)
  f1 = (2 * avg_precision * avg_recall) / (avg_precision + avg_recall)
  table[colu]['f1score'] = f1
  table[colu]['recall'] = avg_recall
  table[colu]['precision'] = avg_precision
  
  return table

for col in cols_name:
  accuracy_table = metrics(col, accuracy_table)


df = pd.DataFrame(accuracy_table).T
df.to_csv(path + f"{filename1}_ocr_{ver}2.csv", index = False)
print(df)