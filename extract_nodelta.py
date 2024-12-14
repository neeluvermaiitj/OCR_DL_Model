import pytesseract
import pandas as pd
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

custom_config = '--psm 11'
#  r'--oem 3 --psm 11'
# image_num = '26308'
path = "barplot_2000/"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def extract_data(image_num):
  try:
    details = pytesseract.image_to_data(path +image_num+'.png',output_type='data.frame', lang='eng',config = custom_config)

    details.dropna(axis=0, inplace=True)
    details.reset_index(drop=True, inplace=True)
    indexdrop = details[(details['text'].isin(['GB', '+', "1", "-1", " ", "EB"] ))].index
    # indexdrop = details[(details['conf']< 90 )].index
    details.drop(indexdrop , inplace=True)


    df = details.copy()
    img = cv2.imread(f'{path}{image_num}.png')
    height, width, channels = img.shape
    # height, width, channels

    # X LABELS
    # details['mindiff'] = height - details['top']
    # extract_index = details[details['mindiff'].isin(range(details['mindiff'].min()+2))]['text'].index
    # x_labels_list = details.loc[extract_index, 'text'].values.tolist()
    # df.drop(extract_index, inplace = True)
    # x_label = ' '.join(x_labels_list)

    from collections import Counter
    a1 = Counter(df['top'])
    x_label_pos = dict(sorted(a1.items(), reverse = True)).keys()

    for left in x_label_pos:
      tempdf = details[(details['top'].isin(range(left-2, left+2)))& (details['left'].isin(range(int(width/2 -200), int(width/2 +150) )))]
      if not tempdf.empty:
        extract_index = tempdf['text'].index
        x_labels_list = details.loc[extract_index, 'text'].values.tolist()
        df.drop(extract_index, inplace = True)
        x_label = ' '.join(x_labels_list) 
        break

    # Y LABELS
    details['maxdiff'] = width - details['left']
    y_label_pos = details['maxdiff'].max()
    extract_index = details[details['maxdiff'].isin(range(y_label_pos, y_label_pos-3, -1))]['text'].index
    y_label_list = details.loc[extract_index, 'text'].tolist()
    df.drop(extract_index, inplace = True)
    df = df.reset_index(drop = True)
    y_label = ' '.join(y_label_list)

    from collections import Counter
    a1 = Counter(df['top'])
    top_count = sorted(a1, key=a1.get, reverse=True)
    title_pos, x_ticks_pos = top_count[0:2]
    title_pos, x_ticks_pos

    # TITLE
    indx = df[df['top'].isin(range(title_pos, title_pos+4))].index.tolist()

    title_list = []
    for num, i in enumerate(indx):
      if df.loc[i,'word_num'] == num+1 :
        title_list.append(df.loc[i, 'text'])
        df.drop(i, inplace = True)
    df = df.reset_index(drop = True)
    title = ' '.join(title_list)
    # title

    # X TICKS
    indx = df[df['top'] == x_ticks_pos].index.tolist()

    xticks_list = []
    for num, i in enumerate(indx):
      xticks_list.append(df.loc[i, 'text'])
      df.drop(i, inplace = True)
    df = df.reset_index(drop = True)
    # xticks = ' '.join(xticks_list)

    # Y TICKS
    new = df[df['left'] < 100]
    extract_index = new.index
    # df.drop(extract_index, inplace = True)
    # df = df.reset_index(drop = True)
    a2 = Counter(new['top'])
    yticks_list = []
    for _value in a2.keys():
      aa = new[new['top'] == _value]['text'].tolist()
      yticks_list.append(' '.join(aa))

    # LEGENDS
    # new = df.copy()
    extract_index = new.index
    df.drop(extract_index, inplace = True)
    df = df.reset_index(drop = True)
    a2 = Counter(new['top'])
    legends_list = []
    for _value in a2.keys():
      aa = df[df['top'] == _value]['text'].tolist()
      legends_list.append(' '.join(aa))

    # Using nearest y labels value
    img_data = {'Image' : image_num, 'image_width': width, 'image_height':height, 'Title' : title,'X-label':x_label, 'X-tick': xticks_list, 'Y-label':y_label, 'Y-tick':yticks_list, 'Legend': legends_list}
    # print(img_data)
    return img_data
  except Exception as e:
    print(image_num)
    print(e)
    return {'Image' : image_num, 'image_width': width, 'image_height':height, 'Title' : '','X-label':'', 'X-tick': '', 'Y-label':'', 'Y-tick':'', 'Legend': ''}




df = pd.read_csv(r"F:\Dataset V2\Dataset V2\Paraphrase\2K-Q-testset_Formatted_with_bartype.csv")
files = df[df['type'] == 'hbar']['Image'].unique().tolist()

# files = [file for file in os.listdir(path) if file.endswith(".png")]
# print(files)
print(len(files))
final_data = {}
for num in tqdm(files):
  # num = image.split('.')[0]
  final_data[num] = extract_data(str(num))


dframe = pd.DataFrame()
for k in final_data.keys():
  dframe = dframe.append(final_data[k], ignore_index = True)

dframe.to_csv(f"Extraction_nodelta.csv", index = False)
