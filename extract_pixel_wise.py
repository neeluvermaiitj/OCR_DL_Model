import pytesseract
import pandas as pd
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import glob
import time
from pytesseract import Output
import warnings
warnings.filterwarnings("ignore")


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


path = "C:\\Users\\Alka\\Downloads\\Neelu\\png\\barplot_2000\\"
df = pd.read_csv(r"F:\Dataset V2\Dataset V2\Paraphrase\2K-Q-testset_Formatted_with_bartype.csv")
img_num = df[df['type'] == 'hbar']['Image'].unique().tolist()
# img_num = df['Image'].unique().tolist()



def edge_detect(image):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	kernel = np.ones((3,3),np.uint8)
	erosion = cv2.erode(gray,kernel,iterations = 2)

	return erosion


def main_bbox(cnts):

	x_ = []
	y_ = []
	w_ = []
	h_ = []

	max_width = 0
	main_ = []
	for cnt in cnts:
		x,y,w,h = cv2.boundingRect(cnt)

		# cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		if w > max_width and (y+h < height):
		  max_width = w
		  main_ = [x,y, w,h]
		x_.append(x)
		y_.append(y)
		w_.append(w)
		h_.append(h)
	cv2.imshow("ok",img)
	cv2.waitKey(0)

	cv2.rectangle(img, (main_[0], main_[1]),(main_[0]+main_[2],main_[1]+main_[3]),(0,255,0),2)
	# print(main_, height)
	# cv2.imshow("ok",img)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows() 
	return main_

def save_parts(num, type, name):
	save_loc = "Extracted_rough\\"
	if not os.path.exists(save_loc):
	  os.mkdir(save_loc)
	
	if not os.path.exists(save_loc+num):
	  os.mkdir(save_loc+num)
	# print(f"Extracted\\{img_num[0]}\\type.png")
	cv2.imwrite(f"{save_loc}{num}\\{name}",type)

custom_config = r'--oem 3 --psm 11'
def extract_text(img, name = ''):
	details = pytesseract.image_to_data(img,output_type='data.frame', lang='eng', config = custom_config)
	# print(details)
	details.dropna(axis=0, inplace=True)
	details.reset_index(drop=True, inplace=True)
	print(details)
	result = details['text'].unique().tolist()

	if name == 'y_tick':
		# print(details)
		temp_list = []
		blocks = details['block_num'].unique().tolist()
		for bn in blocks:
			same = details[details['block_num'] == bn]['text'].unique().tolist()
			if len(same) > 1:
				same = ' '.join(same)
			else:
				same = same[0]
			temp_list.append(same)
		# print(temp_list)
		result = temp_list[::-1]
	if len(result) == 0:
		return ''
	else:
		if not isinstance(result[0], str):
			# print(result[0])
			if name == 'x_tick':
				result.insert(0, 0)
			result = [str(int(x)) for x in result]
		return ' '.join(result)


def preprocess_yaxis(image):
	details = pytesseract.image_to_data(image,output_type='data.frame', lang='eng', )
	details.dropna(axis=0, inplace=True)
	details.reset_index(drop=True, inplace=True)
	# print(details)
	h, w, c = image.shape

	y_labeldf = details[details['height'] > details['width']]


	if y_labeldf.empty:

		ylabel = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE,)
		return ylabel, image

	else:
		# print(y_labeldf)
		y_label_width = y_labeldf['width'].iloc[0]
		y_label_left = y_labeldf['left'].iloc[0]

		ytick = image[0:h, (y_label_left + y_label_width):w]
		ylabel = image[0:h, 0: (y_label_left + y_label_width+5)]
		ylabel = cv2.rotate(ylabel, cv2.ROTATE_90_CLOCKWISE,)
		return ylabel, ytick


def preprocess_xaxis(image):
	h, w, c = image.shape

	details = pytesseract.image_to_data(image,output_type='data.frame', lang='eng', config = "--oem 3 --psm 11")
	# print(details)
	details.dropna(axis=0, inplace=True)
	details.reset_index(drop=True, inplace=True)
	indexdrop = details[(details['conf']< 40 )].index
	details.drop(indexdrop , inplace=True)
	details.reset_index(drop=True, inplace=True)

	# df = details.sort_values('top')
	df = details.copy()

	# print(df)

	xdata = {
	0 : [],       #'xtick'
	1 : [],       #'xlabel'
	2 : []        #'legend'
	}

	for loc,i in enumerate(df.index):
		if df.empty:
			break
		if len(xdata[1]) > 0:
			xdata[2] = df['text'].tolist()
			break
		else:
			start = df['top'].iloc[0]
			found = df[df['top'].isin(range(start-5, start+16))].index
			text = df[df['top'].isin(range(start-5, start+16))]['text'].tolist()
			df.drop(found, inplace = True)
			df.reset_index(drop=True, inplace=True)
			# loc = df['top'].iloc[0]
			
			# print(text)
			# print(loc)
			if not isinstance(text[0], str):
				if loc== 0:
					text.insert(0, 0)
				text = [str(int(x)) for x in text]
			xdata[loc] = text
	return ' '.join(xdata[0]), ' '.join(xdata[1]), xdata[2]


final_data = {}
error = []
# img_num =  ['128318']
for im in tqdm(img_num[:]):
	# try:
	im = str(im)
	img = cv2.imread(path + im + ".png")
	height, width, channels = img.shape

	edges = edge_detect(img)
	(cnts, _) = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

	main_ = main_bbox(cnts)
	# cv2.rectangle(img,(main_[0], main_[1]),(main_[0]+main_[2],main_[1]+main_[3]),(255, 0, 0),2)
	
	x, y, x1, y1 = main_[0], main_[1], main_[0] + main_[2], main_[1] + main_[3]

	img = cv2.imread(path + im + ".png")
	
	y_axis = img[y:y1, 0:x]
	y_label, y_tick = preprocess_yaxis(y_axis)
	t = pytesseract.image_to_data(y_label, output_type='data.frame', config = "--psm 10")
	print("-" * 100)
	print(t['text'])
	cv2.imshow("ok",y_label)
	cv2.waitKey(0)

	cv2.imshow("ok",y_label)
	cv2.waitKey(0)
	save_parts(im, y_tick , "y_tick.png")
	save_parts(im, y_label, "y_label.png")

	title
	title = img[0:y, x:x1]
	x_axis = img[y1:height, 0:x1]
	x_tick, x_label, legend = preprocess_xaxis(x_axis)


	if len(legend) == 0:
		legend = img[ 0:height, x1: width]
		# cv2.imshow('img', legend)
		# cv2.waitKey(0)
	# save_parts(im, y_label, "y_label.png")
		d = pytesseract.image_to_data(legend, output_type='data.frame', config = "--oem 3 --psm 11")
		d.dropna(axis=0, inplace=True)
		d.reset_index(drop=True, inplace=True)
		legend = d['text'].tolist()
		legend = [str(i) for i in legend]
	legend = ' '.join(legend)

	print(f"\nImage: {im}\nX_tick : {x_tick}\nX-label : {x_label}\nLegend : {legend}")
	# save_parts(im, x_label, "x_label.png")


	img_data = {'Image' : im, 
	'image_width': width, 
	'image_height':height, 
	'Title' : extract_text(title),
	'X-label': x_label, 
	'X-tick': x_tick, 
	'Y-label': extract_text(y_label), 
	'Y-tick': extract_text(y_tick, name = "y_tick"),
	'Legend' : legend
	}
	final_data[im] = img_data


	except Exception as e:
		error.append(im)
		print(im)
		print(e)
	print(img_data)


df = pd.DataFrame(final_data).T
df.to_csv(f"Extraction_pixel_wise.csv", index = False)

print(f"Error found in : {error}")
