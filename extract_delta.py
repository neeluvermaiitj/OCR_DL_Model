import pytesseract
import pandas as pd
import numpy as np
import cv2
import os
import pandas as pd
from tqdm import tqdm
import glob
import time
import warnings
warnings.filterwarnings("ignore")


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# path = "Images\\"
# _img = glob.glob(path+'*.png')
# # Separate the image name from the filename
# img_num = [str((name.split("\\")[1]).split('.')[0]) for name in _img]
path = "C:\\Users\\Alka\\Downloads\\Neelu\\png\\barplot_2000\\"
df = pd.read_csv(r"F:\Dataset V2\Dataset V2\Paraphrase\2K-Q-testset_Formatted_with_bartype.csv")
img_num = df[df['type'] == 'hbar']['Image'].unique().tolist()


def edge_detect(image):

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	gray = cv2.bilateralFilter(gray, 1, 7, 17)
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

		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		if w > max_width and (y+h < height):
		  max_width = w
		  main_ = [x,y, w,h]
		x_.append(x)
		y_.append(y)
		w_.append(w)
		h_.append(h)

	cv2.rectangle(img,(main_[0], main_[1]),(main_[0]+main_[2],main_[1]+main_[3]),(0,255,0),2)
	print(main_, height)
	# cv2.imshow("ok",img)

	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return main_

def save_parts(num, type, name):
	save_loc = "Extracted_2\\"
	if not os.path.exists(save_loc):
	  os.mkdir(save_loc)
	
	if not os.path.exists(save_loc+num):
	  os.mkdir(save_loc+num)
	# print(f"Extracted\\{img_num[0]}\\type.png")
	cv2.imwrite(f"{save_loc}{num}\\{name}",type)

custom_config = r'--oem 3 --psm 11'
def extract_text(img, name = ''):
	details = pytesseract.image_to_data(img,output_type='data.frame', lang='eng', config = custom_config)
	details.dropna(axis=0, inplace=True)
	details.reset_index(drop=True, inplace=True)
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

final_data = {}
error = []
# img_num =  ['26317', '26597', '26824']
for im in tqdm(img_num):
	try:
		im = str(im)
		img = cv2.imread(path+im+".png")
		height, width, channels = img.shape

		edges = edge_detect(img)
		(cnts, _) = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		main_ = main_bbox(cnts)
		# cv2.rectangle(img,(main_[0], main_[1]),(main_[0]+main_[2],main_[1]+main_[3]),(0,0,255),2)

		x, y, w, h = 0, main_[1]+main_[3], main_[0]+main_[2], main_[1]+main_[3]+20
		# cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		# cv2.imshow("ok",img)

		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		img = cv2.imread(path+im+".png")

		x_tick = img[y:y+20, x:x+w]
		save_parts(im, x_tick, "x_tick.png")

		x, y, w, h = 0, main_[1]+main_[3]+ 25, main_[0]+main_[2], main_[1]+main_[3]+30
		x_label = img[y:y+20, x:x+w]
		save_parts(im, x_label, "x_label.png")

		x, y, w, h = 0, main_[1], main_[0],main_[1]+main_[3] - 40
		y_tick = img[y:y+h, x+30:x+w]
		save_parts(im, y_tick, "y_tick.png")
		y_label = img[y:y+h, x:x+w-35]
		y_label_rot = cv2.rotate(y_label, cv2.ROTATE_90_CLOCKWISE)
		save_parts(im, y_label_rot, "y_label_rot.png")

		x,y, w, h = main_[0],0, main_[0]+main_[2], main_[1]
		title = img[y:y+h, x:x+w]
		save_parts(im, title, "title.png")

		# print(main_[1])

		img_data = {'Image' : im, 
		# 'image_width': width, 
		# 'image_height':height, 
		'Title' : extract_text(title),
		'X-label': extract_text(x_label), 
		'X-tick': extract_text(x_tick, name = 'x_tick'), 
		'Y-label': extract_text(y_label_rot), 
		'Y-tick': extract_text(y_tick, name = "y_tick")
		}

		final_data[im] = img_data

	except Exception as e:
		error.append(im)
		print(im)
		print(e)
	# print(img_data)


df = pd.DataFrame(final_data).T
df.to_csv(f"extraction_delta.csv", index = False)

print(f"Error found in : {error}")
