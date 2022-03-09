from cgitb import strong
from email.mime import image
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
from sklearn.cluster import KMeans
from collections import Counter
from collections import defaultdict
import sys
import math

class getCorners:
	def __init__(self, frame):
		self.table_center = [0, 0]
		self.frame = frame
		self.hsv = None
		self.isShortSide = False

	def __call__(self):
		#img = load_img(self.frame)
		hsv_img = self.to_hsv(self.frame)
		self.hsv = self.get_cloth_colour(hsv_img)
		print('hsv', self.hsv)
		table_outline = self.get_table_outline(self.hsv, hsv_img)
		corners = self.find_corners(table_outline) #output list is 3D
		#corners = [elem for twod in corners for elem in twod] #convert to 2D
		self.show_img_compar_i([hsv_img, table_outline])
		i = self.order_corners(corners)
		print('i', i)
		return i
		return self.order_corners(corners)

	#table_center = [0, 0]

	def load_img(self, img):
		frame = cv.imread(img)
		#_, src_img = vidcap.read()
		return frame

	def to_hsv(self, img):
		return cv.cvtColor(img, cv.COLOR_BGR2HSV)


	# Function to calculate most common hsv colour in a given image
	def get_cloth_colour(self, img):
		h, w, c = img.shape
		img = (img[int(0.4*h):int(0.7*h),int(0.35*w):int(0.65*w)].copy())
		# Get shape of image and extract a portion if the image near the middle
		# KMeans with cluster of 4
		clt = KMeans(n_clusters=4)
		clt.fit(img.reshape(-1, 3))
		clt.labels_
		clt.cluster_centers_

		# Function which finds the most common colour in the image and returns its HSV value
		n_pixels = len(clt.labels_)
		counter = Counter(clt.labels_) 
		curMax = 0
		index = 0
		for i in counter:
			fraction = np.round(counter[i]/n_pixels, 2)
			if fraction > curMax:
				curMax = fraction
				index = i

		hsv_vals = clt.cluster_centers_[index]
		
		return hsv_vals



	#Function to generate mask and find largest contour
	def get_table_outline(self, hsv, hsv_img):
		# Set upper and lower bounds from the calculated HSV value
		lower_bound = np.array([hsv[0] - 20, hsv[1] - 115, hsv[2] - 115]) 
		upper_bound = np.array([hsv[0] + 20, hsv[1] + 115, hsv[2] + 115]) 

		#lower_bound = np.array([78.8, 39.9 - 115, 152.18 - 115])
		#upper_bound = np.array([98.8 + 20, 39.9 + 115, 152 + 115]) 

		# Compute mask and remove unnecessary noise from it
		mask = cv.inRange(hsv_img, lower_bound, upper_bound)
		kernel = np.ones((7,7),np.uint8)
		mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
		mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
		#median = cv.medianBlur(mask, 5)

		# Compute all the contours in the mask 
		contours, hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
		# Extract the largest of these contours (i.e contour of table)
		maxContour = max(contours, key = cv.contourArea)
		# Compute center of table and draw contour
		M = cv.moments(maxContour)
		self.table_center[0] = int(M["m10"] / M["m00"])
		self.table_center[1] = int(M["m01"] / M["m00"])
		# draw the contour and center of the shape on the image
		img = cv.drawContours(np.zeros_like(hsv_img), [maxContour], -1, (180, 255, 255), 1)

		# Use probabalistic Hough Transform to extract line segments from contour
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

		return img
		


	# Extract 4 lines from the contour and compute intersections
	def find_corners(self, contour_img):
		#lines = cv.HoughLinesP(src_img2, 1, np.pi/180, 150, None, 400, 100) #minlinelength is dependent on pixels
		lines = cv.HoughLines(contour_img, 1, np.pi / 180, 75, None, 0, 0) #CHANGED 75 FROM 100!!!!!!!!
		print("lines", lines)

		strong_lines = np.zeros([4,1,2])
		n2 = 0
		for n1 in range(0,len(lines)):
			for rho,theta in lines[n1]:
				if n1 == 0:
					strong_lines[n2] = lines[n1]
					n2 += 1
				else:
					#if rho < 0:
					#	rho*=-1
					#	theta-=np.pi
					closeness_rho = np.isclose(rho,strong_lines[0:n2,0,0],atol = 50) # dependent on image resolution
					closeness_theta = np.isclose(theta,strong_lines[0:n2,0,1],atol = np.pi/36) #changed atol to 0.26 from np.pi/36
					closeness = np.all([closeness_rho,closeness_theta],axis=0)
					if not any(closeness) and n2 < 4:
						strong_lines[n2] = lines[n1]
						n2 += 1
		
		parallel_indexes = [0, 0]
		cont = True
		for i in range(0, len(strong_lines) - 1):
			if cont == True:
				for j in range(i + 1, len(strong_lines)):
					if np.isclose(strong_lines[i][0][1], strong_lines[j][0][1],atol = 0.35):
						parallel_indexes = [i, j]
						cont = False
						break
			else:
				break

		#print(parallel_indexes)
		print("strong lines", strong_lines)


		def intersection(line1, line2):
			try:
				rho1, theta1 = line1[0]
				rho2, theta2 = line2[0]
				A = np.array([
					[np.cos(theta1), np.sin(theta1)],
					[np.cos(theta2), np.sin(theta2)]
				])
				b = np.array([[rho1], [rho2]])
				x0, y0 = np.linalg.solve(A, b)
				x0, y0 = int(np.round(x0)), int(np.round(y0))
				return (x0, y0)
			except:
				return None

		corners = []
		for i in parallel_indexes:
			line1 = strong_lines[i]
			for j, line2 in enumerate(strong_lines):
				if j not in parallel_indexes:
					inter = intersection(line1, line2)
					if inter != None:
						corners.append(inter) 
		
		print("corners", corners)
		print("center", (self.table_center[0], self.table_center[1]))

		for i in corners:
			cv.circle(contour_img, (i[0], i[1]), 5, (255, 0, 255), -1)
		cv.circle(contour_img, (self.table_center[0], self.table_center[1]), 5, (90, 65, 85), -1)
		cv.putText(contour_img, "center", (self.table_center[0] - 50, self.table_center[1] - 20),
				cv.FONT_HERSHEY_SIMPLEX, 1, (90, 65, 85), 2)

		return corners

	def order_corners(self, corners): #redo this method,by maybe taking two largest y's as the bottom points 
		ul, ur, lr, ll, = None, None, None, None
		corners.sort()
		if corners[0][1] > corners[1][1]:
			ll, ul = corners[0], corners[1]
		else:
			ll, ul = corners[1], corners[0]
		
		if corners[2][1] > corners[3][1]:
			lr, ur = corners[2], corners[3]
		else:
			lr, ur = corners[3], corners[2]


		#for p in corners:
		#	if (p[0] < self.table_center[0]) and (p[1] < self.table_center[1]):
		#		if (ul != None and p
		#		ul = p 
		#	elif (p[0] > self.table_center[0]) and (p[1] < self.table_center[1]):
		#		ur = p 
		#	if (p[0] < self.table_center[0]) and (p[1] > self.table_center[1]):
		#		ll = p 
		#	else:
		#		lr = p 
		#for p in corners:
		#	if (p[1] > self.table_center[1]) and ll != None:
		#		if p[0] < ll[0]:
		#			ll, lr = p, ll
		#		else:
		#			lr = p
		#	elif (p[1] > self.table_center[1]) and ll == None:
		#		ll = p
		#	elif (p[1] < self.table_center[1]) and ul != None:
		#		if p[0] < ul[0]:
		#			ul, ur = p, ul
		#		else:
		#			ur = p
		#	else:
		#		ul = p

		left_side = math.dist(ul, ll) #??????
		right_side = math.dist(ur, lr)
		bottom_side = math.dist(ll, lr)
		top_side = math.dist(ul, ur)

		if (min(left_side, right_side) * 2 >= min(bottom_side, top_side)): #(2 * length >= width):
			#print('shortside', left_side, bottom_side)
			self.isShortSide = True
		print('angle', self.isShortSide)   

		#h = int(2000)
		#w = int(h/2)
		if self.isShortSide:
			return np.array([ur, lr, ll, ul], dtype=np.float32)	
		else:
			return np.array([ul, ur, lr, ll], dtype=np.float32)



#print(get_corners(r'C:\Users\reuby\OneDrive\Pictures\homePoolVideo.mp4'))
# Function to display a given list of images
	def show_img_compar_i(self, imgs):
		f, ax = plt.subplots(1, len(imgs), figsize=(10,10))
		for i in range(len(imgs)):
			img = imgs[i]
			ax[i].imshow(img)
			ax[i].axis('off')
		f.tight_layout()
		plt.savefig('demo3.png', bbox_inches='tight')

		#plt.show()

#img = cv.imread(r'C:\Users\reuby\OneDrive\Pictures\2ball.jpeg')
#c = getCorners(img)
#c()
#print(c())

