import cv2
import numpy as np 

class matchers:
	def __init__(self):
		self.surf = cv2.xfeatures2d.SURF_create()
		FLANN_INDEX_KDTREE = 0
		index_params = dict(algorithm=0, trees=5)
		search_params = dict(checks=50)
		self.flann = cv2.FlannBasedMatcher(index_params, search_params)

	def match(self, i1, i2, direction=None):
		# print i1
		# print i2
		imageSet1 = self.getSURFFeatures(i1)
		imageSet2 = self.getSURFFeatures(i2)
		# print imageSet1
		# print imageSet2
		print "Direction : ", direction
		matches = self.flann.knnMatch(
			imageSet2['des'],
			imageSet1['des'],
			k=2
			)
		# print 'matches ', matches
		good = []
		for i , (m, n) in enumerate(matches):
			if m.distance < 0.7*n.distance:
				good.append((m.trainIdx, m.queryIdx))
		print "good: ", len(good)
		if len(good) > 4:
			pointsCurrent = imageSet2['kp']
			pointsPrevious = imageSet1['kp']
			
			# print pointsCurrent
			# print pointsPrevious

			matchedPointsCurrent = np.float32(
				[pointsCurrent[i].pt for (__, i) in good]
			)
			# print matchedPointsCurrent
			matchedPointsPrev = np.float32(
				[pointsPrevious[i].pt for (i, __) in good]
				)
			# print matchedPointsPrev

			# f=open('a.txt','w')
			# f.write(matchedPointsCurrent)
			# f.write('\n')
			# f.write(matchedPointsPrev)
			# f.close()

			H, s = cv2.findHomography(matchedPointsCurrent, matchedPointsPrev, cv2.RANSAC, 4)
			# print 'hjb', H
			return H
		return None

	def getSURFFeatures(self, im):
		gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		kp, des = self.surf.detectAndCompute(gray, None)
		return {'kp':kp, 'des':des}