# import the necessary packages
from collections import OrderedDict
import numpy as np
import cv2
import dlib

# define a dictionary that maps the indexes of the facial
# landmarks to specific face regions

#For dlib’s 68-point facial landmark detector:
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

#For dlib’s 5-point facial landmark detector:
FACIAL_LANDMARKS_5_IDXS = OrderedDict([
	("right_eye", (2, 3)),
	("left_eye", (0, 1)),
	("nose", (4))
])

# in order to support legacy code, we'll default the indexes to the
# 68-point model
FACIAL_LANDMARKS_IDXS = FACIAL_LANDMARKS_68_IDXS

def rect_to_bb(rect):
	# take a bounding predicted by dlib and convert it
	# to the format (x, y, w, h) as we would normally do
	# with OpenCV
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y

	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
	# create two copies of the input image -- one for the
	# overlay and one for the final output image
	overlay = image.copy()
	output = image.copy()

	# if the colors list is None, initialize it with a unique
	# color for each facial landmark region
	if colors is None:
		colors = [(19, 199, 109), (79, 76, 240), (230, 159, 23),
			(168, 100, 168), (158, 163, 32),
			(163, 38, 32), (180, 42, 220)]

	# loop over the facial landmark regions individually
	for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
		# grab the (x, y)-coordinates associated with the
		# face landmark
		(j, k) = FACIAL_LANDMARKS_IDXS[name]
		pts = shape[j:k]

		# check if are supposed to draw the jawline
		if name == "jaw":
			# since the jawline is a non-enclosed facial region,
			# just draw lines between the (x, y)-coordinates
			for l in range(1, len(pts)):
				ptA = tuple(pts[l - 1])
				ptB = tuple(pts[l])
				cv2.line(overlay, ptA, ptB, colors[i], 2)

		# otherwise, compute the convex hull of the facial
		# landmark coordinates points and display it
		else:
			hull = cv2.convexHull(pts)
			cv2.drawContours(overlay, [hull], -1, colors[i], -1)

	# apply the transparent overlay
	cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

	# return the output image
	return output


def calculateHOG(img_list):
	hist_features = None

	cell_size = (8, 8)  # h x w in pixels
	block_size = (8, 8)  # h x w in cells
	nbins = 32  # number of orientation bins

	img = img_list[0]
	# winSize is the size of the image cropped to an multiple of the cell size
	hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
									  img.shape[0] // cell_size[0] * cell_size[0]),
							_blockSize=(block_size[1] * cell_size[1],
										block_size[0] * cell_size[0]),
							_blockStride=(cell_size[1], cell_size[0]),
							_cellSize=(cell_size[1], cell_size[0]),
							_nbins=nbins)

	for i in range(len(img_list)):
		img = img_list[i].astype(np.uint8)
		#         n_cells = (img.shape[0] // cell_size[0], img.shape[1] // cell_size[1])
		hog_feats = hog.compute(img)  # \
		#                        .reshape(n_cells[1] - block_size[1] + 1,
		#                                 n_cells[0] - block_size[0] + 1,
		#                                 block_size[0], block_size[1], nbins) \
		#                        .transpose((1, 0, 2, 3, 4))  # index blocks by rows first

		if hist_features is None:
			hist_features = np.expand_dims(hog_feats, axis=0)
		else:
			hist_features = np.vstack((hist_features, np.expand_dims(hog_feats, axis=0)))

	return hist_features


def load_dlib_detector(path):
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(path)
    return detector, predictor

def face_detector_dlib(detector, image_gray):
    # запускаем детектор лиц
    rects = detector(image_gray, 1)
#     print(faces)
    if len(rects) > 0:
        rect = max(rects, key=lambda rect: rect.width() * rect.height())
        return rect
    else:
        return None