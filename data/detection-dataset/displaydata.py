from variable import *

l1 = []
for each_source in data_sources:
	width_list = []
	height_list = []
	with open(osp.join(each_source, "data.txt")) as f:
		data_list = f.readlines()
		for img in data_list:
			one_im = img.split(" ")
			path = osp.join(each_source, "images/"+one_im[0]+".jpg")
			boxes = np.reshape(map(float, one_im[1:]), (-1, 4))
			# im = cv2.imread(path)
			for (x1, y1, x2, y2) in boxes:
				# cv2.rectangle(im, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
				print path, x1, y1, x2, y2
				width_list.append(x2-x1)
				height_list.append(y2-y1)
			### show image if you want
			# cv2.imshow("test", im)
			# cv2.waitKey(0)
			# cv2.imwrite("hh.jpg", im)
	l1.append(width_list)
	l1.append(height_list)
for i in range(len(l1)):
	plt.subplot(3, 2, i+1)
	hist, bins = np.histogram(l1[i], bins=50)
	width = 0.7 * (bins[1] - bins[0])
	center = (bins[:-1] + bins[1:]) / 2
	plt.bar(center, hist, align='center', width=width)
	plt.title(data_sources[i / 2] + " " + ("width" if i % 2 == 0 else "height"))
	plt.tight_layout()
plt.show()
