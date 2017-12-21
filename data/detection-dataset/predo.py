from variable import *

for each_source in data_sources:
	annotations_source = osp.join(each_source, annotations)
	annotations_list = []
	for mat_file in os.listdir(annotations_source):
		boxes_data = sio.loadmat(osp.join(annotations_source, mat_file))["boxes"].flatten()
		file_name = mat_file.split(".")[0]
		one_annotations = [file_name, ]
		for box in boxes_data:
			tmp = np.reshape(box[0, 0].tolist()[:4], (-1, 2))
			y1 = round(min(tmp[:, 0]), 2)
			y2 = round(max(tmp[:, 0]), 2)
			x1 = round(min(tmp[:, 1]), 2)
			x2 = round(max(tmp[:, 1]), 2)
			one_annotations.append(x1)
			one_annotations.append(y1)
			one_annotations.append(x2)
			one_annotations.append(y2)
		annotations_list.append(" ".join(map(str, one_annotations)))
		print file_name
	with open(osp.join(each_source, "data.txt"), "w") as f:
		f.write("\n".join(annotations_list))
