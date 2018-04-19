import mAP

p, r, AP = mAP.eval_mAP('/home/hzzone/Hand-Keypoint-Detection/output/iter_1000/egohands.csv', '/home/hzzone/Hand-Keypoint-Detection/data/gth/egohands.csv')
print(AP)