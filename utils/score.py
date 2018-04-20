import mAP

# p, r, AP = mAP.eval_mAP('/home/hzzone/Hand-Keypoint-Detection/output/iter__iter_80000/egohands.csv', '/home/hzzone/Hand-Keypoint-Detection/data/gth/egohands.csv')
p, r, AP = mAP.eval_mAP('/home/hzzone/Hand-Keypoint-Detection/output/iter__iter_80000/stanfordhands.csv', '/home/hzzone/Hand-Keypoint-Detection/data/gth/stanfordhands.csv')
print(AP)