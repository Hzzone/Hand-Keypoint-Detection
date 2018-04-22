import mAP
import os

# p, r, AP = mAP.eval_mAP('/home/hzzone/Hand-Keypoint-Detection/output/iter__iter_80000/egohands.csv', '/home/hzzone/Hand-Keypoint-Detection/data/gth/egohands.csv')
p, r, AP = mAP.eval_mAP('/home/hzzone/Hand-Keypoint-Detection/output/iter__iter_80000/stanfordhands.csv', '/home/hzzone/Hand-Keypoint-Detection/data/gth/stanfordhands.csv')
All_AP = []
for test_data in ['egohands', 'stanfordhands']:
    gth_path = '../data/gth/{}.csv'.format(test_data)
    output_path = [os.path.join('../output', iter_num) for iter_num in os.listdir('../output')]
    for iter_num_output in output_path:
        p, r, AP = mAP.eval_mAP('{}/{}.csv'.format(iter_num_output, test_data), gth_path)
        All_AP.append(AP)

print(All_AP)
