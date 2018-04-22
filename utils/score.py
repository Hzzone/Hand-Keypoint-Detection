import mAP
import os

p, r, AP = mAP.eval_mAP('/home/hzzone/Hand-Keypoint-Detection/output/iter_50000/egohands.csv', '/home/hzzone/Hand-Keypoint-Detection/data/gth/egohands.csv')
print(AP)
p, r, AP = mAP.eval_mAP('/home/hzzone/Hand-Keypoint-Detection/output/iter_50000/stanfordhands.csv', '/home/hzzone/Hand-Keypoint-Detection/data/gth/stanfordhands.csv')
print(AP)
# for test_data in ['egohands', 'stanfordhands']:
#     gth_path = '../data/gth/{}.csv'.format(test_data)
#     output_path = [os.path.join('../output', iter_num) for iter_num in os.listdir('../output')]
#     for iter_num_output in output_path:
#         p, r, AP = mAP.eval_mAP('{}/{}.csv'.format(iter_num_output, test_data), gth_path)
#         print(iter_num_output, AP)
