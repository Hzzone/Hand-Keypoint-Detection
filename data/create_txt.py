import os
import random
data_sources = ['egohands', 'stanfordhands']
root_dir = os.path.dirname(os.path.abspath(__file__))
test_data = []
train_data = []

for data_source in data_sources:
    test_im_dir = os.path.join(root_dir, data_source, 'test', 'JPEGImages')
    train_im_dir = os.path.join(root_dir, data_source, 'trainval', 'JPEGImages')
    for im_file in os.listdir(test_im_dir):
        name = im_file.rstrip('.jpg')
        xml_file_path = os.path.join(data_source, 'test', 'Annotations', name+'.xml')
        im_file_path = os.path.join(data_source, 'test', 'JPEGImages', name+'.jpg')
        test_data.append(" ".join([im_file_path, xml_file_path]))
    for im_file in os.listdir(train_im_dir):
        name = im_file.rstrip('.jpg')
        xml_file_path = os.path.join(data_source, 'trainval', 'Annotations', name+'.xml')
        im_file_path = os.path.join(data_source, 'trainval', 'JPEGImages', name+'.jpg')
        train_data.append(" ".join([im_file_path, xml_file_path]))



random.shuffle(test_data)
random.shuffle(test_data)
random.shuffle(train_data)
random.shuffle(train_data)

with open('test.txt', 'w') as f:
    f.write('\n'.join(test_data))
with open('trainval.txt', 'w') as f:
    f.write('\n'.join(train_data))
