### Roadmap of hand keypoint detection
* 第一步
分为三个网络，大小分别是propoasl-net: 12，refine-net: 24，output-net: 48。propoasl-net和refine-net使用检测手的数据集，output-net使用手部关键点数据集。   
12，24输出之后生成hard example和原来的数据集一起作为下一个网络的输入，具体思路如下:
12-net(生成12大小的数据集) ---> 24-net(12-net生成的hard example+生成的24大小的数据集) ---> 48-net
* 生成数据

生成数据的时候，有一个ground truth, 高或宽小于40的话，我就认为那不是一只手，是错的标签。在12-net, 24-net输出时使用NMS（非极大值抑制）去除重复框，可以既减少计算量。  
[mtcnn-caffe](https://github.com/CongWeilin/mtcnn-caffe)的复现里自定义了data层，我不希望这样做，我会生成hdf5文件，这样更灵活，可以加入测试、训练阶段。 

* 标签

所有在整理数据中，对于每个图片进行了15个label的标注信息：    

1. 第1列：为正负样本标志，１正样本, 0负样本,-1部分样本,3关键点信息

2. 第2-5列：为边框偏移，为float类型，对于无边框信息的数据，全部置为-1

3. 42列：为关键点偏移，为float类型，对于无边框信息的数据，全部置为-1    
 

> 修改softmax_loss_layer.cpp　增加判断，只对于1,0计算loss值
修改euclidean_loss_layer.cpp　增加判断，对于置为-1的不进行loss计算

换句话说，hdf5文件里有四块区域，除了data，还有label(标注正负部分样本), points(21个关键点,所以有42列), roi(边框信息，两个点，四列) 。

* 自定义层
1. 自定义一个fc层，只对标签不是-1的进行向前推进，这样就不区分到底是第几个网络，我也不需要写几个文件生成数据集。相当于修改了softmax_loss_layer
2. 自定义euclidean_loss_layer, 同理也不对-1进行计算