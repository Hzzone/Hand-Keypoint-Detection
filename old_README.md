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

### 数据集
手势识别包括Hand detection和Hand keypoint detection两个问题。首先将手从原图片中提取出，然后针对特征点进行回归，因此需要两种类型的数据集完成问题。
1. Hand detection数据集，用一个矩形分割出图片中的手的位置，标注了矩形在原始图片中的坐标。
2. Hand keypoint detection数据集，标注了包括掌心、关节、指尖等关键点的坐标。

因此，我们采用的数据集如下：  
#### Hand detection数据集  
* [Hand Dataset by Arpit Mittal, Andrew Zisserman and Phil Torr](http://www.robots.ox.ac.uk/~vgg/data/hands/) 
这个数据集从各种不同的公共图像数据集源收集的手图像的全面数据集。总共有13050个实例被注释。大于固定框包围面积（1500平方像素）的手实例被认为是“足够大”用于检测并用于评估，给出了大约4170个高质量的手工实例。在收集数据时，没有对人的姿势或能见度施加限制，也没有对环境施加任何限制。在每个图像中，所有能被人类清晰感知的手都有注释。注解由一个包围矩形组成，它不必是轴向对齐的，而是面向手腕的。

<table border="1" cellpadding="2" cellspacing="0" width="80%">
<tbody>
  <tr>
    <td colspan="3" align="center">
      <b>Training Dataset</b>
    </td>
    <td colspan="3" align="center">
      <b>Validation Dataset</b>
    </td>
  </tr>
  <tr>
    <td align="center">
      <b>Source</b>
    </td>
    <td align="center">
      <b>#Instances</b>
    </td>
    <td align="center">
      <b>#Big Instances</b>
    </td>
    <td align="center">
      <b>Source</b>
    </td>
    <td align="center">
      <b>#Instances</b>
    </td>
    <td align="center">
      <b>#Big Instances</b>
    </td>
  </tr>
  <tr>
    <td>
      <a href="http://www.robots.ox.ac.uk/~vgg/data/stickmen/index.html">
        <font color="black">Buffy Stickman</font></a>
    </td>
    <td align="center">887</td>
    <td align="center">438</td>
    <td>Movie Dataset*</td>
    <td align="center">1856</td>
    <td align="center">649</td></tr>
  <tr>
    <td>
      <a href="http://pascal.inrialpes.fr/data/human/">
        <font color="black">INRIA pedestrian</font></a>
    </td>
    <td align="center">1343</td>
    <td align="center">137</td>
    <td>
      <i>Total</i>
    </td>
    <td align="center">1856</td>
    <td align="center">649</td></tr>
  <tr>
    <td>
      <a href="http://www.eecs.berkeley.edu/~lbourdev/poselets/">
        <font color="black">Poselet (H3D)</font></a>
    </td>
    <td align="center">1355</td>
    <td align="center">580</td>
    <td colspan="3" align="center">
      <b>Test Dataset</b>
    </td>
  </tr>
  <tr>
    <td>Skin Dataset [2]</td>
    <td align="center">703</td>
    <td align="center">139</td>
    <td align="center">
      <b>Source</b>
    </td>
    <td align="center">
      <b>#Instances</b>
    </td>
    <td align="center">
      <b>#Big Instances</b>
    </td>
  </tr>
  <tr>
    <td>
      <a href="http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/">
        <font color="black">PASCAL VOC 2007 train and val set
          <font></font></font>
      </a>
    </td>
    <td align="center">1867</td>
    <td align="center">507</td>
    <td>
      <a href="http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2007/">
        <font color="black">PASCAL VOC 2007 test set</font></a>
    </td>
    <td align="center">1626</td>
    <td align="center">562</td></tr>
  <tr>
    <td width="28%">
      <a href="http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2010/">
        <font color="black">PASCAL VOC 2010 train and val set (except human layout set)</font></a>
    </td>
    <td align="center">3008</td>
    <td align="center">1060</td>
    <td width="20%">
      <a href="http://pascallin.ecs.soton.ac.uk/challenges/VOC/voc2010/">
        <font color="black">PASCAL VOC 2010 human layout val set</font></a>
    </td>
    <td align="center">405</td>
    <td align="center">98</td></tr>
  <tr>
    <td>
      <i>Total</i>
    </td>
    <td align="center">9163</td>
    <td align="center">2861</td>
    <td>
      <i>Total</i>
    </td>
    <td align="center">2031</td>
    <td align="center">660</td></tr>
</tbody>
</table>
数据集预览：
<div align="center">
    <img src="http://omoitwcai.bkt.clouddn.com/2017-12-21-Picture_1.png">
</div>  

* [VIVA Hand Detection Dataset](http://cvrr.ucsd.edu/vivachallenge/index.php/hands/hand-detection/)   
该数据集由2D的bounding box标注司机和乘客的手。由54个在自然驾驶中收集的视频组成，包括照明的变化，大的手运动，和普遍的遮挡问题。一些数据由平台拍摄，还有一些是由YouTube提供。
数据集预览：  
<div align="center">
    <img src="http://omoitwcai.bkt.clouddn.com/2017-12-21-example-1024x576.png">
</div>
#### Hand keypoint detection数据集 
* [CMU Hand Database](http://domedb.perception.cs.cmu.edu/handdb.html)
该数据集由CMU从不同公开数据集进行采集，并进行人工标记手的关键点。并且通过能够容纳关键点的放大的矩形来生成更多的Hand detection数据集。
数据集预览：  
<div align="center">
    <img src="http://omoitwcai.bkt.clouddn.com/2017-12-21-fig_hand_manual-1.jpg">
</div>
### 预处理   
我们主要使用Hand detection数据集，Hand keypoint detection数据集。其中Hand detection数据集包含人手边框标注数据，主要用于检测任务的训练；Hand keypoint detection数据集包含边框标注数据和关键点信息，主要用于关键点的训练。训练集分为四种:负样本，正样本，部分样本，关键点样本. 三个样本的比例为$3:1:1:2$。   
#### 正负样本，部分样本提取    
1. 从Hand detection数据集随机选出边框，然后和标注数据计算$IOU$，如果大于$0.65$，则为正样本，大于$0.4$小于$0.65$为部分样本，小于$0.4$为负样本。   
$IOU$: 简单来讲就是模型产生的目标窗口和原来标记窗口的交叠率。具体我们可以简单的理解为： 即检测结果(DetectionResult)与Ground Truth的交集比上它们的并集，即为检测的准确率IOU，公式如下：   
$$ IOU = \frac{DetectionResult \bigcap Ground Truth }{DetectionResult \bigcup Ground Truth} $$
2. 计算边框偏移．对于边框，$(x1,y1)$为左上角坐标，$(x2,y2)$为右下角坐标，新剪裁的边框坐标为
$(xn1,yn1)$, $(xn2,yn2)$, $width$, $height$。则 
$$ offset x1 = (x1 - xn1)/width$$
同上，计算另三个点的坐标偏移．   

3. 对于正样本，部分样本均有边框信息，而对于负样本不需要边框信息   

#### 关键点样本提取    
从Hand keypoint detection数据集中提取，可以根据标注的边框，在满足正样本的要求下，随机裁剪出图片，然后调整关键点的坐标。   

### 路线    
#### loss修改   
由于训练过程中需要同时计算３个loss，但是对于不同的任务，每个任务需要的loss不同，所以在整理数据中，对于每个图片进行15个label的标注信息：   
1. 第1列：为正负样本标志，1正样本，0负样本，2部分样本，3关键点信息 
2. 第2-5列：为边框偏移，为float类型，对于无边框信息的数据，全部置为-1 
3. 第6-15列：为关键点偏移，为floagt类型，对于无边框信息的数据，全部置为-1  

标注好label之后，在训练过程中，采取以下措施： 
1. 自定义softmax_loss，增加判断，只对于1,0计算loss值。  
2. 自定义euclidean_loss，增加判断，对于置为-1的不进行loss计算。 
3. Hard Example选择，在进行人脸分类任务时，采用了在线困难样本选择，即在训练过程中，根据计算出的loss值，进行排序，只对于70%的值较低的数据，进行反向传播。 
  
#### 网络描述 
分为三个阶段，分别是classifier、boundingbox regression和landmarks detection 
1. stage1: 在构建图像金字塔的基础上，利用fully convolutional   network来进行检测，同时利用boundingbox regression和非极大值抑制（NMS）来合并高度重叠的候选框。在这一步获得了手的区域的候选窗口和边界框的回归向量，并用该边界框做回归，对候选窗口进行了校准。  
2. stage2: 将通过stage1的所有窗口输入作进一步判断，同时也通过boundingbox regression和 NMS去掉那些false-positive区域。 
3. stage3: 作用和stage2相似，但是stage3对手的区域进行了更多的监督和更强的约束即手的关键点，因此在stage3还会输出手的关键点。  
