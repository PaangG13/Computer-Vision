# Computer Vision: Object Detection
## 1. Abstract  
  The task of object detection is to recognize multiple objects based on the input image, locate different objects and provide bounding boxes. The mainstream object detection algorithms are mainly based on deep learning models, which can be divided into two categories: two-state detection algorithm and one-state detection algorithm.

## 2.Data sets
  In data-analysis, I conducted a statistical analysis on the types and number of labels in the train dataset. The training dataset contains a total of 2031 images. The results are shown in the following table. Among them, one image may contain multiple objects that need to be detected.
  <div align="center">

| Category  | Number |
| ---------- | -----------|
| Bird  | 589 |
| Bus  | 430 |
| Car  | 1379 |
| Cat  | 634 |
| Dog  | 815 |

</div>

<p align="center">
  Table 1 Result of statistical analysis
</p>


## 3. Method
  RetinaNet consists of three parts: ResNet, Feature Pyramid Net (FPN), and subnet.
  
  ### 3.1 Ablation studies

  Among them, the default model's parameter settings are: 20 training epochs, learning rate equals to 1e-4, using the ResNet-50 model.

  <div align="center">

|Name|	mAP	|Notes|
| ---------- | -----------| -----------|
|A	|0.328	|Default|
|B	|0.349	|Batch size=4|
|C	|0.418	|Learning rate=1e-5|
|D	|0.466	|depth 101 & learning rate=1e-5|


</div>

<p align="center">
  Table 2 Result of statistical analysis
</p>


## 4. Result




![?](https://github.com/PaangG13/Reinforcement-Learning/blob/main/ouput.gif "Result")
