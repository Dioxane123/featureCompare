# 对于视觉大模型特征性能比较的实验

主要对比DINOv3和CLIP模型特征的性能，通过在SMVS数据集上进行图像内容检索进行性能比较。

## 第一周(2月1日)

利用了timm提供的DINOv3模型。该模型有两种输出模式:embedding模式和feature模式。前者对每张图片仅输出一个768维的张量，后者对每张图片会输出三个[B, 768, 16, 16]的张量，分别对应ViT不同层数输出的特征图结果。
在feature模式下，为了得到可以直接利用的张量结果，我对每张图片的三个特征图先做了16x16的平均池化，随后进行了L2正则化。
对于embedding模式输出的张量我也进行了L2正则化。
以下是初步结果。

### 利用第一层特征图得到查询向量Top@5结果

<table style="margin: auto">
<thead>
  <tr>
    <th></th>
    <th>Canon</th>
    <th>Droid</th>
    <th>E63</th>
    <th>Palm</th>
  </tr></thead>
<tbody>
  <tr>
    <td>base(86M)</td>
    <td>0.56</td>
    <td>0.69</td>
    <td>0.67</td>
    <td>0.33</td>
  </tr>
  <tr>
    <td>large(300M)</td>
    <td>0.59</td>
    <td>0.73</td>
    <td>0.70</td>
    <td>0.33</td>
  </tr>
  <tr>
    <td>huge+(840M)(CPU)</td>
    <td>0.57</td>
    <td>0.69</td>
    <td>0.64</td>
    <td>0.32</td>
  </tr>
</tbody>
</table>

### 利用第二层特征图得到查询向量Top@5结果

<table><thead>
  <tr>
    <th></th>
    <th>Canon</th>
    <th>Droid</th>
    <th>E63</th>
    <th>Palm</th>
  </tr></thead>
<tbody>
  <tr>
    <td>base(86M)</td>
    <td>0.57</td>
    <td>0.67</td>
    <td>0.67</td>
    <td>0.35</td>
  </tr>
  <tr>
    <td>large(300M)</td>
    <td>0.61</td>
    <td>0.76</td>
    <td>0.70</td>
    <td>0.41</td>
  </tr>
  <tr>
    <td>huge+(840M)(CPU)</td>
    <td>0.59</td>
    <td>0.83</td>
    <td>0.59</td>
    <td>0.34</td>
  </tr>
</tbody>
</table>

## 利用第三层特征图得到查询向量Top@5结果

<table><thead>
  <tr>
    <th></th>
    <th>Canon</th>
    <th>Droid</th>
    <th>E63</th>
    <th>Palm</th>
  </tr></thead>
<tbody>
  <tr>
    <td>base(86M)</td>
    <td>0.68</td>
    <td>0.75</td>
    <td>0.74</td>
    <td>0.42</td>
  </tr>
  <tr>
    <td>large(300M)</td>
    <td>0.68</td>
    <td>0.80</td>
    <td>0.78</td>
    <td>0.51</td>
  </tr>
  <tr>
    <td>huge+(840M)</td>
    <td>0.70</td>
    <td>0.75</td>
    <td>0.67</td>
    <td>0.45</td>
  </tr>
</tbody>
</table>

### 直接利用embedding作为查询向量Top@5结果

<table><thead>
  <tr>
    <th></th>
    <th>Canon</th>
    <th>Droid</th>
    <th>E63</th>
    <th>Palm</th>
  </tr></thead>
<tbody>
  <tr>
    <td>base(86M)</td>
    <td>0.68</td>
    <td>0.75</td>
    <td>0.74</td>
    <td>0.42</td>
  </tr>
  <tr>
    <td>large(300M)</td>
    <td>0.68</td>
    <td>0.80</td>
    <td>0.78</td>
    <td>0.51</td>
  </tr>
  <tr>
    <td>huge+(840M)</td>
    <td>0.70</td>
    <td>0.75</td>
    <td>0.67</td>
    <td>0.45</td>
  </tr>
</tbody>
</table>

因为每个子类别里面只有100张图片，所以结果仅供参考。我认为这个结果偏低了，因为在数据集的[原始论文](https://web.stanford.edu/~bgirod/pdfs/ACMMMSys2011_VisualSearchDataset.pdf)中，作者通过一些传统方法得到的特征在该项目上得到了匹配率略低于90%的结果。
但是在一篇[帖子](https://medium.com/aimonks/clip-vs-dinov2-in-image-similarity-6fa5aa7ed8c6)中，展示了DINOv2在另一个数据集上确实拿不到特别高的指标。

> 在更多的数据集上测试先，我们只是测试视觉基础模型的能力。或者这个特征再结合一下传统方法提取的特征，然后试试看效果

## 第二周(2月8日)

找了[DISC21](https://arxiv.org/pdf/2106.09672)数据集打算进行进一步实验。因为这个比赛早就结束了，找不到官方的数据集下载链接，只能找到[kaggle上的别人自用子集](https://www.kaggle.com/datasets/hmendonca/disc21)。子集文件结构如下：

```bash
.
├── dev_ground_truth.csv
├── dev_queries
│   └── images
│       └── queries  [50000 entries exceeds filelimit, not opening dir]
├── disc21_testset_yfcc_attributions.csv
├── disc21_yfcc_attributions.csv
├── final_ground_truth.csv
├── final_queries
│   └── images
│       └── final_queries  [50000 entries exceeds filelimit, not opening dir]
├── metadata_final_10k.csv
├── references_0
│   └── images
│       └── references  [50000 entries exceeds filelimit, not opening dir]
└── train_19
    └── images
        └── train  [50000 entries exceeds filelimit, not opening dir]

```

其中`final_ground_truth.csv`中仅有500张图片对应的reference在`reference_0`中。因此这一次我的queries由500张有真值的图片组成，refrences由全部50000张图片组成。由于`huge+(840M)`参数规模的模型在本地跑时间过长因此这次仅仅计算了`base(86M)`和`large(300M)`的Top-5.

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>feature0</th>
    <th>feature1</th>
    <th>feature2</th>
    <th>embedding</th>
  </tr></thead>
<tbody>
  <tr>
    <td>base(86M)</td>
    <td>0.5160</td>
    <td>0.5220</td>
    <td>0.6480</td>
    <td>0.6780</td>
  </tr>
  <tr>
    <td>large(300M)</td>
    <td>0.6800</td>
    <td>0.6640</td>
    <td>0.7240</td>
    <td>0.7320</td>
  </tr>
</tbody>
</table>

这个数据集上可以明显看出随着图片查找难度变大和样本量变大，对于feature图提取得到的特征，层数越深效果越好，但都比不过直接embedding得到的特征。同时对于相同方法得到的特征，模型参数量越大效果越好。

作为对比的Top-1和Top-3准确率数据

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>feature0</th>
    <th>feature1</th>
    <th>feature2</th>
    <th>embedding</th>
  </tr></thead>
<tbody>
  <tr>
    <td>base(86M)</td>
    <td>0.4540/0.5000</td>
    <td>0.4620/0.5100</td>
    <td>0.6100/0.6400</td>
    <td>0.6380/0.6660</td>
  </tr>
  <tr>
    <td>large(300M)</td>
    <td>0.6240/0.6720</td>
    <td>0.6120/0.6540</td>
    <td>0.6880/0.7160</td>
    <td>0.6940/0.7260</td>
  </tr>
</tbody>
</table>

## 第三周(2月15日)

发现上周的数据集处理流程的问题，不小心往测试集里塞进去了一半左右没有正例的图片，正确修改后结果已经更正，基本和[帖子](https://medium.com/aimonks/clip-vs-dinov2-in-image-similarity-6fa5aa7ed8c6)差不多。接下来打算计算CLIP在SMVS和DISC21上的指标。

以下是CLIP模型提取的特征在DISC21数据集上Top-1/Top-3的结果。

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>top-1</th>
    <th>top-3</th>
  </tr></thead>
<tbody>
  <tr>
    <td>base(86M)</td>
    <td>0.2960</td>
    <td>0.3280</td>
  </tr>
  <tr>
    <td>large(300M)</td>
    <td>0.3020</td>
    <td>0.3560</td>
  </tr>
</tbody>
</table>

发现CLIP在SMVS上数据特别好，同时DINOv3上数据和先前测量不一致。后面需要重新计算指标
