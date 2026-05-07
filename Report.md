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

作为对比的Top-1和Top-3准确率还有mAP数据

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
    <td>small(21M)</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>0.5480/0.5900/0.5667</td>
  </tr>
  <tr>
    <td>small+(29M)</td>
    <td>/</td>
    <td>/</td>
    <td>/</td>
    <td>0.5560/0.6040/0.5760</td>
  </tr>
  <tr>
    <td>base(86M)</td>
    <td>0.4540/0.5000</td>
    <td>0.4620/0.5100</td>
    <td>0.6100/0.6400</td>
    <td>0.6380/0.6660/0.6510</td>
  </tr>
  <tr>
    <td>large(300M)</td>
    <td>0.6240/0.6720</td>
    <td>0.6120/0.6540</td>
    <td>0.6880/0.7160</td>
    <td>0.6940/0.7260/0.7090</td>
  </tr>
</tbody>
</table>

## 第三周(2月15日)

发现上周的数据集处理流程的问题，不小心往测试集里塞进去了一半左右没有正例的图片，正确修改后结果已经更正，基本和[帖子](https://medium.com/aimonks/clip-vs-dinov2-in-image-similarity-6fa5aa7ed8c6)差不多。接下来打算计算CLIP在SMVS和DISC21上的指标。

下面是dinoV3仅使用embedding作为图像特征在SMVS数据集上Top-1/Top-3结果。

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Canon</th>
    <th>Droid</th>
    <th>E63</th>
    <th>Palm</th>
  </tr></thead>
<tbody>
  <tr>
    <td>small(21M)</td>
    <td>0.35/0.52</td>
    <td>0.44/0.63</td>
    <td>0.35/0.51</td>
    <td>0.26/0.39</td>
  </tr>
  <tr>
  </tr>
    <td>small+(29M)</td>
    <td>0.41/0.57</td>
    <td>0.52/0.65</td>
    <td>0.44/0.62</td>
    <td>0.23/0.44</td>
  <tr>
    <td>base(86M)</td>
    <td>0.55/0.67</td>
    <td>0.64/0.75</td>
    <td>0.58/0.71</td>
    <td>0.32/0.48</td>
  </tr>
  <tr>
    <td>large(300M)</td>
    <td>0.53/0.72</td>
    <td>0.60/0.76</td>
    <td>0.61/0.77</td>
    <td>0.28/0.46</td>
  </tr>
  <tr>
    <td>huge+(840M)</td>
    <td>0.24/0.39</td>
    <td>0.57/0.70</td>
    <td>0.15/0.26</td>
    <td>0.10/0.20</td>
  </tr>
</tbody>
</table>

以下是CLIP模型提取的特征在DISC21数据集上Top-1/Top-3的结果。

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>top-1</th>
    <th>top-3</th>
    <th>mAP</th>
  </tr></thead>
<tbody>
  <tr>
    <td>xsmall(8M)</td>
    <td>0.2520</td>
    <td>0.2820</td>
    <td>0.2657</td>
  </tr>
  <tr>
    <td>medium(38M)</td>
    <td>0.2940</td>
    <td>0.3480</td>
    <td>0.3190</td>
  <tr>
    <td>base(86M)</td>
    <td>0.2960</td>
    <td>0.3280</td>
    <td>0.3097</td>
  </tr>
  <tr>
    <td>large(304M)</td>
    <td>0.3020</td>
    <td>0.3560</td>
    <td>0.3250</td>
  </tr>
</tbody>
</table>

以下是CLIP模型提取的特征在SMVS数据集的business_cards分类上Top-1/Top-3的结果
<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Canon</th>
    <th>Droid</th>
    <th>E63</th>
    <th>Palm</th>
  </tr></thead>
<tbody>
  <tr>
    <td>xsmall(8M)</td>
    <td>0.46/0.65</td>
    <td>0.39/0.59</td>
    <td>0.16/0.40</td>
    <td>0.17/0.31</td>
  </tr>
  <tr>
    <td>medium(38M)</td>
    <td>0.91/0.98</td>
    <td>0.90/0.95</td>
    <td>0.81/0.97</td>
    <td>0.36/0.54</td>
  </tr>
  <tr>
    <td>base(86M)</td>
    <td>0.91/0.97</td>
    <td>0.94/0.99</td>
    <td>0.87/0.96</td>
    <td>0.37/0.49</td>
  </tr>
  <tr>
    <td>large(304M)</td>
    <td>0.96/1.00</td>
    <td>0.96/1.00</td>
    <td>0.94/1.00</td>
    <td>0.53/0.63</td>
  </tr>
</tbody>
</table>

## 第四周(3月8日)

这周主要目标是补完所有的实验结果。

### DinoV3-small(21M)

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Top-1</th>
    <th>Top-3</th>
    <th>mAP</th>
  </tr></thead>
<tbody>
  <tr>
    <td>book_covers</td>
    <td>0.5421</td>
    <td>0.6980</td>
    <td>0.6118</td>
  </tr>
  <tr>
    <td>business_cards</td>
    <td>0.3525</td>
    <td>0.5152</td>
    <td>0.4221</td>
  </tr>
  <tr>
    <td>cd_covers</td>
    <td>0.5275</td>
    <td>0.6650</td>
    <td>0.5850</td>
  </tr>
  <tr>
    <td>dvd_covers</td>
    <td>0.3425</td>
    <td>0.4700</td>
    <td>0.3992</td>
  </tr>
  <tr>
    <td>landmarks</td>
    <td>0.5469</td>
    <td>0.7585</td>
    <td>0.6397</td>
  </tr>
  <tr>
    <td>museum_paintings</td>
    <td>0.4725</td>
    <td>0.6209</td>
    <td>0.5421</td>
  </tr>
  <tr>
    <td>print</td>
    <td>0.3200</td>
    <td>0.4575</td>
    <td>0.3792</td>
  </tr>
  <tr>
    <td>video_frames</td>
    <td>0.7650</td>
    <td>0.8575</td>
    <td>0.8063</td>
  </tr>
  <tr>
    <td>AVERAGE</td>
    <td>0.4836</td>
    <td>0.6300</td>
    <td>0.5482</td>
  </tr>
</tbody>
</table>

### DinoV3-small+(29M)

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Top-1</th>
    <th>Top-3</th>
    <th>mAP</th>
  </tr></thead>
<tbody>
  <tr>
    <td>book_covers</td>
    <td>0.4851</td>
    <td>0.6287</td>
    <td>0.5524</td>
  </tr>
  <tr>
    <td>business_cards</td>
    <td>0.4000</td>
    <td>0.5700</td>
    <td>0.4771</td>
  </tr>
  <tr>
    <td>cd_covers</td>
    <td>0.5450</td>
    <td>0.6900</td>
    <td>0.6058</td>
  </tr>
  <tr>
    <td>dvd_covers</td>
    <td>0.3775</td>
    <td>0.5075</td>
    <td>0.4358</td>
  </tr>
  <tr>
    <td>landmarks</td>
    <td>0.5349</td>
    <td>0.7186</td>
    <td>0.6164</td>
  </tr>
  <tr>
    <td>museum_paintings</td>
    <td>0.4643</td>
    <td>0.6209</td>
    <td>0.5288</td>
  </tr>
  <tr>
    <td>print</td>
    <td>0.3125</td>
    <td>0.4825</td>
    <td>0.3858</td>
  </tr>
  <tr>
    <td>video_frames</td>
    <td>0.7675</td>
    <td>0.8850</td>
    <td>0.8167</td>
  </tr>
  <tr>
    <td>AVERAGE</td>
    <td>0.4859</td>
    <td>0.6379</td>
    <td>0.5524</td>
  </tr>
</tbody>
</table>

### DinoV3-base(86M)

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Top-1</th>
    <th>Top-3</th>
    <th>mAP</th>
  </tr></thead>
<tbody>
  <tr>
    <td>book_covers</td>
    <td>0.6535</td>
    <td>0.7921</td>
    <td>0.7162</td>
  </tr>
  <tr>
    <td>business_cards</td>
    <td>0.5225</td>
    <td>0.6525</td>
    <td>0.5804</td>
  </tr>
  <tr>
    <td>cd_covers</td>
    <td>0.6900</td>
    <td>0.7850</td>
    <td>0.7329</td>
  </tr>
  <tr>
    <td>dvd_covers</td>
    <td>0.4825</td>
    <td>0.6175</td>
    <td>0.5408</td>
  </tr>
  <tr>
    <td>landmarks</td>
    <td>0.5749</td>
    <td>0.7984</td>
    <td>0.6747</td>
  </tr>
  <tr>
    <td>museum_paintings</td>
    <td>0.5027</td>
    <td>0.6538</td>
    <td>0.5664</td>
  </tr>
  <tr>
    <td>print</td>
    <td>0.4775</td>
    <td>0.6150</td>
    <td>0.5337</td>
  </tr>
  <tr>
    <td>video_frames</td>
    <td>0.8675</td>
    <td>0.9500</td>
    <td>0.9029</td>
  </tr>
  <tr>
    <td>AVERAGE</td>
    <td>0.5964</td>
    <td>0.7330</td>
    <td>0.6560</td>
  </tr>
</tbody>
</table>

### DinoV3-large(300M)

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Top-1</th>
    <th>Top-3</th>
    <th>mAP</th>
  </tr></thead>
<tbody>
  <tr>
    <td>book_covers</td>
    <td>0.5198</td>
    <td>0.6782</td>
    <td>0.5912</td>
  </tr>
  <tr>
    <td>business_cards</td>
    <td>0.5050</td>
    <td>0.6775</td>
    <td>0.5813</td>
  </tr>
  <tr>
    <td>cd_covers</td>
    <td>0.6300</td>
    <td>0.7500</td>
    <td>0.6846</td>
  </tr>
  <tr>
    <td>dvd_covers</td>
    <td>0.3875</td>
    <td>0.4950</td>
    <td>0.4363</td>
  </tr>
  <tr>
    <td>landmarks</td>
    <td>0.5429</td>
    <td>0.7545</td>
    <td>0.6357</td>
  </tr>
  <tr>
    <td>museum_paintings</td>
    <td>0.5714</td>
    <td>0.7115</td>
    <td>0.6277</td>
  </tr>
  <tr>
    <td>print</td>
    <td>0.4075</td>
    <td>0.5725</td>
    <td>0.4800</td>
  </tr>
  <tr>
    <td>video_frames</td>
    <td>0.9150</td>
    <td>0.9575</td>
    <td>0.9354</td>
  </tr>
  <tr>
    <td>AVERAGE</td>
    <td>0.5599</td>
    <td>0.6996</td>
    <td>0.6216</td>
  </tr>
</tbody>
</table>

### DinoV3-huge+(840M)

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Top-1</th>
    <th>Top-3</th>
    <th>mAP</th>
  </tr></thead>
<tbody>
  <tr>
    <td>book_covers</td>
    <td>0.4282</td>
    <td>0.4851</td>
    <td>0.4538</td>
  </tr>
  <tr>
    <td>business_cards</td>
    <td>0.2650</td>
    <td>0.3875</td>
    <td>0.3171</td>
  </tr>
  <tr>
    <td>cd_covers</td>
    <td>0.2500</td>
    <td>0.3300</td>
    <td>0.2850</td>
  </tr>
  <tr>
    <td>dvd_covers</td>
    <td>0.2275</td>
    <td>0.3625</td>
    <td>0.2867</td>
  </tr>
  <tr>
    <td>landmarks</td>
    <td>0.4511</td>
    <td>0.6886</td>
    <td>0.5559</td>
  </tr>
  <tr>
    <td>museum_paintings</td>
    <td>0.5879</td>
    <td>0.7418</td>
    <td>0.6566</td>
  </tr>
  <tr>
    <td>print</td>
    <td>0.4050</td>
    <td>0.5225</td>
    <td>0.4575</td>
  </tr>
  <tr>
    <td>video_frames</td>
    <td>0.8100</td>
    <td>0.8450</td>
    <td>0.8254</td>
  </tr>
  <tr>
    <td>AVERAGE</td>
    <td>0.4281</td>
    <td>0.5454</td>
    <td>0.4797</td>
  </tr>
</tbody>
</table>

### CLIP-xsmall(8M)

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Top-1</th>
    <th>Top-3</th>
    <th>mAP</th>
  </tr></thead>
<tbody>
  <tr>
    <td>book_covers</td>
    <td>0.3416</td>
    <td>0.5198</td>
    <td>0.4229</td>
  </tr>
  <tr>
    <td>business_cards</td>
    <td>0.2950</td>
    <td>0.4875</td>
    <td>0.3779</td>
  </tr>
  <tr>
    <td>cd_covers</td>
    <td>0.3625</td>
    <td>0.5575</td>
    <td>0.4483</td>
  </tr>
  <tr>
    <td>dvd_covers</td>
    <td>0.3150</td>
    <td>0.5125</td>
    <td>0.4012</td>
  </tr>
  <tr>
    <td>landmarks</td>
    <td>0.2974</td>
    <td>0.4611</td>
    <td>0.3699</td>
  </tr>
  <tr>
    <td>museum_paintings</td>
    <td>0.5797</td>
    <td>0.7390</td>
    <td>0.6506</td>
  </tr>
  <tr>
    <td>print</td>
    <td>0.1500</td>
    <td>0.3125</td>
    <td>0.2208</td>
  </tr>
  <tr>
    <td>video_frames</td>
    <td>0.7100</td>
    <td>0.8750</td>
    <td>0.7825</td>
  </tr>
  <tr>
    <td>AVERAGE</td>
    <td>0.3814</td>
    <td>0.5581</td>
    <td>0.4593</td>
  </tr>
</tbody>
</table>

### CLIP-medium(38M)

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Top-1</th>
    <th>Top-3</th>
    <th>mAP</th>
  </tr></thead>
<tbody>
  <tr>
    <td>book_covers</td>
    <td>0.5866</td>
    <td>0.7153</td>
    <td>0.6407</td>
  </tr>
  <tr>
    <td>business_cards</td>
    <td>0.7450</td>
    <td>0.8600</td>
    <td>0.7992</td>
  </tr>
  <tr>
    <td>cd_covers</td>
    <td>0.6875</td>
    <td>0.7575</td>
    <td>0.7175</td>
  </tr>
  <tr>
    <td>dvd_covers</td>
    <td>0.7575</td>
    <td>0.8275</td>
    <td>0.7892</td>
  </tr>
  <tr>
    <td>landmarks</td>
    <td>0.3513</td>
    <td>0.5449</td>
    <td>0.4391</td>
  </tr>
  <tr>
    <td>museum_paintings</td>
    <td>0.7692</td>
    <td>0.8956</td>
    <td>0.8265</td>
  </tr>
  <tr>
    <td>print</td>
    <td>0.4300</td>
    <td>0.6275</td>
    <td>0.5167</td>
  </tr>
  <tr>
    <td>video_frames</td>
    <td>0.9175</td>
    <td>0.9650</td>
    <td>0.9392</td>
  </tr>
  <tr>
    <td>AVERAGE</td>
    <td>0.6556</td>
    <td>0.7742</td>
    <td>0.7085</td>
  </tr>
</tbody>
</table>

### CLIP-base(86M)

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Top-1</th>
    <th>Top-3</th>
    <th>mAP</th>
  </tr></thead>
<tbody>
  <tr>
    <td>book_covers</td>
    <td>0.7005</td>
    <td>0.7698</td>
    <td>0.7323</td>
  </tr>
  <tr>
    <td>business_cards</td>
    <td>0.7725</td>
    <td>0.8525</td>
    <td>0.8083</td>
  </tr>
  <tr>
    <td>cd_covers</td>
    <td>0.7575</td>
    <td>0.8325</td>
    <td>0.7921</td>
  </tr>
  <tr>
    <td>dvd_covers</td>
    <td>0.8225</td>
    <td>0.8850</td>
    <td>0.8508</td>
  </tr>
  <tr>
    <td>landmarks</td>
    <td>0.2754</td>
    <td>0.4671</td>
    <td>0.3580</td>
  </tr>
  <tr>
    <td>museum_paintings</td>
    <td>0.5027</td>
    <td>0.7253</td>
    <td>0.5957</td>
  </tr>
  <tr>
    <td>print</td>
    <td>0.4125</td>
    <td>0.5375</td>
    <td>0.4675</td>
  </tr>
  <tr>
    <td>video_frames</td>
    <td>0.9475</td>
    <td>0.9800</td>
    <td>0.9629</td>
  </tr>
  <tr>
    <td>AVERAGE</td>
    <td>0.6489</td>
    <td>0.7562</td>
    <td>0.6959</td>
  </tr>
</tbody>
</table>

### CLIP-large(304M)

<table style="margin: auto"><thead>
  <tr>
    <th></th>
    <th>Top-1</th>
    <th>Top-3</th>
    <th>mAP</th>
  </tr></thead>
<tbody>
  <tr>
    <td>book_covers</td>
    <td>0.7921</td>
    <td>0.8713</td>
    <td>0.8271</td>
  </tr>
  <tr>
    <td>business_cards</td>
    <td>0.8475</td>
    <td>0.9075</td>
    <td>0.8758</td>
  </tr>
  <tr>
    <td>cd_covers</td>
    <td>0.7825</td>
    <td>0.8625</td>
    <td>0.8200</td>
  </tr>
  <tr>
    <td>dvd_covers</td>
    <td>0.9200</td>
    <td>0.9550</td>
    <td>0.9350</td>
  </tr>
  <tr>
    <td>landmarks</td>
    <td>0.2994</td>
    <td>0.4910</td>
    <td>0.3826</td>
  </tr>
  <tr>
    <td>museum_paintings</td>
    <td>0.6319</td>
    <td>0.8269</td>
    <td>0.7202</td>
  </tr>
  <tr>
    <td>print</td>
    <td>0.4675</td>
    <td>0.6275</td>
    <td>0.5329</td>
  </tr>
  <tr>
    <td>video_frames</td>
    <td>0.9125</td>
    <td>0.9725</td>
    <td>0.9404</td>
  </tr>
  <tr>
    <td>AVERAGE</td>
    <td>0.7067</td>
    <td>0.8143</td>
    <td>0.7543</td>
  </tr>
</tbody>
</table>
