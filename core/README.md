## Introduction

![](../assets/03.jpeg)
This is a DensNet  which contains a [SE](https://arxiv.org/abs/1709.01507) (Squeeze-and-Excitation Networks by Jie Hu, Li Shen and Gang Sun) module.
Using densenet as backbone, I add senet module into densenet as pic shows below, but it's not the whole structure of se_densenet. 

![](../assets/02.png)

Please click my **[blog](http://www.zhouyuangan.cn/2018/11/se_densenet-modify-densenet-with-champion-network-of-the-2017-classification-task-named-squeeze-and-excitation-network/)**  if you want to know more edited se_densenet details. And Chinese version blog is [here](https://zhuanlan.zhihu.com/p/48499356)

## Table of contents

- Experiment on Cifar dataset
- Experiment on my Own datasets
- How to train model
- Conclusion
- Todo
-Cifar数据集的实验研究
-在我自己的数据集上做实验
-如何训练模型
-结论
-待办事项


Before we start, let's known how to test se_densenet first.

```bash
cd core
python3 se_densenet.py
```

And it will print the structure of se_densenet.

Let's input an tensor which shape is (32, 3, 224, 224) into se_densenet

```bash
cd core
python3 test_se_densenet.py
```

It will print ``torch.size(32, 1000)``

## Experiment on Cifar dataset

### core/baseline.py (baseline)

- Train
![](../assets/cifar_densenet121_train_acc.png)
![](../assets/cifar_densenet121_train_loss.png)

- val
![](../assets/cifar_densenet121_val_acc.png)
![](../assets/cifar_densenet121_val_loss.png)

The best val acc is 0.9406 at epoch 98

### core/se_densenet_w_block.py

In this part, I removed some selayers from densenet' ``transition`` layers, pls check [se_densenet_w_block.py](https://github.com/zhouyuangan/SE_DenseNet/blob/master/se_densenet_w_block.py) and you will find some commented code which point to selayers I have mentioned above.
在这部分中，我从densenet的“transition”层中删除了一些selayers，请检查[se_densenet_w_block.py]
你会发现一些注释代码指向我上面提到的selayers。

- train

![](../assets/cifar_se_densenet121_w_block_train_acc.png)
![](../assets/cifar_se_densenet121_w_block_train_loss.png)

- val

![](../assets/cifar_se_densenet121_w_block_val_acc.png)
![](../assets/cifar_se_densenet121_w_block_val_loss.png)

The best acc is 0.9381 at epoch 98.

### core/se_densenet_full.py

Pls check [se_densenet_full.py](https://github.com/zhouyuangan/SE_DenseNet/blob/master/se_densenet_full.py) get more details, I add senet into both denseblock and transition, thanks for [@john1231983](https://github.com/John1231983)'s issue, I remove some redundant code in se_densenet_full.py, check this [issue](https://github.com/zhouyuangan/SE_DenseNet/issues/1) you will know what I say, here is train-val result on cifar-10:
请查收[se_densenet_full.py]要了解更多详细信息，我将senet添加到denseblock和transition中，感谢[@john1231983](https://github.com/john12311983)的问题，
我删除了seu densenet中的一些冗余代码_完整.py, 检查[问题]你会知道我说什么，这是cifar-10上的train val结果：
[问题]：John1231983于2018年11月8日发表评论：您好，感谢您分享实验结果。我检查发现，您在_Dense层中可能有一些多余的代码，这些代码增加了卷积。您在循环中（for）和第一次卷积后添加了它。为什么再次在_Dense_layer中添加seblock？谢谢
所有者 zhouyuangan于2018年11月9日发表评论      @ John1231983嗨，我发现了您所指出的冗余代码，我认为这是我的错误。为了测试se_densenet，我今天编写了一些脚本来测试cifar10数据集上的se_densenet。我将测试删除的冗余代码是否会影响测试结果。让我们等几天看结果，然后一周内更新README。
作者      John1231983于2018年11月9日发表评论      好的。并考虑在过渡块之后也去除seblock。我们通常仅在密集块之后使用seblock。过渡块仅有助于减小特征尺寸
 1 1    所有者     zhouyuangan于2018年11月10日发表评论     @ John1231983是的，谢谢您的建议，我将接受它并进行更多比较实验。
所有者 zhouyuangan于2018年11月15日发表评论     @ John1231983嗨，约翰。我现在更新测试结果，请检查一下。非常感谢你。         zhouyuangan于2018年11月15日添加了好建议标签
作者      John1231983于2018年11月15日发表评论     做得很好，但是结果表明，使用和不使用seblock都具有相似的性能。用完整的代码在Trans层和Dense层中添加了sebock。如何将它们添加到循环函数中并在_Transition和_Dense层中删除？我的意思是添加
SE_DenseNet/se_densenet_full.py     e5bb2fd中的209行
 ＃self.features.add_module（“ SELayer_％da”％（i + 1），SELayer（channel = num_features））
和       SE_DenseNet / se_densenet_full.py       e5bb2fd中的219行
 ＃self.features.add_module（“ SELayer_％db”％（i + 1），SELayer（channel = num_features））
并删除它           SE_DenseNet / se_densenet_full.py        e5bb2fd中的167行
 self.add_module（“ selayer”，SELayer（channel = num_input_features））
SE_DenseNet / se_densenet_full.py       e5bb2fd中的137行        self.add_module（“ selayer”，SELayer（channel = num_input_features）），
谢谢       1 1        所有者     zhouyuangan于2018年11月15日发表评论     @ John1231983值得进行测试，我将在完成工作后更新新结果，请继续观察。谢谢。
 zhouyuangan更改标题出什么问题了？ se_densnet的其他变体架构需要在2018年11月15日进行尝试和测试
 zhouyuangan于2018年11月15日添加了增强标签         所有者         zhouyuangan于2018年11月17日发表评论     新的测试结果已更新。          我将在几天后发布培训和测试代码。
作者          John1231983于2018年11月17日发表评论                 好的。这就是我的期望。您还可以尝试以下操作：
仅在密集块之后，Sebock才进入循环（在传输过程中删除seblock）    仅在过渡块之后才在循环中执行Seblock（在密集块中删除seblock）   
实际上，我们不知道seblock对密集网络体系结构有何好处。我认为第一种情况可能会取得更好的结果

- train

![](../assets/cifar_se_densenet121_full_train_acc.png)
![](../assets/cifar_se_densenet121_full_train_loss.png)

- val

![](../assets/cifar_se_densenet121_full_val_acc.png)
![](../assets/cifar_se_densenet121_full_val_loss.png)

The best acc is 0.9407 at epoch 86.

### core/se_densenet_full_in_loop.py

Pls check [se_densenet_full_in_loop.py](https://github.com/zhouyuangan/SE_DenseNet/blob/master/se_densenet_full_in_loop.py) get more details, and this [issue](https://github.com/zhouyuangan/SE_DenseNet/issues/1#issuecomment-438891133) illustrate what I have changed, here is train-val result on cifar-10:

- train

![](../assets/cifar_se_densenet121_full_in_loop_train_acc.png)
![](../assets/cifar_se_densenet121_full_in_loop_train_loss.png)

- val

![](../assets/cifar_se_densenet121_full_in_loop_val_acc.png)
![](../assets/cifar_se_densenet121_full_in_loop_val_loss.png)

The best acc is 0.9434 at epoch 97.

### Result

|network|best val acc|epoch|
|--|--|--|
|``densenet``|0.9406|98|
|``se_densenet_w_block``|0.9381|98|
|``se_densenet_full``|0.9407|**86**|
|``se_densenet_full_in_loop``|**0.9434**|97|

## Experiment on my Own datasets


### core/baseline.py (baseline)

- train
![](../assets/densenet121_train_acc.png)
![](../assets/densenet121_train_loss.png)

- val
![](../assets/densenet121_val_acc.png)
![](../assets/densenet121_val_loss.png)

The best acc is: 98.5417%

### core/se_densenet.py

- train

![](../assets/se_densenet121_train_acc.png)
![](../assets/se_densenet121_train_loss.png)

- val

![](../assets/se_densenet121_val_acc.png)
![](../assets/se_densenet121_val_loss.png)

The best acc is: 98.6154%

### Result

|network|best train acc|best val acc|
|--|--|--|
|``densenet``|0.966953|0.985417|
|``se_densenet``|**0.967772**|**0.986154**|

``Se_densenet`` has got **0.0737%** higher accuracy than ``densenet``. I didn't train and test on public dataset like cifar and coco, because of low capacity of machine computation, you can train and test on cifar or coco dataset by yourself if you have the will.
``。我没有在cifar和coco之类的公共数据集上进行训练和测试，因为机器计算能力较低，如果有意愿，您可以自己在cifar或coco数据集上进行训练和测试。

## How to train model

- Download dataset

Cifar10 dataset is easy to access it's website and download it into `data/cifar10`, you can refer to pytorch official tutorials about how to train on cifar10 dataset.

- Training

There are some modules in `core` folder, before you start training, you need to edit code in `cifar10.py` file to change `from core.se_densenet_xxxx import se_densenet121`,

Then, open your terminal, type:
```bash
python cifar10.py
```

- Visualize training

In your terminal, type:

```bash
python visual/viz.py
```
Note: please change your state file path in /visual/viz.py.

### Conclusion

- ``se_densenet_full_in_loop`` gets **the best accuracy** at epoch 97.
-  ``se_densenet_full`` performs well because of less epoch at 86,and it gets ``0.9407`` the second accuracy.
- In the contrast, both ``densenet`` and ``se_densenet_w_block`` get their own the highest accuracy are ``98`` epoch.
-“seu densenet\u full\u in\u loop”在epoch 97获得**最佳精度**。
-“seu densenet\u full”在86秒时由于历元少而表现出色，获得了“0.9407”的第二精度。
-相比之下，``densenet``和``se\u densenet\u w\u block``得到了它们自己的最高精度是``98``epoch。

## TODO

I will release my training code on github as quickly as possible.

- [x] Usage of my codes
- [x] Test result on my own dataset
- [x] Train and test on ``cifar-10`` dataset
- [x] Release train and test code
