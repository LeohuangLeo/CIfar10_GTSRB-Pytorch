# CIfar10_GTSRB-Pytorch

## Environment
[Pytorch](https://pytorch.org/)

GTX1080
## Training method
### Model
[DenseNet](https://arxiv.org/abs/1608.06993)

[EfficientNet](https://arxiv.org/abs/1905.11946)
### Optimizer
[diffGrad](https://arxiv.org/abs/1909.11015)

[Ranger](https://medium.com/@lessw/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d) ([lockAhead](https://arxiv.org/abs/1907.08610) + [RAdam](https://arxiv.org/pdf/1908.03265.pdf))
## Datasets
[The German Traffic Sign Recognition Benchmark(GTSRB)](http://benchmark.ini.rub.de/?section=gtsrb&subsection=news)

[cifar-10](https://www.cs.toronto.edu/~kriz/cifar.html)

the source code of "gtsrb_dataset.py" is from: https://github.com/tomlawrenceuk/GTSRB-Dataloader. This is dataloader for GTSRB

## Resutls

### Cifar-10 [Efficient-b7+Image Augementation+Ranger] test accuracy = 92.57%
![image](https://github.com/LeohuangLeo/CIfar10_GTSRB-Pytorch/blob/master/image/cifarbest.png)
