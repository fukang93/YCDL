1. 基于pslite 的DNN框架，详细介绍可见 https://zhuanlan.zhihu.com/p/333847581

2. 依赖的库有：
  boost
  Eigen
  ps-lite
  generator：https://github.com/TheLartians/Generator
  json: https://github.com/nlohmann/json.git
  
3. 依赖库安装
    ./run_build.sh

4. 编译
   ./run.sh build

5. test
  ./run.sh test 

6. 运行demo
  单机LR ./bin/output/lr_uci
  单机DNN ./bin/output/network
  伪分布式LR sh script/local.sh 2 2 ./bin/output/lr_uci_dist
  伪分布式DNN sh script/local.sh 2 2 ./bin/output/network_dist

7. YCDL简介

本文旨在介绍一个基于ps-lite实现的分布式DNN框架（YCDL），与其他开源的大型深度学习框架相比，YCDL非常轻量。希望能对大家学习和开发深度学习框架带来一些帮助。

目前开源的一些深度学习框架已经非常成熟（比如tensorflow、pytorch等），能够灵活支持图像、视频、文本、语音等各种场景。但是对于需要大规模离散id特征的业务场景（比如推荐算法、广告算法等），开源的深度学习框架支持的不是很好。所以工业界许多公司都会自研深度学习框架或者在开源tensorflow上做改写，来支持这种大规模离散id特征的场景。

YCDL是基于ps-lite的分布式DNN框架，适用于推荐或者广告等领域的ctr、cvr模型。其中ps-lite作为底层的参数服务器（parameter server）用来支持大规模的离散id特征。具体关于参数服务器和ps-lite的介绍这里不做赘述。简单来说，参数服务器是一种编程框架，重点支持大规模参数的分布式存储和通信，而ps-lite是参数服务器的一种实现。

下面会详细阐述YCDL的实现细节。

YCDL整体用C++实现，通过CMAKE来管理工程项目；主要依赖库有ps-lite（负责参数的存储和通信）和 Eigen（加速矩阵运算）；模型网络结构由json文件来定义；计算过程包括训练和预测两个过程，其中训练过程包括以下几个步骤：

数据预处理 包括生成批处理的输入数据，各种特征处理（category特征one-hot embedding、连续特征离散化等）
拉取参数 client从ps-lite拉取模型参数
前向传播 依据网络模型结构和输入数据，一层一层做forward计算
损失计算 基于模型输出值和label做运算，得到loss值和梯度
反向传播 依据网络模型结构和梯度值，一层一层做backward计算
更新参数 将更新的梯度值传给ps-lite更新模型参数
预测过程和训练过程类似，只是少了反向传播和更新参数的步骤。YCDL整体的代码结构也是基于上述几个步骤来抽象得到：

dataload 负责生成批处理的输入样本
instance 包含一个样本的label和特征数据
dist_optimizer 包含两个功能，一个是实现client侧的pull/push参数的操作（获取/更新参数），一个是实现 parameter server 侧的模型参数更新的算子（包括sgd、adagrad等参数更新方法）
matrix_val 在client侧对模型参数和梯度进行抽象，并且负责调用optimizer的pull/push算子
layer 负责实现网络层的各种算子（包括全连接层、concat层、activation层、loss层等），其中loss_layer是一种特殊的layer，用于损失计算
network 负责连接各个layer，做一层一层的forward和backward算子调用
