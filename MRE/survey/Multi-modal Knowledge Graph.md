# Multi-modal Knowledge Graph

## Multi-modal link prediction

### Dataset

1. WN18-IMG

   [Translating Embeddings for Modeling Multi-relational Data (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)

2. FB15K-237-IMG

   [Translating Embeddings for Modeling Multi-relational Data (neurips.cc)](https://proceedings.neurips.cc/paper_files/paper/2013/file/1cecc7a77928ca8133fa24680a88d2f9-Paper.pdf)

   [wangmengsd/RSME (github.com)](https://github.com/wangmengsd/RSME)

### Baseline

1. IKRL
2. TransAE
3. RSME

## Multi-modal Relation Extraction

### Dataset

1. MNRE

   Paper : [IEEE Xplore Full-Text PDF:](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9428274)

   Code : [thecharm/MNRE: Resource and Code for ICME 2021 paper "MNRE: A Challenge Multimodal Dataset for Neural Relation Extraction with Visual Evidence in Social Media Posts" (github.com)](https://github.com/thecharm/MNRE)
   
   说是提供了一些baseline方法，其实就是简单魔改了之前的一些工作，并且baseline代码没有开源。
   
   这个数据集作者指出的一个特点就是，句子的长度一般不长，所以文本提供的信息可能比较有限，想要解决的问题也是在这种短句子下，通过图像信息来增加收益。

### Baseline

1. BERT+SG

   Paper : [Multimodal Relation Extraction with Efficient Graph Alignment (acm.org)](https://dl.acm.org/doi/pdf/10.1145/3474085.3476968)

   Code : [thecharm/Mega: Code for ACM MM 2021 Paper "Multimodal Relation Extraction with Efficient Graph Alignment". (github.com)](https://github.com/thecharm/Mega)

2. MEGA

   Paper : [Multimodal Relation Extraction with Efficient Graph Alignment (acm.org)](https://dl.acm.org/doi/pdf/10.1145/3474085.3476968)

   Code : [thecharm/Mega: Code for ACM MM 2021 Paper "Multimodal Relation Extraction with Efficient Graph Alignment". (github.com)](https://github.com/thecharm/Mega)
   
   以上两个方法都来自同一个工作，该工作主要是发现了在社交媒体的post中，文本比较短，噪声影响比较大。
   
   motivation应该是希望能够从图像中不仅仅学习到object和entity之间的关系，还要学习到object之间的关系，然后和文本建模的关系之间进行对齐。
   
   

## Multi-modal Named Entity Recognition

### Dataset

1. Twitter-2017

   [Visual Attention Model for Name Tagging in Multimodal Social Media (illinois.edu)](https://blender.cs.illinois.edu/paper/multimediaentity.pdf)

### Baseline

1. AdapCoAtt-BERT-CRF

   [Adaptive Co-Attention Network for Named Entity Recognition in Tweets (acm.org)](https://dl.acm.org/doi/pdf/10.5555/3504035.3504731)

2. UMT

   [Improving Multimodal Named Entity Recognition via Entity Span Detection with Unified Multimodal Transformer (aclanthology.org)](https://aclanthology.org/2020.acl-main.306.pdf)

3. UMGF

   [Multi-modal Graph Fusion for Named Entity Recognition with Targeted Visual Guidance | Proceedings of the AAAI Conference on Artificial Intelligence](https://ojs.aaai.org/index.php/AAAI/article/view/17687)



## MEGA

### Code

[thecharm/Mega: Code for ACM MM 2021 Paper "Multimodal Relation Extraction with Efficient Graph Alignment". (github.com)](https://github.com/thecharm/Mega)

### Motivation

1. 社交媒体的post中文本比较短，缺少上下文信息，因此如何解决短文本下的关系抽取，考虑借助post中的图像信息进行补充

2. 相比于多模态命名实体识别MNER问题，MRE不仅仅要将text中的entity和image中的object对应起来，还应该借助图像生成object之间的relation作为参考信息

   感觉这个motivation可以考虑对图像生成描述，然后建立描述和文本之间的对应，这样是不是可以直接借用CLIP的架子

### Method

#### Pipeline

![image-20230326131630058](E:\project\paper\MRE\survey\image-20230326131630058.png)

1. 首先是使用BERT对text的文本语义信息进行表征，然后是对image生scene graph，其中scene graph中包含了object的特征以及object之间的relation，并且将object特征作为视觉语义特征。
2. 对输入文本生成语义依赖树，描述了文本信息的结构化语义，并且scene graph中object的视觉语义特征也可以结构化表示
3. 将之前抽取的结构化特征和语义信息进行对齐，来抽取多模态信息之间的相关表征
4. 将文本表征和视觉表征拼接起来，预测所属关系

#### 语义特征表示

##### 文本语义表征

对输入的文本进行tokenize，然后在开头结尾加上[CLS]和[SEP]。此外还需要用[E1start]、[E1end]、[E2start]、[E2end]来标识出entity。然后使用[PAD]将长度对齐到$l$。同样，采用一种类似于掩码的token sequence来表示哪些是有效token，长度也为$l$。

![image-20230324154003065](E:\project\paper\MRE\survey\image-20230324154003065.png)

![image-20230324154014422](E:\project\paper\MRE\survey\image-20230324154014422.png)

![image-20230324154020413](E:\project\paper\MRE\survey\image-20230324154020413.png)

最终将$X \in R^{l\times d_x}$

##### 视觉语义表征

对于视觉语义特征的抽取，首先将图片输入到基于Faster R-CNN的scene graph生成模型中，在scene graph中，每一个节点表示object特征，每一条表示object之间的relation。由于图像中可能会抽取出多个object，为了不引入噪声，只选取最显著的m个object。如果不足就用0填充。最终能够获取到视觉语义特征矩阵$T\in R^{m\times d_y}$

![image-20230324154659051](E:\project\paper\MRE\survey\image-20230324154659051.png)

#### 结构特征表示

为了能够更好提供一些结构化的信息，分别使用语义依赖树和场景图生成模型来构建文本和图像的关系图。

##### 语义依赖树

使用ELMo来获取一句文本中各个单词之间的关系。

![image-20230324155359863](E:\project\paper\MRE\survey\image-20230324155359863.png)

![image-20230324155444806](E:\project\paper\MRE\survey\image-20230324155444806.png)

V表示的是单词的集合，E表示的是单词之间的关系集合，G表示文本语义依赖树对应的图结构

##### 场景图生成

使用之前的场景图生成模型就能够完成这部分的工作

![image-20230324160249104](E:\project\paper\MRE\survey\image-20230324160249104.png)

#### 多模态特征对齐

##### 图结构的对齐

这部分的方法就是参考[REGAL: Representation Learning-based Graph Alignment (arxiv.org)](https://arxiv.org/pdf/1802.06257.pdf)这里感觉不太合适，虽然只是考虑图的结构性，但是关系图是有严格方向的，感觉应该采用一些有向图相似性计算的方法。

首先将节点集合取并集，感觉就是直接做concatenate，然后不考虑方向，计算每个节点和K步内邻居数的和，然后使用这个值计算相似性。

![image-20230325111736193](E:\project\paper\MRE\survey\image-20230325111736193.png)

![image-20230325111742124](E:\project\paper\MRE\survey\image-20230325111742124.png)

然后随机挑选$p$个点作为代表，使用这$p$个点对两个图的结构进行近似表征。这里的目的是要得到对于两个图中的点的一个统一的表征，这个可以近似为

![image-20230325113119171](E:\project\paper\MRE\survey\image-20230325113119171.png)

![image-20230325113128205](E:\project\paper\MRE\survey\image-20230325113128205.png)

![image-20230325113203488](E:\project\paper\MRE\survey\image-20230325113203488.png)

然后使用得到的关于两个图节点的一致表征，计算节点之间的相似性，然后选取相似性最高的保留，其余的置0，得到对齐矩阵$\alpha$

![image-20230325113410096](E:\project\paper\MRE\survey\image-20230325113410096.png)

##### 语义特征对齐

对于抽取到的文本语义特征和视觉语义特征，分别使用如下三个投影关系做变化$W_k \in R^{l\times d_a}$，$W_Q \in R^{m\times d_a}$ 和 $W_v \in R^{m\times d_a}$
$$
K = W_k X + b_k\\
Q = W_q Y + b_q\\
V = W_v X + b_v
$$
然后得到模态之间的cross-attetion的结果
$$
Y_s = CrossAttn(K,Q,V) = softmax(\frac{QK^T}{\sqrt{d}})V
$$

##### 对齐融合

将图结构对齐和模态cross-attention进行融合得到最终每个object的特征$Y^* \in R^{m\times d_a}$
$$
Y^*=\alpha^TV+Y_s
$$

#### 实体表征拼接

对所有object的特征进行求和，来表示最终抽取出来的视觉特征
$$
\hat{y} = \sum_{i=1}^my_i^*
$$
然后分别用$v_{[E1_{start}]}$和$v_{[E2_{start}]}$来表示两个实体抽取出来的文本特征
$$
\hat{v} = [v_{[E1_{start}]},v_{[E2_{start}]}]
$$
最终整体的特征表示为
$$
z=concat(\hat{v},\hat{y})
$$
分类方式如下，$res\in R^{1\times n_c}$
$$
res=softmax(MLP(z))
$$

### Experiment

![image-20230325135116496](E:\project\paper\MRE\survey\image-20230325135116496.png)

### 可以借鉴的点

首先，对于图像的整体信息一定是要利用的，FL-MSRE原始工作中，只使用了人物的面部信息，这一定是不够的。

其次，在这个工作中，对于图像语义的场景图，并没有考虑关系之间的方向性，而实际中关系是有着严格方向性的，因此可以看看如何在有向图之间进行相似性匹配。

后面再去看看最近的场景图的生成工作，以及对于有向图相似匹配的工作

## MTB

### Code

[plkmo/BERT-Relation-Extraction: PyTorch implementation for "Matching the Blanks: Distributional Similarity for Relation Learning" paper (github.com)](https://github.com/plkmo/BERT-Relation-Extraction)

### Motivation

实际上关系抽取关心的更多的是text中的关系，而不是text中的实体到底是什么，因此在训练的过程中，我们应该引导模型去学习关系描述到关系的映射，而不是学习实体到关系的映射。因此这篇工作采用了一种使用[BLANK]标记的预训练方式，采用对比学习的方式，拉近相同关系之间的距离，而拉远不同关系之间的距离。

### Method

首先对于特征的抽取表达方式，文章中总共探索了六种不同的组合，最终选用了增加实体标签并且采用每个实体的embedding进行拼接的方式作为抽取的特征，也就是图中的最后一种。

![image-20230326160008866](E:\project\paper\MRE\survey\image-20230326160008866.png)

使用 $f_{\theta}$ 来表示特征编码得到的结果，那么两个text是否相似采用如下方式进行衡量
$$
p(l=1|r,r')=\frac{1}{\exp{(f_{\theta}(r)\cdot f_{\theta}(r'))}}
$$
损失函数采用类似于交叉熵的形式，其中$\delta$用来表示两个实体是否一致
$$
L(D)=\frac{1}{|D|^2}\sum_{(r,e_1,e_2)\in D}\sum_{(r',e'_1,e'_2)\in D}\delta_{e_1,e'_1}\delta_{e_2,e'_2}\cdot \log{p(l=1|r,r')} \\
+ (1-\delta_{e_1,e'_1}\delta_{e_2,e'_2})\cdot \log{(1-p(l=1|r,r'))}
$$
在训练的过程中，输入的关系描述中，将标注出来的entity以一定的概率标记为[BLANK]，这样一来就强制模型学习到的是text到relation的映射，而不是entity pair到relation的映射关系。这个预训练的过程是他们自己收集的一个数据集，这里不太清楚做中文任务有没有合适的数据集。

### Experiment

实验结果可以充分证实，在小样本学习下，使用[BLANK]替换实体的方式，能够更好地学习到text到relation的映射关系

![image-20230326161956456](E:\project\paper\MRE\survey\image-20230326161956456.png)

### 可以借鉴的点

考虑在训练的过程中，弱化掉对于entity的描述，比如说在text中将其mask掉，或者是在图像中将人脸那部分也加入噪声，但是感觉人脸这块还是挺重要的，不应该直接扔掉，它能够提供一些细节信息。

## RSN : Relational Siamese Networks

### Motivation

在开放的数据域上，会存在很多之前从来没有出现过的关系，因此实际上训练使用的数据集和真正开放数据域可能存在比较大的差异，RSN就是希望能够从有监督的数据中，学习到可以迁移的关系知识，进而能够更好解决开放数据域上的问题。还提出了基于半监督和远程监督的强化方式。

### Method

#### Relational Siamese Network

![image-20230327003306339](E:\project\paper\MRE\survey\image-20230327003306339.png)

这篇工作出的比较早，当时还没有Bert，所以encoder还是用的CNN。然后直接计算两个向量之间元素级的距离绝对值向量，输入到sigmod激活函数中，得到两个向量是否相似，相似则说明这两句话描述的是同一种关系。损失函数就可以直接使用交叉熵损失函数。

#### 可以借鉴的点

暂时没get到这篇工作在小样本学习上的点，但是他提出了对于support set，其中心点的表示不应该简单采用平均的方式。

