# Multi-model Entity Resolution

## 相关工作

1. [swj2018_product_data.pdf (heikopaulheim.com)](http://www.heikopaulheim.com/docs/swj2018_product_data.pdf)(2018)

   related work中综述了单独使用属性信息和单独使用图像的方法

   * method

     ![image-20221110140619635](E:\project\paper\survey\image-20221110140619635.png)

   * dataset

     * Yahoo’s Gemini Product Ads (GPA) for supervision
     * Unstructured Product Offers - WDC Microdata Dataset

2. [WDC-EC_GS.pdf (webdatacommons.org)](http://webdatacommons.org/productcorpus/paper/WDC-EC_GS.pdf)

   

1. [The WDC Training Dataset and Gold Standard for Large-Scale Product Matching (acm.org)](https://dl.acm.org/doi/pdf/10.1145/3308560.3316609)

1. [Holistic evaluation in multi-model databases benchmarking (springer.com)](https://link.springer.com/content/pdf/10.1007/s10619-019-07279-6.pdf)（2019）

1. [paper_T2.pdf (openproceedings.org)](https://openproceedings.org/2020/conf/edbt/paper_T2.pdf)（2020）

   提到了Entity Resolution的Challenges and Final Remarks中就包含Multi-modal Entity Resolution

1. [GvDB21_multi_modal_product_matching.pdf (uni-leipzig.de)](https://dbs.uni-leipzig.de/file/GvDB21_multi_modal_product_matching.pdf)(2021)

   截止到这篇工作之前，还没有一个包含产品图像和描述的公开数据集

   这篇工作的主要贡献就是收集了一个数据集，并且提出了一个多模态的方法

   * 多模态方面的综述

     [Trends in Integration of Vision and Language Research: A Survey of Tasks, Datasets, and Methods | Journal of Artificial Intelligence Research (jair.org)](https://www.jair.org/index.php/jair/article/view/11688)
   
   * 数据集：
   
     * 使用：extend WDC dataset 未开源 自己爬图片
   
       [Web Data Commons](http://webdatacommons.org/)
   
     * 提到：dataset for the SIGIR eCom 2020
   
       [SIGIR 2020 E-Commerce Workshop Data Challenge (sigir-ecom.github.io)](https://sigir-ecom.github.io/files/Rakuten_Data_Challenge.pdf)
   
       数据集地址 [Challenge data (ens.fr)](https://challengedata.ens.fr/participants/challenges/35/) 约等于只有一句话描述和图片
   
   * 方法
   
     * 参考了SIGMOD2018 [Deep Learning for Entity Matching: A Design Space Exploration (wisc.edu)](https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf)
   
       ![image-20221110004137468](E:\project\paper\survey\image-20221110004137468.png)
   
       ![image-20221110003140714](E:\project\paper\survey\image-20221110003140714.png)
   
7. [An E-Commerce Dataset in French for Multi-modal Product Categorization and Cross-Modal Retrieval (researchgate.net)](https://www.researchgate.net/publication/350420878_An_E-Commerce_Dataset_in_French_for_Multi-modal_Product_Categorization_and_Cross-Modal_Retrieval)（2021）

   使用的是法国乐天集团的challenge

8. [FULLTEXT01.pdf (diva-portal.org)](https://www.diva-portal.org/smash/get/diva2:1591884/FULLTEXT01.pdf)（2021）

9. [p2459-thirumuruganathan.pdf (vldb.org)](http://vldb.org/pvldb/vol14/p2459-thirumuruganathan.pdf)（Proc. VLDB Endow. 2021)

10. [Applied Sciences | Free Full-Text | Leveraging Multi-Modal Information for Cross-Lingual Entity Matching across Knowledge Graphs | HTML (mdpi.com)](https://www.mdpi.com/2076-3417/12/19/10107/htm)（2022）

11. Effective Deep Learning Based Multi-Modal Retrieval
    [vldbj-msae.pdf (nus.edu.sg)](https://www.comp.nus.edu.sg/~ooibc/vldbj-msae.pdf)

12. Entity Matching综述（2021）

    [TKDD1503-52 (acm.org)](https://dl.acm.org/doi/pdf/10.1145/3442200)

    [JDIQ1301-01 (acm.org)](https://dl.acm.org/doi/pdf/10.1145/3431816)

13. Entity Resolution 综述

    [An Overview of End-to-End Entity Resolution for Big Data (archives-ouvertes.fr)](https://hal.archives-ouvertes.fr/hal-02955445/document)

14. Multi-Modal Coreference Resolution with the Correlation between Space Structures （2018）

    [1804.08010.pdf (arxiv.org)](https://arxiv.org/pdf/1804.08010.pdf)

15. Visual Entity Linking via Multi-modal Learning（2021）

    [dint_a_00114.pdf (silverchair.com)](https://watermark.silverchair.com/dint_a_00114.pdf?token=AQECAHi208BE49Ooan9kkhW_Ercy7Dm3ZL_9Cf3qfKAc485ysgAAAsQwggLABgkqhkiG9w0BBwagggKxMIICrQIBADCCAqYGCSqGSIb3DQEHATAeBglghkgBZQMEAS4wEQQMvHCG75IePLhIwDGqAgEQgIICd-882tLBHvuqOzPnf2cCLyzMtLOlvoxXh7OBJLr79nDf8tOK1wI5Gyy8WyY37dKElzLqmmpffObRIU5MtAXTnUMyE5hf9czjCdIWZSSm37OVI7-k7WOyrtLFTFnnJn4RrUwibCWVT3DWEjKcOKmF5c0xL7zmmWyWb4-6otskkmOQoOGDUoEA0xmX0p5lH1Jrs-8y-C2VYjkvt6OkNR6xEJJ2SU24vmtxO3EpVbBYem8FX_sOlDrlNAQcf5QL24WAsJNRsKBaszROYcWCPk0cOpKU408ihne8oAuoaLwvB51cSfr3vaHsG4bMo3CS7OYawzAwzLTKfsaoUiuEPAlLgrmTPeKZRpIuwL_kkFC9MS7d7rd8AB3BfAwki9v4mZSAlIpef0N-w-qBwGpriR4wptJoRpzcKGQwxWAdJzcHHy4jT-A3l4Fvf9yZi3PCmPfWu5So9KicKo3sXyDME7V6qKO11yJ6LUqa1eRiwWhXMShyL4xEdplzbpX8W5NkhTeAk22wNW5GTVecx-dLzlhKbpV9Ww5zydA67DVNO1K75F_Fek78nF715b7pA7ypk14eb51eTKI17B4VbyGhG9B5xIo5iolpLw0i-6Cr43vUUZ5Cu6NT4ChGxaRbuSAo86V8OADAOWDShA65VmprWRNCxmsT1nosu9aguGigXz3qfuwsJYXpTRZSRypuMiLcUVFR1For1morW6P1A83U8e00OQd0sn-gq6g8YhGdTzBoiDe3-5JBpBWvys8R460WZD7UMbQsMakyOmaaAC1UnV35YvnTQiBpXjo-s5ge0X_2PKHFtudJivSVtUtCjS9hi5KEh4-RPh8B5Dk)

16. Bridging the Gap between Reality and Ideality of Entity Matching: A Revisiting and Benchmark Re-Construction（2022）

    [2205.05889.pdf (arxiv.org)](https://arxiv.org/pdf/2205.05889.pdf)

17. MMKG: Multi-modal Knowledge Graphs （2019）

    [MMKG: Multi-modal Knowledge Graphs (lihui.info)](https://lihui.info/doc/ESWC19.pdf)

18. MMKGR: Multi-hop Multi-modal Knowledge Graph Reasoning （2022）

    [2209.01416.pdf (arxiv.org)](https://arxiv.org/pdf/2209.01416.pdf)

19. 

## 数据集

1. [Shopee - Price Match Guarantee | Kaggle](https://www.kaggle.com/competitions/shopee-product-matching/data)

   kaggle比赛 数据集也是一句描述+一张图片

2. [中文大规模多模态评测基准MUGE_数据集-阿里云天池 (aliyun.com)](https://tianchi.aliyun.com/dataset/107332)

   阿里达摩院多模态查询的测评

3. multi-modal dataset

   [多模态分析数据集（Multimodal Dataset）整理 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/189876288)

4. multi-modal structured dataset

   [2110.02577v1.pdf (arxiv.org)](https://arxiv.org/pdf/2110.02577v1.pdf)

5. 
