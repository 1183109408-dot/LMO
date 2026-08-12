# **README**

\<h1 align="center">

&#x20; 基于小样本多类别自适应任务的大语言模型优化方法

\</h1>

\<p align="center">

&#x20; 段钇作\<sup>1\</sup>

\</p>

\<p align="center">

&#x20; \<sup>1\</sup>上海大学

\</p>

## **🔥 概述**

研究聚焦于大语言模型在**小样本多类别自适应任务**中的优化问题，即如何在标注数据稀缺的条件下，使单一模型同时适配多个知识体系与推理逻辑差异显著的细粒度子领域类别，同时缓解小样本训练中噪声与暴露偏差耦合带来的次生影响。本仓库包含两项研究工作：其一，**融合分类提示与条件输入的参数高效优化框架** ，通过条件化低秩适配与类别专属指令生成，以极少参数增量实现多类别协同适配；其二，**自校准计划采样方法**，通过双解码器架构与 token 级自适应动态概率目标，实现噪声抑制与暴露偏差缓解的平衡。两项工作可分别简称为项目一与项目二。

## **📥 数据与模型**

**模型：**&#x9879;目一模型包含LLama3.1-8B-Instruct、LLama3.2-3B-Instruct、QWen3-8B、MiniCPM5-1B-SFT，项目二模型包含LLama3.1-8B、QWen2.5-Math-7B、deepseek-math-7b-base。实验所用模型皆为标准预训练模型，可自行于huggingface或modelscope下载。

**数据：**&#x9879;目一原始数据集包含ScienceQA、Tabmwp、DeepMath-103K，项目二原始数据集包含Math（测试集Math500）、MathQA、DeepMath-103K，对原始数据集有兴趣的可自行下载。改造后的数据集已上传至百度网盘，链接https://pan.baidu.com/s/1I93cZJVnuY0DWQU45WEQPg，提取码：0000。

项目一每个数据集完成评分后用于扩充数据的指令集也随改造数据集一起上传至百度网盘，文件命名为correct\_samples\_with\_weights.csv。此外，项目一论文实验只展示了测试准确率的平均值与标准差，其原始数据则上传至百度网盘，文件命名为Experiment\_accuracy.xlsx。

### **📦 代码运行**

**项目一**示例代码已上传至CPCP文件夹，示例代码以ScienceQA数据集为基准，运行时可根据需要替换相应超参数。运行步骤如下：

1、前往modelscope下载 **bert-base-cased&#x20;**&#x6A21;型，同时准备好实验测试的模型与数据集。

2、运行ScienceQA\_generator.py，生成指令集并评分筛选，代码中使用的教师模型为deepseek-V3，其中的API key请自行准备。

3、运行scienceQA\_classify.py，训练SVM分类器。

4、运行ScienceQA\_test.py，不经过微调，仅测试添加指令后的准确率。

5、运行ScienceQA\_train.py，微调模型。

6、运行ScienceQA\_finetune\_test.py，测试微调后的准确率。

**项目二**示例代码已上传至Schedule\_Sampling文件夹，示例代码以Math500数据集为基准，运行时可根据需要替换相应超参数。运行步骤如下：

1、准备好实验测试的模型与数据集。

2、运行Math500\_train.py，微调模型。

3、运行Math500\_test.py，测试微调后的准确率。

## **📖 引用**

待定。
