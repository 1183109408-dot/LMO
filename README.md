LMO包含“融合分类提示与条件输入的参数高效优化框架”与“出自校准计划采样”两个项目，分别对应“CPCP”和“Schedule_Sampling”两个文件夹。

CPCP运行步骤：
1、自行下载数据集，本实验使用的是ScienceQA、Tabmwp、DeepMath103K这三个数据集，使用者也可替换其他数据集，只需对应修改scienceQA_load.py文件。
2、自行下载SLM，本实验使用LLama3.1-8B-Instruct、LLama3.2-3B-Instruct、QWen3-8B、MiniCPM5-1B-SFT模型，可供参考。
3、运行ScienceQA_generator.py，生成指令集并评分筛选，其中LLM的API key请自行更改。本实验使用的LLM为deepseek-V3，用于评分的SLM为LLama3.1-8B-Instruct。
4、运行scienceQA_classify.py，训练SVM分类器。
5、运行ScienceQA_test.py，不经过FineTune，仅测试添加指令后的精度。
6、运行ScienceQA_train.py，训练SLM。
7、运行ScienceQA_finetune_test.py，测试微调后的精度。
Experiment_accuracy.xlsx是本实验测试的原始精度数据，可供参考。

Schedule_Sampling运行步骤：
1、自行下载数据集，本实验使用的是Math、MathQA、DeepMath103K这三个数据集，使用者也可替换其他数据集，只需对应修改Math500_load.py文件。
2、自行下载SLM，本实验使用LLama3.1-8B、QWen2.5-Math-7B、deepseek-math-7b-base模型，可供参考。
3、运行Math500_train.py，训练SLM。
4、运行Math500_test.py，测试微调后的精度。
