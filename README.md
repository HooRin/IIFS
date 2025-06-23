# IIFS
一种面向审计大模型微调的不平衡指令筛选策略<br>
An Imbalanced Instruction Filtering Strategy for  Instruction tuning Audit Large Language Model

## 结构目录
├── step1-dataFiltering.py                       &nbsp; &nbsp; &nbsp; &nbsp;# 指令冗余性，SIDS算法对高资源任务指令数据进行聚类<br>
├── step2-necessityInfer-buildDataset.py         &nbsp; &nbsp; &nbsp; &nbsp;# 构建适用于Llama-Factory进行模型推理的的指令数据集<br>
├── step3-predictionMergeWithLabel.py            &nbsp; &nbsp; &nbsp; &nbsp;# 对step1的聚类结果及step2的推理结果进行合并<br>
├── step4-necessityInfer-bert_rouge-compute.py   &nbsp; &nbsp; &nbsp; &nbsp;# 基于step3的推理结果运用ROUGE-L与BERTScore指标进行评估<br>
├── step5-1-similarityScore-compute.py           &nbsp; &nbsp; &nbsp; &nbsp;# 计算指令数据与每条核心任务指令数据的相似度<br>
├── step5-2-similarityScore-MeanDistribution.py  &nbsp; &nbsp; &nbsp; &nbsp;# 基于相似度得分的均值分布情况绘制图表寻找拐点<br>
├── step5-3-similarityScore-final.py             &nbsp; &nbsp; &nbsp; &nbsp;# 基于拐点计算最终相似度得分<br>
└── step6-finalSelect.py                         &nbsp; &nbsp; &nbsp; &nbsp;# 计算每条指令的综合质量分，并基于聚类规模自适应采样函数确认高质量子集<br>
