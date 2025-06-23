# IIFS
一种面向审计大模型微调的不平衡指令筛选策略<br>
An Imbalanced Instruction Filtering Strategy for  Instruction tuning Audit Large Language Model

## 结构目录
├── step1-dataFiltering.py        # 数据过滤步骤<br>
├── step2-necessityInfer-buildDataset.py  # 必要性推断并构建数据集<br>
├── step3-predictionMergeWithLabel.py    # 预测结果与标签合并<br>
├── step4-necessityInfer-bert_rouge-compute.py  # 使用BERT和ROUGE计算必要性<br>
├── step5-1-similarityScore-compute.py    # 计算相似度得分<br>
├── step5-2-similarityScore-MeanDistribution.py  # 相似度得分的均值分布<br>
├── step5-3-similarityScore-final.py      # 最终相似度得分计算<br>
└── step6-finalSelect.py                  # 最终选择步骤<br>
