# Zhinao-ChineseModernBert: 面向高吞吐低内存场景的中文基座与向量嵌入模型
<p align="center">
<a href="https://huggingface.co/qihoo360/Zhinao-ChineseModernBert-Embedding"><img src="https://img.shields.io/badge/Hugging%20Face-模型仓库-blue?logo=huggingface"></a>
<a href="https://huggingface.co/spaces/mteb/leaderboard"><img src="https://img.shields.io/badge/CMTEB-Base级SOTA-brightgreen"></a>
</p>

中文 | [English](./README_en.md)

## 项目简介
Zhinao-ChineseModernBert系列是针对**高推理速度要求、严苛内存限制**的工业级场景，从头预训练的中文Base级基座模型与语义嵌入模型。本系列基于ModernBert高效架构与Qwen2Tokenizer分词器，依托超大规模中英文语料完成全流程预训练，在保持Base级参数量（除Embedding外约100M参数）轻量化优势的同时，实现了对同量级模型的全面超越，甚至性能优于更大参数量的主流模型，为中文NLP理解任务、语义检索、向量数据库、RAG检索增强等场景提供高性价比的开箱即用解决方案。

本项目包含两个核心模型：
- **Zhinao-ChineseModernBert**：通用中文理解基座，基于两阶段掩码语言建模(MLM)预训练，支持最长1536序列长度，适配各类中文NLU下游任务。
- **Zhinao-ChineseModernBert-Embedding**：专业中文语义嵌入模型，在基座的基础上做两阶段Embedding训练，支持最长512序列长度，专为语义检索、向量表征、相似度计算等场景深度优化。

---

## 核心亮点
### 1. 高效架构+先进分词体系，兼顾速度与泛化性
- 采用**ModernBert**高效Transformer架构从头预训练，针对高吞吐推理、长文本处理场景做了深度优化，相比传统Bert架构实现显著的推理速度提升，内存占用更友好。
- 适配**Qwen2Tokenizer**大词表分词器，大幅降低中文及中英夹杂文本的OOV（未登录词）率，分词精度更高，对网络用语、专业术语、中英混合场景的适配性更强。

### 2. 超大规模语料预训练，覆盖全场景中文语义
基于**1T Tokens**高质量中英文语料完成预训练，以中文语料为核心（占比超65%），辅以英文语料，全面通用互联网、科技、金融、医疗、法律、教育、代码等多领域场景，模型语义理解能力与跨域泛化性远超同量级模型。

### 3. 同量级领先性能，极致性价比
- **Zhinao-ChineseModernBert**综合性能超越RoBERTa-wwm-large等大参数量模型。
- **Zhinao-ChineseModernBert-Embedding**为CMTEB基准榜单Base级参数量最优模型，综合性能超越Qwen3-Embedding-0.6B等主流大参数量嵌入模型，在检索、聚类、相似度计算等核心任务上表现突出。

---

## 模型详情
| 模型全称 | 核心定位 | 预训练数据规模 | 最大序列长度 | 核心适用场景 | 核心优势 |
|----------|----------|----------------|--------------|--------------|----------|
| Zhinao-ChineseModernBert | 通用中文理解基座 | 1T Tokens 中英文MLM预训练 | 1536 | 文本分类、实体识别、情感分析、关系抽取、长文本理解等各类NLU任务 | Base级参数量，长文本友好，推理速度快，内存占用低 |
| Zhinao-ChineseModernBert-Embedding | 专业中文语义嵌入模型 | 1T Tokens RetroMAE预训练 + 检索预训练 + 中英文MTEB微调 | 512 | 语义检索、向量数据库、RAG检索增强、文本相似度计算、聚类、重排序等场景 | CMTEB Base级参数量 SOTA，向量表征精度高，中英夹杂适配性强，检索性能领先 |

---

## 训练方案
### 一、Zhinao-ChineseModernBert 通用基座预训练
1. **MLM预训练 (1T Tokens)** 本模型基于ModernBert架构，采用Masked Language Modeling （MLM）作为预训练目标，**从头完成全流程预训练**，无任何第三方基座权重依赖。
    - 预训练语料：1T Tokens高质量中英文语料，覆盖通用新闻、百科、书籍、对话、代码、法律、专业文献等多个领域，经过严格的数据清洗、去重、质量过滤，保障预训练语料的多样性与纯净度。
    - 序列长度：预训练阶段以8192长度做数据切片，适配长文本理解场景，解决传统Bert类模型长文本建模能力不足的痛点。
    - 训练优化：采用分布式预训练框架，结合动态WWM掩码策略、混合精度训练、梯度累积等优化手段，保障预训练的稳定性与收敛效果。
2. **RetroMAE 预训练** 
    - 预训练语料：在一阶段语料的基础上，增加下游业务常用的中短长度的文本语料。
    - 序列长度：以自然段落切分语料，最大序列长度为1536。
    - 采用RetroMAE自监督预训练目标，针对句子级语义表征做专项优化，通过双向掩码重建与自编码学习，强化模型对句子整体语义的建模能力，解决传统MLM预训练对句级表征适配性不足的问题。

### 二、Zhinao-ChineseModernBert-Embedding 语义嵌入模型两阶段训练
本模型在Zhinao-ChineseModernBert通用基座的基础上，针对语义表征场景完成两阶段Embedding训练，实现从通用语义理解到精准向量表征的能力跃迁：
1. **检索预训练**
   基于数亿级中英文检索和翻译语料做对比学习预训练，构建正负样本对的语义区分能力，深度适配下游检索、相似度计算等核心场景，大幅提升模型的跨域泛化性。
2. **MTEB中英文全任务微调**
   基于CMTEB与MTEB全维度任务相关数据集，完成多任务有监督微调，对齐检索、聚类、分类、STS、重排序等核心任务的优化目标，进一步提升模型在真实业务场景中的落地效果，达到多场景即开即用的目的。

---

## 性能评测
### 1. CLUE 中文语言理解基准测评
Zhinao-ChineseModernBert在CLUE基准榜单上，实现了以Base级参数量（除Embedding外约100M参数）综合性能超越RoBERTa-wwm-large等大参数量模型，为资源有限的业务场景提供更多可选项。

| model | params | afqmc | tnews | iflytek | cmnli | wsc | csl | ocnli | c3 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ChineseModernBERT(large) | ~310M | 73.87 | 56.90 | 60.15 | 83.96 | 52.10 | 86.20 | 79.10 | 82.65 | 71.87 |
| RoBERTa-wwm-large | ~310M | 76.55 | 58.61 | 62.98 | 82.12 | 74.60 | 82.13 | 78.20 | 73.82 | 73.63 |
| RoBERTa-wwm-ext(base) | ~90M | 74.04 | 56.94 | **60.31** | 80.51 | 67.80 | 81.00 | 74.72 | 66.50 | 70.23 |
| Zhinao-ChineseModernBERT(base) | ~110M | **76.99** | **57.51** | 59.56 | **83.82** | **78.95** | **85.70** | **79.08** | **75.42** | **74.63** |

### 2. CMTEB 中文海量文本嵌入基准测评
Zhinao-ChineseModernBert-Embedding在CMTEB基准榜单上，登顶Base级参数量（除Embedding外约100M参数）模型最优排名，综合性能超越Qwen3-Embedding-0.6B等主流大参数量嵌入模型。

| Model Name | Params | Dimension | Classification | Clustering | Pair Classification | Reranking | Retrieval | STS | Mean Task Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bge-base-zh-v1.5 | ~90M | 1024 | 71.79 | 47.49 | 73.18 | 65.02 | 69.41 | 51.67 | 63.07 |
| piccolo-base-zh | ~90M | 768 | 70.05 | 47.12 | 70.17 | 66.68 | 71.20 | 54.39 | 63.27 |
| setlla-base-zh-v3-1792d | ~90M | 1792 | 74.40 | 53.29 | 82.50 | 67.84 | 72.28 | 61.92 | 68.71 |
| setlla-large-zh-v3.5-1792d | ~310M | 1792 | 74.66 | 54.31 | 82.92 | 68.45 | 73.52 | 51.93 | 69.30 |
| Qwen3-Embedding-0.6B | ~596M | 1024 | 71.4 | 68.74 | 76.42 | 62.58 | 71.03 | 54.52 | 67.45 |
| 360Zhinao-Embedding-Base | ~110M | 768 | 73.50 | 65.36 | 86.15 | 67.99 | 69.32 | 60.29 | **70.16** |

> 注：测评代码与详细测评指标可查看[eval_mteb.py](./eval_mteb.py)与[CMTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard)，模型综合性能实现同量级领先，超越600M参数量的Qwen3-Embedding-0.6B模型。

---

## 快速开始
### 环境依赖
推荐使用Python 3.10+，核心依赖如下：
```bash
pip install torch>=2.6.0 transformers>=4.56.2 sentence-transformers>=5.1.2
```

### 模型下载
我们已将模型权重开源至主流模型平台，可通过以下地址获取：
| 模型 | Hugging Face 地址 | 
|:------:|:-------------------:|
| Zhinao-ChineseModernBert | [🤗](https://huggingface.co/qihoo360/Zhinao-ChineseModernBert) |
| Zhinao-ChineseModernBert-Embedding | [🤗](https://huggingface.co/qihoo360/Zhinao-ChineseModernBert-Embedding) |

### 使用示例
#### 1. Zhinao-ChineseModernBert 通用基座使用示例
基座支持Hugging Face Transformers直接加载，需微调后方可用于下游任务。详见[clue_evaluator.py](./clue_evaluator.py)
```python

```

#### 2. Zhinao-ChineseModernBert-Embedding 语义嵌入模型使用示例

```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer("qihoo360/360Zhinao-Embedding-Base")

# 推荐使用flash attention
# model = SentenceTransformer("qihoo360/360Zhinao-Embedding-Base", 
#                             model_kwargs={"attn_implementation": "flash_attention_2", "dtype": torch.bfloat16})

# 输入文本（原生支持中文、英文、中英混合文本）
sentences = [
    "这是一个中文测试句子",
    "This is an English test sentence",
    "这是一个中英夹杂的test sentence",
]

# 生成语义embedding向量
embeddings = model.encode(sentences, prompt="Instruct: Retrieve semantically similar text.\nQuery: ")

# 输出结果
print(f"生成的向量维度: {embeddings.shape}")
print(embeddings@embeddings.T)
```

#### 语义相似度计算示例
```python
from sentence_transformers import SentenceTransformer, util

# 加载模型
model = SentenceTransformer("qihoo360/360Zhinao-Embedding-Base")

# 待匹配的查询语句与候选文档
query = "什么是语义嵌入模型？"
docs = [
    "语义嵌入模型是将文本转化为高维向量的AI模型，可用于计算文本间的语义相似度",
    "大语言模型是能够生成自然语言文本的AI模型，广泛应用于对话、写作等场景",
    "计算机视觉模型用于处理图像数据，实现图像分类、目标检测等功能",
]

# 生成向量
query_embedding = model.encode(query, prompt="Instruct: Given a Chinese search query, retrieve web passages that answer the question.\nQuery: ")
doc_embeddings = model.encode(docs)

# 计算余弦相似度并输出结果
cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
for i, score in enumerate(cos_scores):
    print(f"文档{i+1} 相似度: {score:.4f} | 内容: {docs[i]}")
```
---

## 模型局限性与使用说明
1. **序列长度限制**：Zhinao-ChineseModernBert支持1536序列长度，理论上可在8192的长度范围内微调，Zhinao-ChineseModernBert-Embedding最大支持512序列长度，超长度文本建议根据场景做合理截断或分块处理。
2. **语言适配范围**：模型以中文预训练为主、英文为辅，核心优化中文及中英夹杂场景，纯英文专业场景建议使用英文原生嵌入模型。
3. **Embedding模型的prompt**：Embedding模型采用了通用的Prompt设置，各任务的prompt见[config_sentence_transformers.json](https://huggingface.co/qihoo360/Zhinao-ChineseModernBert-Embedding/blob/main/config_sentence_transformers.json)，其他任务建议搭配prompt使用。
3. **合规使用**：模型仅可用于合法合规的科研与工业场景，严禁用于危害国家安全、网络暴力、虚假信息传播等违法违规场景，使用时请遵守相关法律法规与开源许可证协议。
4. **免责声明**：模型基于公开语料预训练，我们已尽力保障数据的合规性与模型的安全性，但不对模型输出的内容承担任何法律责任，模型输出不代表我们的立场与观点。

---

## 许可证
本项目模型权重与代码基于 **Apache 2.0 许可证** 开源，可免费用于学术研究与商业用途，详细条款请查看LICENSE文件。

---

## 引用
如果本项目对您的研究或工作有帮助，请引用我们的项目：
```bibtex
@misc{zhinao-chinesemodernbert,
  title={Zhinao-ChineseModernBert: Chinese Foundation & Vector Embedding Model for High-Throughput, Low-Memory Scenarios},
  author={zhinao team},
  year={2026},
  howpublished={\url{https://github.com/your-repo/zhinao-chinesemodernbert}},
}
```

---

## 致谢
- 感谢ModernBert团队开源的高效Transformer架构，为本项目提供了坚实的架构基础
- 感谢Qwen团队开源的Qwen2Tokenizer，为本项目提供了优秀的分词体系
- 感谢CLUE、MTEB、CMTEB提供的权威中文NLP测评基准
- 感谢开源社区为中文NLP发展做出的贡献