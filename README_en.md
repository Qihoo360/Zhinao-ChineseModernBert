# Zhinao-ChineseModernBert: A Chinese Foundation & Vector Embedding Model for High-Throughput and Low-Memory Scenarios
<p align="center">
<a href="https://huggingface.co/qihoo360/Zhinao-ChineseModernBert-Embedding"><img src="https://img.shields.io/badge/Hugging%20Face-Model%20Repo-blue?logo=huggingface"></a>
<a href="https://huggingface.co/spaces/mteb/leaderboard"><img src="https://img.shields.io/badge/CMTEB-Base-level%20SOTA-brightgreen"></a>
</p>

[中文](./README.md) | English

## Overview
The Zhinao-ChineseModernBert series consists of **Chinese Base-level foundation models and semantic embedding models** pre-trained from scratch for industrial scenarios with **high inference speed requirements and strict memory constraints**. Built on the efficient ModernBert architecture and Qwen2Tokenizer, this series is fully pre-trained on ultra-large-scale Chinese-English corpora. While retaining the lightweight advantage of Base-level parameters (~100M parameters excluding Embedding), it comprehensively outperforms models of similar scale and even mainstream models with larger parameter counts, providing a cost-effective, out-of-the-box solution for Chinese NLP understanding, semantic retrieval, vector databases, RAG, and other scenarios.

This project includes two core models:
- **Zhinao-ChineseModernBert**: General Chinese understanding foundation model, pre-trained with two-stage Masked Language Modeling (MLM), supporting a maximum sequence length of 1536, suitable for various Chinese NLU downstream tasks.
- **Zhinao-ChineseModernBert-Embedding**: Specialized Chinese semantic embedding model, trained with two-stage embedding fine-tuning based on the foundation model, supporting a maximum sequence length of 512, deeply optimized for semantic retrieval, vector representation, similarity calculation, etc.

---

## Key Features
### 1. Efficient Architecture + Advanced Tokenization, Balancing Speed and Generalization
- Pre-trained from scratch using the **ModernBert** efficient Transformer architecture, deeply optimized for high-throughput inference and long-text processing. It achieves significant inference speed improvements and lower memory footprint compared to traditional Bert architectures.
- Compatible with the **Qwen2Tokenizer** large vocabulary tokenizer, which greatly reduces the out-of-vocabulary (OOV) rate for Chinese and Chinese-English mixed text, delivers higher tokenization accuracy, and better adapts to internet slang, professional terms, and mixed-language scenarios.

### 2. Pre-trained on Ultra-Large-Scale Corpora, Covering Full-Scenario Chinese Semantics
Pre-trained on **1T Tokens** of high-quality Chinese-English corpora, with Chinese as the core (over 65%), supplemented by English. It fully covers general internet, technology, finance, healthcare, law, education, code, and other domains. The model’s semantic understanding and cross-domain generalization far exceed those of similar-scale models.

### 3. Leading Performance at Similar Scale, Ultimate Cost-Effectiveness
- **Zhinao-ChineseModernBert** outperforms large-parameter models such as RoBERTa-wwm-large in overall performance.
- **Zhinao-ChineseModernBert-Embedding** is the top-performing Base-level model on the CMTEB benchmark, surpassing mainstream large-parameter embedding models such as Qwen3-Embedding-0.6B in overall performance, with outstanding results on core tasks including retrieval, clustering, and similarity calculation.

---

## Model Details
| Full Model Name | Core Positioning | Pre-training Data Scale | Max Sequence Length | Core Application Scenarios | Key Advantages |
|-----------------|------------------|------------------------|---------------------|----------------------------|----------------|
| Zhinao-ChineseModernBert | General Chinese Understanding Foundation | 1T Tokens Chinese-English MLM Pre-training | 1536 | Text classification, NER, sentiment analysis, relation extraction, long-text understanding, and other NLU tasks | Base-level parameters, long-text friendly, fast inference, low memory footprint |
| Zhinao-ChineseModernBert-Embedding | Specialized Chinese Semantic Embedding Model | 1T Tokens RetroMAE Pre-training + Retrieval Pre-training + Chinese-English MTEB Fine-tuning | 512 | Semantic retrieval, vector databases, RAG, text similarity calculation, clustering, reranking, etc. | CMTEB Base-level SOTA, high vector representation accuracy, strong Chinese-English mixing support, leading retrieval performance |

---

## Training Scheme
### I. Zhinao-ChineseModernBert General Foundation Pre-training
1. **MLM Pre-training (1T Tokens)**
   Built on the ModernBert architecture with Masked Language Modeling (MLM) as the pre-training objective, the model is **fully pre-trained from scratch** with no dependency on third-party foundation weights.
    - Pre-training corpus: 1T Tokens of high-quality Chinese-English text covering general news, encyclopedias, books, dialogues, code, law, professional literature, etc. Strict cleaning, deduplication, and quality filtering ensure diversity and purity.
    - Sequence length: Data sliced at 8192 tokens during pre-training to support long-text understanding and solve the limited long-text modeling ability of traditional Bert models.
    - Training optimization: Distributed pre-training framework with dynamic WWM masking, mixed-precision training, gradient accumulation, and other techniques to ensure stability and convergence.

2. **RetroMAE Pre-training**
    - Pre-training corpus: Based on Stage-1 corpora, supplemented by medium-short length text commonly used in downstream business scenarios.
    - Sequence length: Corpus split by natural paragraphs, max sequence length 1536.
    - Uses the RetroMAE self-supervised objective, specially optimized for sentence-level semantic representation. Bi-directional masked reconstruction and autoencoding enhance the model’s ability to model full-sentence semantics, addressing the weak adaptability of traditional MLM to sentence-level representations.

### II. Zhinao-ChineseModernBert-Embedding Two-Stage Embedding Training
Based on the Zhinao-ChineseModernBert general foundation, this model undergoes two-stage embedding training for semantic representation, upgrading from general semantic understanding to precise vector embedding:
1. **Retrieval Pre-training**
   Contrastive learning pre-training on hundreds of millions of Chinese-English retrieval and translation pairs, building semantic discrimination between positive and negative samples. Deeply adapted to core downstream tasks such as retrieval and similarity calculation, greatly improving cross-domain generalization.

2. **MTEB Chinese-English Full-Task Fine-tuning**
   Multi-task supervised fine-tuning on CMTEB and MTEB datasets, aligning optimization targets for retrieval, clustering, classification, STS, reranking, and other core tasks. Further improves real-world deployment performance for ready-to-use multi-scenario applications.

---

## Performance Evaluation
### 1. CLUE (Chinese Language Understanding Evaluation)
Zhinao-ChineseModernBert achieves **Base-level parameters (~100M excluding Embedding)** with overall performance exceeding large-parameter models such as RoBERTa-wwm-large on the CLUE benchmark, offering more options for resource-constrained business scenarios.

| model | params | afqmc | tnews | iflytek | cmnli | wsc | csl | ocnli | c3 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ChineseModernBERT(large) | ~310M | 73.87 | 56.90 | 60.15 | 83.96 | 52.10 | 86.20 | 79.10 | 82.65 | 71.87 |
| RoBERTa-wwm-large | ~310M | 76.55 | 58.61 | 62.98 | 82.12 | 74.60 | 82.13 | 78.20 | 73.82 | 73.63 |
| RoBERTa-wwm-ext(base) | ~90M | 74.04 | 56.94 | **60.31** | 80.51 | 67.80 | 81.00 | 74.72 | 66.50 | 70.23 |
| Zhinao-ChineseModernBERT(base) | ~110M | **76.99** | **57.51** | 59.56 | **83.82** | **78.95** | **85.70** | **79.08** | **75.42** | **74.63** |

### 2. CMTEB (Chinese Massive Text Embedding Benchmark)
Zhinao-ChineseModernBert-Embedding ranks **1st among Base-level models (~100M parameters excluding Embedding)** on the CMTEB benchmark, outperforming mainstream large-parameter embedding models such as Qwen3-Embedding-0.6B.

| Model Name | Params | Dimension | Classification | Clustering | Pair Classification | Reranking | Retrieval | STS | Mean Task Type |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| bge-base-zh-v1.5 | ~90M | 1024 | 71.79 | 47.49 | 73.18 | 65.02 | 69.41 | 51.67 | 63.07 |
| piccolo-base-zh | ~90M | 768 | 70.05 | 47.12 | 70.17 | 66.68 | 71.20 | 54.39 | 63.27 |
| setlla-base-zh-v3-1792d | ~90M | 1792 | 74.40 | 53.29 | 82.50 | 67.84 | 72.28 | 61.92 | 68.71 |
| setlla-large-zh-v3.5-1792d | ~310M | 1792 | 74.66 | 54.31 | 82.92 | 68.45 | 73.52 | 51.93 | 69.30 |
| Qwen3-Embedding-0.6B | ~596M | 1024 | 71.4 | 68.74 | 76.42 | 62.58 | 71.03 | 54.52 | 67.45 |
| 360Zhinao-Embedding-Base | ~110M | 768 | 73.50 | 65.36 | 86.15 | 67.99 | 69.32 | 60.29 | **70.16** |

> Note: Evaluation code and detailed metrics can be found in [eval_mteb.py](./eval_mteb.py) and the [CMTEB leaderboard](https://huggingface.co/spaces/mteb/leaderboard). The model achieves leading performance at similar scales, surpassing the 600M-parameter Qwen3-Embedding-0.6B.

---

## Quick Start
### Requirements
Python 3.10+ is recommended. Core dependencies:
```bash
pip install torch>=2.6.0 transformers>=4.56.2 sentence-transformers>=5.1.2
```

### Model Download
Model weights are open-sourced on mainstream platforms and can be obtained at:
| Model | Hugging Face Link |
|:------:|:-------------------:|
| Zhinao-ChineseModernBert | [🤗](https://huggingface.co/qihoo360/Zhinao-ChineseModernBert) |
| Zhinao-ChineseModernBert-Embedding | [🤗](https://huggingface.co/qihoo360/Zhinao-ChineseModernBert-Embedding) |

### Usage Examples
#### 1. Zhinao-ChineseModernBert General Foundation Example
The foundation model can be directly loaded via Hugging Face Transformers and requires fine-tuning for downstream tasks. See [clue_evaluator.py](./clue_evaluator.py) for details.
```python

```

#### 2. Zhinao-ChineseModernBert-Embedding Semantic Embedding Example

```python
from sentence_transformers import SentenceTransformer

# Load model
model = SentenceTransformer("qihoo360/360Zhinao-Embedding-Base")

# Flash attention is recommended
# model = SentenceTransformer("qihoo360/360Zhinao-Embedding-Base", 
#                             model_kwargs={"attn_implementation": "flash_attention_2", "dtype": torch.bfloat16})

# Input text (natively supports Chinese, English, and mixed text)
sentences = [
    "这是一个中文测试句子",
    "This is an English test sentence",
    "这是一个中英夹杂的test sentence",
]

# Generate semantic embeddings
embeddings = model.encode(sentences, prompt="Instruct: Retrieve semantically similar text.\nQuery: ")

# Output results
print(f"Embedding dimension: {embeddings.shape}")
print(embeddings@embeddings.T)
```

#### Semantic Similarity Calculation Example
```python
from sentence_transformers import SentenceTransformer, util

# Load model
model = SentenceTransformer("qihoo360/360Zhinao-Embedding-Base")

# Query and candidate documents
query = "什么是语义嵌入模型？"
docs = [
    "语义嵌入模型是将文本转化为高维向量的AI模型，可用于计算文本间的语义相似度",
    "大语言模型是能够生成自然语言文本的AI模型，广泛应用于对话、写作等场景",
    "计算机视觉模型用于处理图像数据，实现图像分类、目标检测等功能",
]

# Generate embeddings
query_embedding = model.encode(query, prompt="Instruct: Given a Chinese search query, retrieve web passages that answer the question.\nQuery: ")
doc_embeddings = model.encode(docs)

# Compute cosine similarity and print results
cos_scores = util.cos_sim(query_embedding, doc_embeddings)[0]
for i, score in enumerate(cos_scores):
    print(f"Document {i+1} Similarity: {score:.4f} | Content: {docs[i]}")
```

---

## Limitations & Usage Notes
1. **Sequence Length Limits**: Zhinao-ChineseModernBert supports 1536 tokens and can theoretically be fine-tuned up to 8192. Zhinao-ChineseModernBert-Embedding supports a maximum of 512 tokens. Over-length text should be properly truncated or chunked based on the scenario.
2. **Language Support**: The model is primarily pre-trained on Chinese with English support, optimized for Chinese and Chinese-English mixed text. For pure English professional scenarios, native English embedding models are recommended.
3. **Prompt for Embedding Model**: The embedding model uses a general prompt setup. Prompts for each task can be found in [config_sentence_transformers.json](https://huggingface.co/qihoo360/Zhinao-ChineseModernBert-Embedding/blob/main/config_sentence_transformers.json). Prompts are recommended for other tasks.
4. **Compliance**: The model may only be used for legal and compliant academic and industrial purposes. It is strictly prohibited for illegal activities such as endangering national security, cyberbullying, and disinformation. Please abide by relevant laws, regulations, and the open-source license.
5. **Disclaimer**: The model is pre-trained on public corpora. We have made every effort to ensure data compliance and model safety, but assume no legal liability for model outputs. Model outputs do not represent our official stance or views.

---

## License
Model weights and code in this project are open-sourced under the **Apache License 2.0**, free for academic research and commercial use. See the LICENSE file for full terms.

---

## Citation
If this project helps your research or work, please cite:
```bibtex
@misc{zhinao-chinesemodernbert,
  title={Zhinao-ChineseModernBert: Chinese Foundation & Vector Embedding Model for High-Throughput, Low-Memory Scenarios},
  author={zhinao team},
  year={2026},
  howpublished={\url{https://github.com/your-repo/zhinao-chinesemodernbert}},
}
```

---

## Acknowledgments
- We thank the ModernBert team for open-sourcing the efficient Transformer architecture, which provides a solid foundation for this project.
- We thank the Qwen team for open-sourcing Qwen2Tokenizer, offering an excellent tokenization system.
- We thank CLUE, MTEB, and CMTEB for providing authoritative Chinese NLP evaluation benchmarks.
- We thank the open-source community for contributions to Chinese NLP development.