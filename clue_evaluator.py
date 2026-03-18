import os
import argparse
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    AutoModelForMultipleChoice,
)
import numpy as np
from einops import rearrange

# 设置日志
from loguru import logger

# CLUE任务列表及其相关信息
CLUE_TASKS = {
    "afqmc": {"num_labels": 2, "task_type": "classification", "bsz": 32},
    "cmnli": {"num_labels": 3, "task_type": "classification", "bsz": 256, "epochs": 3},
    "copa": {"num_labels": 2, "task_type": "classification", "bsz": 32},
    "cpolar": {"num_labels": 2, "task_type": "classification", "bsz": 32},
    "csl": {"num_labels": 2, "task_type": "classification", "bsz": 16, "epochs": 5},
    "iflytek": {"num_labels": 119, "task_type": "classification", "bsz": 16, "epochs": 5},
    "ocnli": {"num_labels": 3, "task_type": "classification", "bsz": 32},
    "tnews": {"num_labels": 15, "task_type": "classification", "bsz": 32},
    "wsc": {"num_labels": 2, "task_type": "classification", "bsz": 8, "epochs": 10},
    "chnsenticorp": {"num_labels": 2, "task_type": "classification", "bsz": 32},
    "cluener": {"num_labels": 20, "task_type": "token_classification", "bsz": 32},  # 中文NER任务
    "c3": {"num_labels": 4, "task_type": "multiple_choice", "bsz": 32, "epochs": 5},
    # "c3": {"num_labels": 4, "task_type": "classification"},
}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate BERT-like models on CLUE benchmark")
    parser.add_argument("--model_name_or_path", type=str, required=True,
                      help="Path to pre-trained model or model name (e.g., bert-base-chinese)")
    parser.add_argument("--task_name", type=str, required=True, choices=CLUE_TASKS.keys(),
                      help="Name of the CLUE task to evaluate")
    parser.add_argument("--output_dir", type=str, default="./clue_results",
                      help="Directory to save evaluation results")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for evaluation")
    parser.add_argument("--max_seq_length", type=int, default=512,
                      help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=5,
                      help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                      help="Learning rate")
    return parser.parse_args()

def preprocess_function(examples, tokenizer, task_name, max_seq_length):
    """根据不同任务预处理数据"""
    if task_name in ["cpolar", "chnsenticorp"]:
        # 句子分类任务
        return tokenizer(examples["sentence"], truncation=True, max_length=max_seq_length)
    if task_name in ["afqmc"]:
        # 句子分类任务
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_seq_length)
    
    elif task_name in ["cmnli", "ocnli"]:
        # 自然语言推理任务
        return tokenizer(examples["sentence1"], examples["sentence2"], truncation=True, max_length=max_seq_length)
    
    elif task_name == "copa":
        # 因果推理任务 - 需要分别处理两个选项
        # 这里简化处理，实际应该分别对两个选项进行编码
        premise = examples["premise"]
        choice1 = examples["choice1"]
        choice2 = examples["choice2"]
        
        # 为每个选项创建输入
        choice = choice1 + tokenizer.eos_token + choice2
        
        # 这里简化处理，实际需要更复杂的逻辑
        return tokenizer(premise, choice, truncation=True, max_length=max_seq_length)
    
    elif task_name == "csl":
        # 论文关键词分类任务
        # 将关键词列表合并为字符串
        keywords = [" ".join(kw_list) if isinstance(kw_list, list) else " ".join(kw_list.split()) for kw_list in examples["keyword"]]
        return tokenizer(keywords, examples["abst"], 
                        truncation=True, max_length=max_seq_length)
    
    elif task_name == "iflytek":
        # 应用分类任务
        return tokenizer(examples["sentence"], truncation=True, max_length=max_seq_length)
    
    elif task_name == "tnews":
        # 新闻分类任务
        return tokenizer(examples["sentence"], truncation=True, max_length=max_seq_length)
    
    elif task_name == "wsc":
        # 指代消解任务 - 需要特殊处理
        # 简化处理：直接使用文本
        texts = []
        targets = []
        labels = []
        for i in range(len(examples["text"])):
            text = examples["text"][i]
            target = examples["target"][i] if "target" in examples else {}
            texts.append(f"{target['span2_text']}[SEP]{text}")
            targets.append(target['span1_text'])
            labels.append(examples["label"][i])
        
        inputs = tokenizer(texts, targets, truncation=True, max_length=max_seq_length)
        inputs["labels"] = labels
        return inputs
    
    elif task_name == "cluener":
        # 命名实体识别任务
        tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True,
                                    truncation=True, max_length=max_seq_length)
        
        labels = []
        for i, label in enumerate(examples["labels"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    # 对于subword tokens，使用-100（忽略）
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    elif task_name == "c3":
        # C3多选题任务
        # C3数据格式: context(对话上下文), question(问题), choice(选项列表), answer(答案索引)
        contexts = examples["context"]
        questions = examples["question"]
        choices = examples["choice"]
        answers = examples["answer"] if "answer" in examples else None
        
        
        # 存储处理后的结果
        labels_list = []
        
        strs_a = []
        strs_b = []
        
        # 对每个样本进行处理
        for i, (context, question, choice_list) in enumerate(zip(contexts, questions, choices)):
            # 将对话上下文连接成一段文本
            context_text = ";".join(context) if isinstance(context, list) else str(context)
            full_context = context_text + "[SEP]" + question
            
            strs_a.extend([full_context for _ in range(4)])
            strs_b.extend([choice for choice in (choice_list + ['无效答案']*2)[:4]])
            
            # 处理标签
            if answers is not None:
                answer = answers[i]
                # 如果答案是选项文本，找到对应的索引
                if isinstance(answer, str):
                    try:
                        label_idx = choice_list.index(answer)
                    except ValueError:
                        # 如果找不到确切匹配，尝试模糊匹配
                        label_idx = 0  # 默认为第一个选项
                        for idx, choice_text in enumerate(choice_list):
                            if answer in choice_text or choice_text in answer:
                                label_idx = idx
                                break
                elif isinstance(answer, (int, list)):
                    label_idx = answer if isinstance(answer, int) else answer[0]
                else:
                    label_idx = 0
                labels_list.append(label_idx)
        
        tokenized_inputs = tokenizer(strs_a, strs_b, truncation=True, padding=True, return_tensors="pt", max_length=max_seq_length)
        
        
        result = {
            "input_ids": rearrange(tokenized_inputs["input_ids"], "(b n) d -> b n d", n=4),
            "attention_mask": rearrange(tokenized_inputs["attention_mask"], "(b n) d -> b n d", n=4),
        }
        
        if labels_list:
            print(labels_list[:10])
            result["labels"] = labels_list
            
        return result
    else:
        raise ValueError(f"Unknown task: {task_name}")

def compute_metrics(eval_pred, task_name, label_list=None):
    """根据不同任务计算评估指标"""
    logits, labels = eval_pred
    
    if task_name in ["afqmc", "cmnli", "cpolar", "csl", "iflytek", "ocnli", "tnews", "chnsenticorp", "wsc", "copa", "c3"]:
        # 分类任务：计算准确率
        predictions = np.argmax(logits, axis=1)
        accuracy = np.mean(predictions == labels)
        return {"accuracy": accuracy}
    
    elif task_name == "cluener":
        # NER任务：计算F1分数
        try:
            from seqeval.metrics import f1_score, precision_score, recall_score, accuracy_score
            
            predictions = np.argmax(logits, axis=2)
            
            # 移除特殊符号的标签
            true_predictions = [
                [label_list[p] for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(predictions, labels)
            ]
            true_labels = [
                [label_list[l] for (p, l) in zip(pred, label) if l != -100]
                for pred, label in zip(predictions, labels)
            ]
            
            results = {
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
                "accuracy": accuracy_score(true_labels, true_predictions),
            }
            return results
        except ImportError:
            logger.warning("seqeval not installed, using simple accuracy for NER")
            predictions = np.argmax(logits, axis=2)
            # 简单的准确率计算
            flat_predictions = predictions.flatten()
            flat_labels = labels.flatten()
            valid_indices = flat_labels != -100
            accuracy = np.mean(flat_predictions[valid_indices] == flat_labels[valid_indices])
            return {"accuracy": accuracy}
    
    else:
        raise ValueError(f"Unknown task: {task_name}")

def main():
    args = parse_args()
    task_info = CLUE_TASKS[args.task_name]
    task_type = task_info["task_type"]
    num_labels = task_info["num_labels"]
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Starting evaluation for task: {args.task_name} with model: {args.model_name_or_path}")
    
    # 加载数据集
    try:
        if args.task_name == "wsc":
            # 加载中文WSC数据集
            datasets = load_dataset("clue", "cluewsc2020")
        else:
            datasets = load_dataset("clue", args.task_name)
        logger.info(f"Successfully loaded dataset: {args.task_name}")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    # 预处理数据集
    tokenized_datasets = datasets.map(
        lambda examples: preprocess_function(examples, tokenizer, args.task_name, args.max_seq_length),
        batched=True
    )
    
    # 准备训练和验证集
    if "train" in tokenized_datasets and "validation" in tokenized_datasets:
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]
    elif "train" in tokenized_datasets:
        # 如果没有验证集，从训练集分割
        train_test = tokenized_datasets["train"].train_test_split(test_size=0.1)
        train_dataset = train_test["train"]
        eval_dataset = train_test["test"]
    else:
        logger.error("No training data available")
        return
    
    # 加载模型
    try:
        if task_type == "classification":
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name_or_path, 
                num_labels=num_labels,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        elif task_type == "token_classification":
            model = AutoModelForTokenClassification.from_pretrained(
                args.model_name_or_path, 
                num_labels=num_labels,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        elif task_type == "multiple_choice":
            model = AutoModelForMultipleChoice.from_pretrained(
                args.model_name_or_path,
                attn_implementation="flash_attention_2",
                trust_remote_code=True,
            )
        logger.info(f"Successfully loaded model: {args.model_name_or_path}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # 准备标签列表（用于NER任务）
    label_list = None
    if args.task_name == "cluener":
        label_list = datasets["train"].features["labels"].feature.names
    
    # 定义训练参数
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.task_name),
        num_train_epochs=CLUE_TASKS[args.task_name].get("epochs", args.num_epochs),
        per_device_train_batch_size=CLUE_TASKS[args.task_name].get("bsz", args.batch_size),
        per_device_eval_batch_size=CLUE_TASKS[args.task_name].get("bsz", args.batch_size),
        warmup_ratio=0.1,
        weight_decay=5e-8,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        learning_rate=args.learning_rate,
        lr_scheduler_type='cosine'
    )
    
    # 数据收集器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 定义Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=lambda x: compute_metrics(x, args.task_name, label_list),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # 训练模型
    logger.info("Starting training...")
    trainer.train()
    
    # 评估模型
    logger.info("Starting evaluation...")
    eval_results = trainer.evaluate()
    
    # 保存并打印评估结果
    logger.info(f"Evaluation results for {args.task_name}:")
    for key, value in eval_results.items():
        logger.info(f"{key}: {value}")
    
    # 保存结果到文件
    result_file = os.path.join(args.output_dir, f"{args.task_name}_results.txt")
    with open(result_file, "w") as f:
        f.write(f"Model: {args.model_name_or_path}\n")
        f.write(f"Task: {args.task_name}\n")
        for key, value in eval_results.items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Evaluation completed successfully! Results saved to {result_file}")

if __name__ == "__main__":
    main()