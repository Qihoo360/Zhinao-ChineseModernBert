import torch
import mteb
from sentence_transformers import SentenceTransformer
from mteb.cache import ResultCache
import os

MODEL_PATH = 'qihoo360/Zhinao-ChineseModernBert-Embedding'

def eval():

    # 初始化模型
    model = SentenceTransformer(MODEL_PATH, model_kwargs={"attn_implementation": "flash_attention_2", "dtype": torch.bfloat16})
    model.eval()
    benchmark = mteb.get_benchmark('MTEB(cmn, v1)')
        
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache/sentence-transformers", os.path.basename(MODEL_PATH))
    results = mteb.evaluate(model, tasks=benchmark, cache=ResultCache(cache_dir))
    for task_name, score in results.items():
        print(f"{task_name}: {score['main_score']:.4f}")
    

if __name__ == '__main__':
    eval()
