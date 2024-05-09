from datasets import load_dataset
import sentencepiece as spm
import os

# 데이터셋 로드
dataset = load_dataset("iwslt2017", "iwslt2017-de-en", split="train")
dataset = dataset.train_test_split(test_size = 0.2)
print("dataload finished")
# 학습용 텍스트 추출
de_text = [example["translation"]["de"] for example in dataset['train']]
print(".")
en_text = [example["translation"]["en"] for example in dataset['train']]
texts = de_text+en_text
print("open text file")
# 텍스트 파일로 저장
with open("train_text.txt", "w", encoding="utf-8") as f:
    for text in texts:
        f.write(text + "\n")
print("train start")
# SentencePiece 모델 학습
spm.SentencePieceTrainer.Train(
    input="train_text.txt", 
    model_prefix="german_tokenizer", 
    vocab_size=30000, 
    model_type="unigram"
)
print("train finished")
# 학습된 모델 파일 이동
os.makedirs("tokenizer", exist_ok=True)
os.rename("tokenizer.model", os.path.join("tokenizer", "german_tokenizer.model"))
os.rename("tokenizer.vocab", os.path.join("tokenizer", "german_tokenizer.vocab"))
print("saved")