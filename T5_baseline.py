from datasets import load_dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config, Seq2SeqTrainer, Seq2SeqTrainingArguments
from datasets import load_metric
#--------------data load---------------
from datasets import load_dataset
dataset = load_dataset("iwslt2017", "iwslt2017-de-en", split="train")

dataset = dataset.train_test_split(test_size = 0.2)

print(dataset)
print(dataset['train']['translation'][0])

#---------------model train-----------------

from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

tokenizer = T5Tokenizer.from_pretrained("t5-small", legacy=False)
config = T5Config.from_pretrained("t5-small")
config.decoder_start_token_id = tokenizer.pad_token_id
model = T5ForConditionalGeneration(config)

task_prefix = "translate German to English: "
train_sentences = [task_prefix + dataset['train']['translation'][i]['de'] for i in range(len(dataset['train']['translation']))]
train_inputs = tokenizer(train_sentences, return_tensors="pt", padding=True, truncation=True)
train_labels = tokenizer([dataset['train']['translation'][i]['en'] for i in range(len(dataset['train']['translation']))], return_tensors="pt", padding=True, truncation=True)
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
)
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_inputs,
    eval_dataset=train_labels,
    device = "cuda"
)
trainer.train()

#---------------test--------------------------


eval_sentences = [task_prefix + dataset['test']['translation'][i]['de'] for i in range(len(dataset['test']['translation']))]
eval_inputs = tokenizer(eval_sentences, return_tensors="pt", padding=True, truncation=True)
eval_labels = tokenizer([dataset['test']['translation'][i]['en'] for i in range(len(dataset['test']['translation']))], return_tensors="pt", padding=True, truncation=True)

metric = load_metric("bleu")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # 디코딩
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    # BLEU 스코어 계산
    result = metric.compute(predictions=pred_str, references=labels_str)
    return {"bleu": result["bleu"]}

# 평가
trainer = Seq2SeqTrainer(
    model=model,
    args=Seq2SeqTrainingArguments(output_dir="./results"),
    compute_metrics=compute_metrics,
    device="cuda"
)
evaluation_results = trainer.evaluate(eval_dataset=eval_inputs, eval_label=eval_labels)
print("Evaluation Results : ", evaluation_results)