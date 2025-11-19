# ================================================
# Imports
# ================================================
from datasets import load_dataset, Audio
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import evaluate
import torch
import os
import soundfile as sf

# ================================================
# Désactivation de torchcodec (obligatoire sous WSL)
# ================================================
os.environ["HF_DATASETS_AUDIO_ALLOW_TORCHCODEC"] = "0"

# ================================================
# Chargement des manifests JSON
# ================================================
train_data = load_dataset("json", data_files="nemo_manifests/train_manifest.json")["train"]
dev_data = load_dataset("json", data_files="nemo_manifests/dev_manifest.json")["train"]

# Debug: sélection petite portion pour tests
train_data = train_data.select(range(20))
dev_data = dev_data.select(range(10))

# ================================================
# Chargement modèle + processeur Whisper
# ================================================
model_name = "openai/whisper-small"
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name)

# Forcer la langue + tâche
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="fr", task="transcribe")
model.config.suppress_tokens = []

# ================================================
# Prétraitement dataset
# ================================================
def prepare(batch):
    audio_path = batch["audio_filepath"]
    audio, sr = sf.read(audio_path)

    if sr != 16000:
        raise ValueError(f"Sample rate {sr} != 16000. Resample avant l'entraînement")

    batch["input_features"] = processor.feature_extractor(
        audio, sampling_rate=sr
    )["input_features"][0]

    batch["labels"] = processor.tokenizer(batch["text"])["input_ids"]
    return batch

train_data = train_data.map(prepare, remove_columns=train_data.column_names)
dev_data = dev_data.map(prepare, remove_columns=dev_data.column_names)

# ================================================
# Arguments d'entraînement
# ================================================
training_args = Seq2SeqTrainingArguments(
    output_dir="whisper-finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=500,
    max_steps=2000,
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    predict_with_generate=True  # indispensable pour Whisper
)

# ================================================
# Métrique WER
# ================================================
wer = evaluate.load("wer")

def compute_metrics(pred):
    preds = processor.batch_decode(pred.predictions, skip_special_tokens=True)
    labels = processor.batch_decode(pred.label_ids, skip_special_tokens=True)
    return {"wer": wer.compute(predictions=preds, references=labels)}

# ================================================
# Trainer HF
# ================================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=dev_data,
    tokenizer=processor,  # correct dans les versions récentes
    compute_metrics=compute_metrics
)

# ================================================
# Lancement entraînement
# ================================================
trainer.train()
