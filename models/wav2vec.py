from datasets import load_dataset, load_metric, Audio
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import re, json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, \
    TrainingArguments, Trainer


chars_to_remove_regex = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                             do_normalize=True, return_attention_mask=True)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./", unk_token="[UNK]", pad_token="[PAD]",
                                                     word_delimiter_token="|")

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

wer_metric = load_metric("wer")



def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

def extract_characters(batch):
    txt = " ".join(batch["sentence"])
    vocab = list(set(txt))
    dict_out = {'vocab': [vocab], 'txt': [txt]}
    return dict_out


def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


def process_common_voice_dataset(lang="tr"):
    common_voice_train = load_dataset("mozilla-foundation/common_voice_11_0", lang, split="train+validation")
    common_voice_test = load_dataset("mozilla-foundation/common_voice_11_0", lang, split="test")
    common_voice_train = common_voice_train.map(remove_special_characters).cast_column("audio", Audio(sampling_rate=16_000))\
        .map(prepare_dataset, remove_columns=common_voice_train.column_names)
    common_voice_test = common_voice_test.map(remove_special_characters).cast_column("audio", Audio(sampling_rate=16_000)).map(prepare_dataset, remove_columns=common_voice_train.column_names)

    vocab_train = common_voice_train.map(extract_characters, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=common_voice_train.column_names)
    vocab_test = common_voice_test.map(extract_characters, batched=True, batch_size=-1, keep_in_memory=True,
                                       remove_columns=common_voice_test.column_names)
    vocab = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
    vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
    vocab_dict["|"] = vocab_dict[" "]
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)
    with open('vocab.json', 'w') as vocab_file:
        json.dump(vocab_dict, vocab_file)
    return common_voice_train, common_voice_test



@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch



def train(output_dir):
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-xls-r-300m",
        attention_dropout=0.0,
        hidden_dropout=0.0,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.0,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer),
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        gradient_checkpointing=True,
        fp16=True,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
        push_to_hub=True,
    )
    model.freeze_feature_extractor()
    common_voice_train, common_voice_test = process_common_voice_dataset()

    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=common_voice_train,
        eval_dataset=common_voice_test,
        tokenizer=processor.feature_extractor,
    )

    trainer.train()

    print("training completed")

