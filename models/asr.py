from datasets import load_dataset, load_metric, Audio
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union
import re, json
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, \
    TrainingArguments, Trainer, WhisperFeatureExtractor, WhisperProcessor, WhisperForConditionalGeneration, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer
import argparse
from arabic_preprocess import process_text


wer_metric = load_metric("wer")


chars_to_remove_regex = '[\,\؟\.\!\-\;\:\'\"\☭\«\»\؛\—\ـ\_\،\“\%\‘\”\�]'

def remove_special_characters(batch):
    batch["sentence"] = re.sub(chars_to_remove_regex, '', batch["sentence"]).lower()
    return batch

def remove_ar_special_characters(batch):
    #Change batch["text"] to batch ["sentence"] if using Common Voice dataset or to batch["transcription"] if using FLEURS
    batch["sentence"] = process_text(batch["transcription"]).lower()
    return batch

def extract_characters(batch):
    txt = " ".join(batch["sentence"])
    vocab = list(set(txt))
    dict_out = {'vocab': [vocab], 'txt': [txt]}
    return dict_out


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


def prepare_dataset(batch):
    audio = batch["audio"]

    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
    batch["input_length"] = len(batch["input_values"])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch


@dataclass
class DataCollatorCTCWithPadding:
    '''
    For padding data dynamically
    '''

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
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


class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)


        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

def train_asr(output_dir, model_id, batch_size, num_epochs):
    if 'whisper' in model_id:
        data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
        model = WhisperForConditionalGeneration.from_pretrained(model_id).to(device)
        model.config.forced_decoder_ids = None
        model.config.suppress_tokens = []
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            learning_rate=1e-5,
            warmup_steps=500,
            max_steps=4000,
            gradient_checkpointing=True,
            fp16=True,
            evaluation_strategy="steps",
            per_device_eval_batch_size=batch_size,
            predict_with_generate=True,
            generation_max_length=225,
            save_steps=1000,
            eval_steps=1000,
            logging_steps=50,
            #report_to=["tensorboard"],
            load_best_model_at_end=True,
            metric_for_best_model="wer",
            greater_is_better=False,
            push_to_hub=False,
        )
        trainer = Seq2SeqTrainer(
            args=training_args,
            model=model,
            train_dataset=speech_train,
            eval_dataset=speech_test,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            tokenizer=processor.feature_extractor,
        )


    else:
        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
        model = Wav2Vec2ForCTC.from_pretrained(
            model_id,
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.0,
            ctc_loss_reduction="mean",
            pad_token_id=processor.tokenizer.pad_token_id,
            vocab_size=len(processor.tokenizer),
        ).to(device)

        training_args = TrainingArguments(
            output_dir=output_dir,
            remove_unused_columns=False,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=2,
            evaluation_strategy="steps",
            num_train_epochs=num_epochs,
            gradient_checkpointing=True,
            fp16=True,
            save_steps=400,
            eval_steps=1000,
            logging_steps=100,
            learning_rate=3e-4,
            warmup_steps=500,
            save_total_limit=2,
            push_to_hub=False,
        )
        model.freeze_feature_extractor()

        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=speech_train,
            eval_dataset=speech_test,
            tokenizer=processor.feature_extractor,
        )

    trainer.train(resume_from_checkpoint=True)

    print("training completed")



if __name__ == '__main__':
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    parser = argparse.ArgumentParser()
    parser.add_argument( '--model_id')
    parser.add_argument('--num_epochs')
    parser.add_argument('--batch_size')
    parser.add_argument('--lang')
    parser.add_argument('--data_lang')
    parser.add_argument('--data_folder')
    parser.add_argument('--dataset')
    parser.add_argument('--output_dir')
    parser.add_argument('--train_test')
    args = parser.parse_args()

    if args.train_test == 'train':
        #For datasets that are not hosted in Huggingface but on local disk
        if args.data_folder is not None:
            speech_train = load_dataset("audiofolder", data_dir=args.data_folder, split="train")
            speech_test = load_dataset("audiofolder", data_dir=args.data_folder, split="test")
        else:
            speech_train = load_dataset(args.dataset, args.data_lang, split="train+validation")
            speech_test = load_dataset(args.dataset, args.data_lang, split="test")

        if args.data_lang =='ar':
            speech_train = speech_train.map(remove_ar_special_characters)
            speech_test = speech_test.map(remove_ar_special_characters)
        else:
            speech_train = speech_train.map(remove_special_characters)
            speech_test = speech_test.map(remove_special_characters)

        vocab_train = speech_train.map(extract_characters, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=speech_train.column_names)
        vocab_test = speech_test.map(extract_characters, batched=True, batch_size=-1, keep_in_memory=True,
                                               remove_columns=speech_test.column_names)

        vocab = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))
        vocab_dict = {v: k for k, v in enumerate(sorted(vocab))}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        with open('vocab.json', 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)

        speech_train = speech_train.cast_column("audio", Audio(sampling_rate=16_000))
        speech_test = speech_test.cast_column("audio", Audio(sampling_rate=16_000))


        if 'whisper' in args.model_id:
            feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model_id)
            processor = WhisperProcessor.from_pretrained(args.model_id, language=args.lang, task="transcribe")

        elif 'facebook' or 'wav2vec2' in args.model_id:
            feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                         do_normalize=True, return_attention_mask=True)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(".", unk_token="[UNK]", pad_token="[PAD]",
                                                                 word_delimiter_token="|")
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

        speech_train = speech_train.map(prepare_dataset, remove_columns=speech_train.column_names)
        speech_test = speech_test.map(prepare_dataset, remove_columns=speech_test.column_names)

        train_asr(args.output_dir, args.model_id, args.batch_size, args.num_epochs)

    elif args.train_test == 'test':

        if not 'openai' in args.model_id:
            feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0,
                                                         do_normalize=True, return_attention_mask=True)
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(args.model_id, unk_token="[UNK]", pad_token="[PAD]",
                                                             word_delimiter_token="|")
            processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
            model = Wav2Vec2ForCTC.from_pretrained(args.model_id).to(device)
            if args.data_folder is not None:
                speech_test = load_dataset("audiofolder", data_dir=args.data_folder, split="test")
                speech_test = speech_test.map(remove_ar_special_characters)
                speech_test = speech_test.map(prepare_dataset, remove_columns=speech_test.column_names)
            else:
                speech_test = load_dataset(args.dataset, args.data_lang, split="test")
                speech_test = speech_test.map(remove_ar_special_characters)
                speech_test = speech_test.cast_column("audio", Audio(sampling_rate=16_000))
                speech_test = speech_test.map(prepare_dataset, remove_columns=speech_test.column_names)

            def get_results(batch):
                with torch.no_grad():
                    input_dict = processor(batch["input_values"], return_tensors="pt", padding=True)
                    logits = model(input_dict.input_values.to(device),
                                   attention_mask=input_dict.attention_mask.to(device)).logits
                    pred_ids = torch.argmax(logits, dim=-1)
                    batch["pred_txt"] = processor.batch_decode(pred_ids)[0]
                    batch["txt"] = processor.decode(batch["labels"])
                    return batch

            results = speech_test.map(get_results)

            print("Test WER: {:.2f}".format(
                100 * wer_metric.compute(predictions=results["pred_txt"], references=results["txt"])))

        elif "openai" in args.model_id:
            processor = WhisperProcessor.from_pretrained(args.model_id)
            model = WhisperForConditionalGeneration.from_pretrained(args.model_id).to(device)
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=args.lang, task="transcribe")
            if args.data_folder is not None:
                speech_test = load_dataset("audiofolder", data_dir=args.data_folder, split="test")
            else:
                speech_test = load_dataset(args.dataset, args.data_lang, split="test")
                speech_test = speech_test.cast_column("audio", Audio(sampling_rate=16_000))

            def map_to_pred(batch):
                audio = batch["audio"]
                input_features = processor(audio["array"], sampling_rate=audio["sampling_rate"],
                                           return_tensors="pt").input_features
                #Change batch['transcription'] to batch['text'] if not using FLEURS
                batch["reference"] = processor.tokenizer._normalize(batch['sentence'])

                with torch.no_grad():
                    predicted_ids = model.generate(input_features.to(device), forced_decoder_ids=forced_decoder_ids)[0]
                transcription = processor.decode(predicted_ids)
                batch["prediction"] = processor.tokenizer._normalize(transcription)
                return batch

            result = speech_test.map(map_to_pred)
            print("Test WER: {:.2f}".format(
                100 * wer_metric.compute(references=result["reference"], predictions=result["prediction"])))
