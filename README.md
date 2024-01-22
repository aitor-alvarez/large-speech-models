## Fine-tuning Multilingual Large Speech Recognition Models: Wav2vec and Whisper

This repository contains code to easily fine-tune pre-trained large speech recognition models on single or multiple GPUs.

Start by cloning the repository:
```
git clone https://github.com/aitor-alvarez/large-speech-models
```

Then install cd into the directory and install requirements:

```
pip install -r requirements.txt
```

There is only one configuration file model_asr.sh
Inside this file you will find all the parameters needed.

```
python models/asr.py  \
--model_id='facebook/wav2vec2-xls-r-300m' \
--num_epochs=30 \
--batch_size=16 \
--lang='ar' \
--dataset='mozilla-foundation/common_voice_11_0' \
--output_dir='fine_tuned_models' \
--train_test='train'
'''

Parameters:
-model_id: string use either a huggingface pretrained model (like above) or a local directory with the pre-trained model.
-num_epochs: int
-batch_size: int
-lang: string use language code if using CV (https://huggingface.co/datasets/common_voice).
-dataset: string a dataset from transformers library datasets.
-output_dir: string directorry where the fine-tuned model will be saved.
-train_test: string either 'train' or 'test' depending on whether you are fine-tuning or using a fine-tuned model for inference.

If using a custom dataset you can use the following parameter instead of "dataset":
-data_folder: string provide the path to your local dataset following the format of transformers dataset library (https://huggingface.co/docs/datasets/create_dataset)

If using data_folder you will need to use data_lang with language code.
'''

This code was done with the idea of fine-tuning Wav2vec and Whisper for Arabic.

Some pre-trained models can be found here: https://huggingface.co/aitor-alvarez/wav2vec2-xls-r-300m-ar 
