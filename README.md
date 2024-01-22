## Fine-tuning Multilingual Large Speech Recognition Models: Wav2vec and Whisper

This repository contains code to easily fine-tune pre-trained large speech recognition models on single or multiple GPUs.

Start by cloning the repository:
'''
git clone https://github.com/aitor-alvarez/large-speech-models
'''
Then install cd into the directory and install requirements:

'''
pip install -r requirements.txt
'''

There is only one configuration file model_asr.sh
Inside this file you will find all the parameters needed.

'''
python models/asr.py  \
--model_id='facebook/wav2vec2-xls-r-300m' \
--num_epochs=30 \
--batch_size=16 \
--lang='ar' \
--dataset='mozilla-foundation/common_voice_11_0' \
--output_dir='fine_tuned_models' \
--train_test='test'
'''
