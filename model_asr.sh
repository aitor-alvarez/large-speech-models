#Large multilingual wav2vec and Whisper models
#facebook/wav2vec2-large-xlsr-53
#facebook/wav2vec2-xls-r-300m
#facebook/Wav2Vec2-XLS-R-1B
#facebook/Wav2Vec2-XLS-R-2B
#facebook/mms-1b-all
#openai/whisper-small
#openai/whisper-medium
#openai/whisper-large-v3

python models/asr.py  \
--model_id='facebook/wav2vec2-xls-r-300m' \
--num_epochs=30 \
--batch_size=16 \
--lang='ar' \
--dataset='mozilla-foundation/common_voice_11_0' \
--output_dir='fine_tuned_models' \
--train_test='test'
