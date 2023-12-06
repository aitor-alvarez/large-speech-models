#Large multilingual wav2vec models
#facebook/wav2vec2-large-xlsr-53
#facebook/wav2vec2-xls-r-300m
#facebook/Wav2Vec2-XLS-R-1B
#facebook/Wav2Vec2-XLS-R-2B
#facebook/mms-1b-all

python models/asr.py  \
--model_id='facebook/wav2vec2-large-xlsr-53' \
--lang='ar'\
--dataset='mozilla-foundation/common_voice_13_0' \
--output_dir='fine_tuned_models'\
--train_test='test'