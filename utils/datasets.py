from pydub import AudioSegment
import pandas as pd


#Function to extract MGB-5 dataset from a text file to individual mp3 files to be loaded into a data_load for ASR
def extract_mgb5(txt_file_path, audio_path):
    txt = open(txt_file_path, encoding="utf8").readlines()
    file_name =[]
    text=[]
    for t in txt:
        wav_file = t.split("_annotator")[0]+'.wav'
        wav_name = t.split(' ')[0]
        txt_out = t.replace(t.split(' ')[0], '').replace('\n', '')
        text.append(txt_out)
        file_name.append('data/train/'+wav_name+'.mp3')
        '''
        time_from = wav_name.split("_")[-2]
        time_to = wav_name.split("_")[-1]
        start_audio = float(time_from) * 1000
        end_audio = float(time_to) * 1000
        audio_segment = AudioSegment.from_file(audio_path+wav_file)
        audio_segment = audio_segment[start_audio:end_audio]
        audio_segment.export(audio_path + 'test/' + wav_name+'.mp3', format="mp3")
        '''
    df = pd.DataFrame({'file_name': file_name, 'text':text})
    df.to_csv(audio_path + 'test/'+'metadata.csv', encoding='utf-8-sig', index=False)