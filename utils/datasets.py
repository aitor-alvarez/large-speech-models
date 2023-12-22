from pydub import AudioSegment
import pandas as pd
import re, os


#Function to extract MGB-5 and MGB-3 datasets from a text file to individual mp3 files to be loaded into a data_load for ASR
def extract_mgb5(txt_file_path, audio_path):
    txt = open(txt_file_path, encoding="utf8").readlines()
    file_name =[]
    text=[]
    for t in txt:
        wav_file = t.split("_annotator")[0]+'.wav'
        wav_name = t.split(' ')[0]
        txt_out = t.replace(t.split(' ')[0], '').replace('\n', '').strip()
        text.append(txt_out)
        file_name.append('data/train/'+wav_name+'.mp3')
        time_from = wav_name.split("_")[-2]
        time_to = wav_name.split("_")[-1]
        start_audio = float(time_from) * 1000
        end_audio = float(time_to) * 1000
        audio_segment = AudioSegment.from_file(audio_path+wav_file)
        audio_segment = audio_segment[start_audio:end_audio]
        audio_segment.export(audio_path + 'test/' + wav_name+'.mp3', format="mp3")
    df = pd.DataFrame({'file_name': file_name, 'text':text})
    df.to_csv(audio_path + 'test/'+'metadata.csv', encoding='utf-8-sig', index=False)


def extract_mgb3(txt_file_path, audio_path, split="test"):
    txt = open(txt_file_path, encoding="utf8").readlines()
    file_name =[]
    text=[]
    for t in txt:
        if split == "test":
            wav_name = t.split(" ")[0]
            wav_file = wav_name+'.wav'
            if os.path.exists(audio_path+wav_file):
                ind = t.find(re.findall('\d+', t)[-1])+len(re.findall('\d+', t)[-1])
                txt_out = t[ind:].replace('\n', '').strip()
                text.append(txt_out)
                time_from = t.split(" ")[3]
                time_to = t.split(" ")[4]
                start_audio = float(time_from) * 1000
                end_audio = float(time_to) * 1000

                audio_segment = AudioSegment.from_file(audio_path+wav_file)
                audio_segment = audio_segment[start_audio:end_audio]
                audio_segment.export(audio_path + 'test/' + wav_name+'_'+time_from+'_'+time_to+'.mp3', format="mp3")
                if os.path.exists(audio_path + 'test/' + wav_name+'_'+time_from+'_'+time_to+'.mp3'):
                    file_name.append('data/test/' + wav_name + '_' + time_from + '_' + time_to + '.mp3')
            else:
                continue

    df = pd.DataFrame({'file_name': file_name, 'text':text})
    df.to_csv(audio_path + 'test/'+'metadata.csv', encoding='utf-8-sig', index=False)