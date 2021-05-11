import os
import glob2
import shutil
import tqdm
import random
import soundfile as sf
import librosa
import pandas as pd
import numpy as np
import argparse

# some constants
FSD50K_TRAIN_VAL_SPLIT = .9
CLIP_LEN_SECONDS = 4
FINAL_SR = 16000
CLIP_LEN_SAMPLES = CLIP_LEN_SECONDS * FINAL_SR
FSD_SPEECH_LABELS = \
set(['Crowd',
'Cheering',
'Human_group_actions',
'Crying_and_sobbing',
'Shout',
'Human_voice',
'Male_singing',
'Singing',
'Chuckle_and_chortle',
'Laughter',
'Chatter',
'Conversation',
'Speech',
'Male_speech_and_man_speaking',
'Female_speech_and_woman_speaking',
'Breathing',
'Respiratory_sounds',
'Child_speech_and_kid_speaking',
'Female_singing',
'Yell',
'Sneeze',
'Sigh',
'Whispering',
'Giggle',
'Speech_synthesizer',
'Screaming',
'Cough',
'Burping_and_eructation',])

def get_libri_speaker_maps(libri_dir):    
    libri_train_sets = [os.path.join(libri_dir, 'train-clean-100'), 
                        os.path.join(libri_dir,'train-clean-360')]
    
    libri_val_set = [os.path.join(libri_dir,'dev-clean')]
    libri_test_set = [os.path.join(libri_dir, 'test-clean')]
    
    train_speaker_map = get_speaker_utterance_map(libri_train_sets)
    val_speaker_map = get_speaker_utterance_map(libri_val_set)
    test_speaker_map = get_speaker_utterance_map(libri_test_set)
    return train_speaker_map, val_speaker_map, test_speaker_map
    
def get_nonspeech_fsd(fsd50k_dir, fsd_df_dev, fsd_df_test):
    fsd_dev_dir = os.path.join(fsd50k_dir, 'FSD50K.dev_audio')    
    fsd_test_dir = os.path.join(fsd50k_dir, 'FSD50K.eval_audio')

    fsd_dev_files = get_nonspeech_fsd_files(fsd_df_dev, fsd_dev_dir)
    fsd_test_files = get_nonspeech_fsd_files(fsd_df_test, fsd_test_dir)

    split_len = int(len(fsd_dev_files) * FSD50K_TRAIN_VAL_SPLIT)
    fsd_train_files = fsd_dev_files[:split_len]
    fsd_val_files = fsd_dev_files[split_len:]
    
    return fsd_train_files, fsd_val_files, fsd_test_files

def get_speaker_utterance_map(libri_sets):
    # get all the speakers
    all_speakers = []
    for libri_set in libri_sets:
        all_speakers.extend(glob2.glob(libri_set + '/*'))
        
    # make a map from speaker sto utterances
    speaker_file_map = {}
    for speaker in all_speakers:
        speaker_id = os.path.basename(speaker)
        speaker_files = glob2.glob(speaker + '/*/**.flac')
        speaker_file_map[speaker_id] = speaker_files
    return speaker_file_map

def get_nonspeech_fsd_files(fsd_df, fsd_dir):
    valid_ids = []
    for i in range(len(fsd_df.fname)):
        if not(set(fsd_df.labels[i].split(',')) & FSD_SPEECH_LABELS):
            valid_ids.append(fsd_df.fname[i])
            
    all_files = glob2.glob(fsd_dir + '/*.wav')
    random.shuffle(all_files)

    all_valid_files = []
    for file in all_files:
        if int(os.path.splitext(os.path.basename(file))[0]) in valid_ids:
            all_valid_files.append(file)
        
    return all_valid_files

def make_fsd_libri_pairs(speaker_map, fsd_files):
    speaker_fsd_map = {}
    fsd_idx = 0
    for key in speaker_map.keys():
        speaker_fsd_map[key] = []
        
        for utterance in speaker_map[key]:
            speaker_fsd_map[key].append([utterance, fsd_files[fsd_idx]])
            fsd_idx+=1
            
            if fsd_idx >= len(fsd_files):
                return speaker_fsd_map
            
    return speaker_fsd_map

def write_pairs(pairs_map, out_dir, fsd_df):
    os.makedirs(out_dir, exist_ok=True)
    
    for key in tqdm.tqdm(pairs_map.keys()):
        speaker_path = os.path.join(out_dir, key)
        speech_speaker_path = os.path.join(speaker_path, 'speech')
        noise_speaker_path = os.path.join(speaker_path, 'noise')
        
        os.makedirs(speaker_path, exist_ok=True)
        os.makedirs(speech_speaker_path, exist_ok=True)
        os.makedirs(noise_speaker_path, exist_ok=True)
        
        for i_pair, (utterance, noise) in enumerate(pairs_map[key]):
            data_utterance, sr = sf.read(utterance)
            data_utterance = librosa.resample(data_utterance, sr, FINAL_SR)
            data_utterance = np.pad(data_utterance, (0, max(0, CLIP_LEN_SAMPLES - len(data_utterance))))
        
            data_noise, sr = sf.read(noise)
            data_noise = librosa.resample(data_noise, sr, FINAL_SR)
            data_noise = np.pad(data_noise, (0, max(0, CLIP_LEN_SAMPLES - len(data_noise))))

            n_chunks = min(len(data_utterance), len(data_noise)) // CLIP_LEN_SAMPLES
            for i_chunk in range(n_chunks):
                cur_utterance = data_utterance[i_chunk * CLIP_LEN_SAMPLES:(i_chunk + 1) * CLIP_LEN_SAMPLES]
                cur_noise = data_noise[i_chunk * CLIP_LEN_SAMPLES:(i_chunk + 1) * CLIP_LEN_SAMPLES]

                cur_utterance_path = os.path.join(speech_speaker_path, 
                                             '{}_{}_id_{}.wav'.format(i_pair, 
                                                                      i_chunk, 
                                                                      os.path.splitext(os.path.basename(utterance))[0]))
                sf.write(cur_utterance_path, cur_utterance, FINAL_SR)

                fsd_id = os.path.splitext(os.path.basename(noise))[0]            
                noise_class = fsd_df.labels[fsd_df.fname == int(fsd_id)].values[0].split(',')[-1]
                cur_noise_path = os.path.join(noise_speaker_path, 
                                             '{}_{}_{}_id_{}.wav'.format(i_pair, 
                                                                         i_chunk, 
                                                                         noise_class, 
                                                                         fsd_id))
                sf.write(cur_noise_path, cur_noise, FINAL_SR)

def make_dataset(libri_dir, fsd50k_dir, librifsd50k_dir):
    train_speaker_map, val_speaker_map, test_speaker_map = get_libri_speaker_maps(libri_dir)
    print(' --- Loaded LibriSpeech Data ---')
    
    fsd_dev_metadata = os.path.join(fsd50k_dir, 'FSD50K.ground_truth/dev.csv')
    fsd_test_metadata = os.path.join(fsd50k_dir, 'FSD50K.ground_truth/eval.csv')

    fsd_df_dev = pd.read_table(fsd_dev_metadata, sep=',')
    fsd_df_test = pd.read_table(fsd_test_metadata, sep=',')

    fsd_train_files, fsd_val_files, fsd_test_files,  = get_nonspeech_fsd(fsd50k_dir, fsd_df_dev, fsd_df_test)
    print(' --- Loaded FSD50k Data ---')

    train_pairs = make_fsd_libri_pairs(train_speaker_map, fsd_train_files)
    val_pairs = make_fsd_libri_pairs(val_speaker_map, fsd_val_files)
    test_pairs = make_fsd_libri_pairs(test_speaker_map, fsd_test_files)
    
    librifsd50k_train_dir = os.path.join(librifsd50k_dir, 'train')
    librifsd50k_val_dir = os.path.join(librifsd50k_dir, 'val')
    librifsd50k_test_dir = os.path.join(librifsd50k_dir, 'test')
    
    print(' --- Writing LibriFSD50k Data ---')    
    print(f' --- Writing Train to {librifsd50k_train_dir} ---')
    write_pairs(train_pairs, librifsd50k_train_dir, fsd_df_dev)
    
    print(f' --- Writing Val. to {librifsd50k_val_dir} ---')
    write_pairs(val_pairs, librifsd50k_val_dir, fsd_df_dev)

    print(f' --- Writing Test to {librifsd50k_test_dir} ---')
    write_pairs(test_pairs, librifsd50k_test_dir, fsd_df_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", default="", help="the location you downloaded to when using get_librifsd50k.sh")
    parser.add_argument('--out_dir', default="", help="the location you want to write librifsd50k to")
    args = vars(parser.parse_args())
    
    download_dir = args['download_dir']
    libri_dir = os.path.join(download_dir, 'LibriSpeech')
    fsd50k_dir = os.path.join(download_dir, 'FSD50K')
    librifsd50k_dir = args['out_dir']
    
    # for reproducibility
    random.seed(10)
    np.random.seed(10)

    make_dataset(libri_dir, fsd50k_dir, librifsd50k_dir)
        