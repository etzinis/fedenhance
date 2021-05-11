import os
import glob2
from shutil import copyfile
import tqdm
import argparse

WHAM_SPLITS = {'tt':'test', 'tr':'train', 'cv':'val'}

def make_dataset(wham_dir, fed_wham_dir):
    for mode in WHAM_SPLITS.keys():
        print(f' --- Working on {mode}, {WHAM_SPLITS[mode]} ---')
        
        s1_path = os.path.join(wham_dir, mode, 's1')
        noise_path = os.path.join(wham_dir, mode, 'noise')

        all_speech = glob2.glob(s1_path + '/*.wav')
        all_noise = glob2.glob(noise_path + '/*.wav')

        all_speech.sort()
        all_noise.sort()

        cur_dir = os.path.join(fed_wham_dir, WHAM_SPLITS[mode])
        os.makedirs(cur_dir, exist_ok=True)

        speaker_dict = {}
        for i in tqdm.tqdm(range(len(all_speech))):
            speech_basename = os.path.basename(all_speech[i])
            noise_basename = os.path.basename(all_noise[i])

            assert speech_basename == noise_basename

            speaker = speech_basename.split('_')[0][:3]
            speaker_dir = os.path.join(cur_dir, speaker)

            if speaker not in speaker_dict:
                speaker_dict[speaker] = 0
                os.makedirs(os.path.join(speaker_dir, 'speech'), exist_ok=True)
                os.makedirs(os.path.join(speaker_dir, 'noise'), exist_ok=True)
            else:
                speaker_dict[speaker] += 1

            new_speech_name = f'{speaker_dict[speaker]}_id_{speaker}.wav'
            new_noise_name = f'{speaker_dict[speaker]}.wav'

            copyfile(all_speech[i], os.path.join(speaker_dir, 'speech', new_speech_name))
            copyfile(all_noise[i], os.path.join(speaker_dir, 'noise', new_noise_name))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wham_dir", default="", help="The wham/wav16k/max directory")
    parser.add_argument('--out_dir', default="", help="The dir to make and write fed_wham to")
    args = vars(parser.parse_args())
    
    wham_dir = args['wham_dir']
    fed_wham_dir = args['out_dir']

    make_dataset(wham_dir, fed_wham_dir)
