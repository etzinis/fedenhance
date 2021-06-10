#!/bin/bash

# RUN THIS WITH ./get_librifsd50k.sh /dir_to_download_in

# make the download directory
mkdir -p $1

# get all the speech data
wget https://www.openslr.org/resources/12/dev-clean.tar.gz -P $1
wget https://www.openslr.org/resources/12/test-clean.tar.gz -P $1
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz -P $1
wget https://www.openslr.org/resources/12/train-clean-360.tar.gz -P $1

# unzip and organize it all
tar xvzf $1/'dev-clean.tar.gz' --directory $1

tar xvzf $1/'test-clean.tar.gz' --directory $1/LibriSpeech
mv $1/LibriSpeech/LibriSpeech/test-clean $1/LibriSpeech/test-clean
rm -r $1/LibriSpeech/LibriSpeech/

tar xvzf $1/'train-clean-100.tar.gz' --directory $1/LibriSpeech
mv $1/LibriSpeech/LibriSpeech/train-clean-100 $1/LibriSpeech/train-clean-100
rm -r $1/LibriSpeech/LibriSpeech/

tar xvzf $1/'train-clean-360.tar.gz' --directory $1/LibriSpeech
mv $1/LibriSpeech/LibriSpeech/train-clean-360 $1/LibriSpeech/train-clean-360
rm -r $1/LibriSpeech/LibriSpeech/

# get the noise data
wget https://zenodo.org/record/4060432/files/FSD50K.ground_truth.zip -P $1
wget https://zenodo.org/record/4060432/files/FSD50K.eval_audio.zip -P $1
wget https://zenodo.org/record/4060432/files/FSD50K.eval_audio.z01 -P $1

wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.zip -P $1
for i in {1..5}
do
  wget https://zenodo.org/record/4060432/files/FSD50K.dev_audio.z0$i -P $1
done

mkdir -p $1/FSD50K

unzip $1/FSD50K.ground_truth.zip -d $1/FSD50K

# combine the zips
zip -s 0 $1/FSD50K.eval_audio.zip --out $1/eval_audio.zip
unzip $1/eval_audio.zip -d $1/FSD50K


# combine the zips
zip -s 0 $1/FSD50K.dev_audio.zip --out $1/dev_audio.zip
unzip $1/dev_audio.zip -d $1/FSD50K

