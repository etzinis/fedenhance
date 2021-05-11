# Separate but Together: Unsupervised Federated Learning for Speech Enhancement from Non-IID Data

TLDR; The main contribution of this paper is to develop a federated learning (FL) system which is capable of learnign a sound separation model when mixtures are distributed in a non-IID fashion across multiple clients.

Specifically, we make the following contributions:

1. We provide the first completely unsupervised system which is capable to be trained in a federated learning fashion for speech enhancement.
2. We show that we can expedite the convergence and boost the overall performance of our FL method by transferring knowledge from another medium-size speech enhancement dataset like WHAM.
3. We can effectively combine updates from clients with supervised and unsupervised data using different loss functions.
4. We provide the recipes for the creation of a benchmark LibriFSD50K in order to measure our performance.


## Table of contents

- [Fedenhance results](#fedenhance-results)
- [Datasets Generation](#datasets-generation)
- [How to run](#how-to-run)
- [Copyright and license](#copyright-and-license)


## Fedenhance results

As we discuss in the paper, our main objective is to find efficient architectures not only in terms of one metric but in terms of all metrics which might become a bottleneck during training or inference. This will facilitate the needs of users that do not have in their disposal (or use case) the considerable requirements that many modern models exhibit. This will enable people with no GPU access, or users with interest in edge applications to also make use of this model and not be locked out of good performance.

We present here the results from our paper:

![ESC-50-results](images/Selection_061.png "ESC-50-results")
SI-S

## Datasets Generation
Stay tuned!


## How to run

1. Setup your cometml credentials and paths for datasets in the config file.
```shell
vim ./fedenhance/__config__.py
```

2. Set the environment variable.
```shell
export PYTHONPATH=/your_directory_path/fedenhance/:$PYTHONPATH
```

3. Run a federated learning experiment with all nodes unsupervised. 

```shell
cd fedenhance/experiments
python run_federated_unsup_sep_enh_v2.py \
--n_global_epochs 50000 --model_type sudo_groupcomm_v2 \
--enc_kernel_size 41 --out_channels 256 --enc_num_basis 512 --in_channels 512 --num_blocks 8 \
--learning_rate 0.001 -bs 6 --divide_lr_by 2. --patience 0 --clip_grad_norm 5. --optimizer adam -cad 0 \
--audio_timelength 4. --max_num_sources 3 --project_name fedenhance -tags fedenhance_is_the_best \
--n_fed_nodes 16 --p_supervised 0.25 --available_speech_percentage 0.5 --p_single_mix 0.0 --p_available_users 0.25
```


## Copyright and license
University of Illinois Open Source License

Copyright © 2020, University of Illinois at Urbana Champaign. All rights reserved.

Developed by: Efthymios Tzinis 1, Jonah Casebeer 1, Zhepei Wang 1 and Paris Smaragdis 1,2

1: University of Illinois at Urbana-Champaign 

2: Adobe Research 

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal with the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions: Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimers. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimers in the documentation and/or other materials provided with the distribution. Neither the names of Computational Audio Group, University of Illinois at Urbana-Champaign, nor the names of its contributors may be used to endorse or promote products derived from this Software without specific prior written permission. THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH THE SOFTWARE.
