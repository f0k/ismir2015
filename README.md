Follow-up Experiments for Singing Voice Detection with Neural Networks
======================================================================

This is the implementation for most experiments presented in the "Extensions
and Dead Ends" section of the "Singing Voice Detection" chapter in my Phd
thesis titled "Deep Learning for Event Detection, Sequence Labelling and
Similarity Estimation in Music Signals" (Section 9.8; to appear).

Specifically, it includes experiments for learning a magnitude transformation
of the input spectrograms, for learning the center frequencies of a mel
filterbank, and for comparing different variants of convolutional dropout.

For the baseline experiments of my ISMIR 2015 paper, see the
[`master`](//github.com/f0k/ismir2015) branch, and for experiments on training
a network to not be irritated by wiggly lines, see the
[`unhorse`](//github.com/f0k/ismir2015/tree/unhorse) branch. For a
demonstration on how the networks can be fooled with hand-drawn wiggly lines,
see the [`singing_horse`](//github.com/f0k/singing_horse) repository.


Preliminaries
-------------

The code requires the following software:
* Python 2.7+ or 3.4+
* Python packages: numpy, scipy, Theano, Lasagne
* bash or a compatible shell with wget and tar
* ffmpeg or avconv

For better performance, the following Python packages are recommended:
* pyfftw (for much faster spectrogram computation)
* scipy version 0.15+ (to allow time stretching and pitch shifting
  augmentations to be parallelized by multithreading, not only by
  multiprocessing, https://github.com/scipy/scipy/pull/3943)

For Theano and Lasagne, you may need the bleeding-edge versions from github.
In short, they can be installed with:
```bash
pip install --upgrade --no-deps https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade --no-deps https://github.com/Lasagne/Lasagne/archive/master.zip
```
(Add `--user` to install in your home directory, or `sudo` to install globally.)
For more complete installation instructions including GPU setup, please refer
to the [From Zero to Lasagne](https://github.com/Lasagne/Lasagne/wiki/From-Zero-to-Lasagne)
guides.

On Ubuntu, pyfftw can be installed with the following two commands:
```bash
sudo apt-get install libfftw3-dev
pip install pyfftw
```


Setup
-----

For preparing the experiments, clone the repository somewhere and checkout the
`phd_extra` branch:
```bash
git clone https://github.com/f0k/ismir2015.git
git checkout phd_extra
```
If you do not have `git` available, download the code from
https://github.com/f0k/ismir2015/archive/phd_extra.zip and extract it.

The experiments use the public [Jamendo dataset by Mathieu Ramona](www.mathieuramona.com/wp/data/jamendo/).
To download and prepare it, open the cloned or extracted repository in a
bash terminal and execute the following scripts (in this order):
```bash
./datasets/jamendo/audio/recreate.sh
./datasets/jamendo/filelists/recreate.sh
./datasets/jamendo/labels/recreate.sh
```
The dataset format is the same as used in the `master` branch; so in case you
tried that already, you can skip this step (and just checkout `phd_extra`).


Experiments
-----------

To train the networks for learned magnitude transformations, learned mel
filterbanks and convolutional dropout variants, simply run:
```bash
cd experiments
mkdir cache
./train_all.sh
```
This will train 16 network variants with 5 repetitions each. On an Nvidia GTX
970 GPU, a single training run will take about 30 minutes. Spectrograms will
be cached in the `cache` directory -- if you'd like to store them in `/tmp`
instead, replace `mkdir cache` with `ln -s /tmp cache` or edit the
`train_all.sh` file where it says `--cache=cache`.

If you have multiple GPUs, you can distribute runs over these GPUs by running
the script multiple times in multiple terminals with different target devices,
e.g., `THEANO_FLAGS=device=cuda1 ./train_all.sh`. If you have multiple servers
that can access the same directory via NFS, you can also run the script on
each server for further distribution of runs (runs are blocked with lockfiles).

The script will also compute network predictions after each training run. If
this failed for some jobs for some reasons, run:
```bash
./predict_missing.sh
```
This will compute any missing network predictions (if none are missing, nothing
happens).

### baseline

To obtain results for the baseline network, run:
```bash
./eval_all.sh spectlearn/allfixed
```
This will produce results for two variants; one using a fixed mel filterbank
compatible to yaafe, one using a mel filterbank layer within the network, but
without training the mel filterbank layer. The filterbanks are not exactly
identical, but should achieve similar results; in my case I got 6.5(±0.3)%.
For each experiment, the evaluation script determines the optimal
classification threshold on the validation set and uses it to evaluate
performance on the test set.

### learned magnitude transformation

For a learned magnitude transformation of the form `log(1 + 10**a * x)`, run:
```bash
./eval_all.sh spectlearn/maglearn_log1p0
```
This will give results with `a` initialized to 0, and three different learning
rate boosts (1, 10, 50). For `a` initialized to 7 (which recovers a function
similar to `log(max(10**-7, x))` used in my ISMIR 2015 paper), run:
```bash
./eval_all.sh spectlearn/maglearn_log1p7
```
To reproduce Figure 9.12 (left) of my thesis, which shows the evolution of `a`
over training time, you can use `matplotlib`. In a Python session, run:
```python
import numpy as np
from matplotlib import pyplot as plt
plt.figure()
for name in 'log1p0', 'log1p0_boost10', 'log1p7_boost10':
    for r in range(1, 6):
        plt.plot(np.load('spectlearn/maglearn_%s_%d.hist.npz' % (name, r))['param0'])
plt.show()  # or plt.savefig('yourfilename.pdf')
```

For a learned magnitude transformation of the form `x**sigm(a)`, run:
```bash
./eval_all.sh spectlearn/maglearn_pow
```
Again, it will give results with learning rate boosts 1, 10, 50, and the
change of `a` over training time can be plotted with about the same commands.

As reported in the thesis, results will be close to the baseline, potentially
a bit worse, not better.

### learned mel filterbank

For results with learned mel filterbanks, run:
```bash
./eval_all.sh spectlearn/mellearn
```
This will give results for mel filterbanks that have the array of 82
frequencies required to produce 80 overlapping triangular filters initialized
with equal distances on the mel scale (as usual for a mel filterbank), with
the individual distances becoming learnable network parameters (measured in
mel). Again, results will be reported with different learning rate boosts, and
tend to be worse than the baseline network.

To reproduce Figure 9.13, in a Python session, run:
```python
import numpy as np
from matplotlib import pyplot as plt
plt.figure()
plt.plot(np.load('spectlearn/mellearn_boost50_1.hist.npz')['param0'].cumsum(axis=-1))
plt.show()  # or plt.savefig('yourfilename.pdf')
```
This shows the evolution of the 82 frequencies over training time (in mel).

### sloped lines augmentation

Just in case you are reading Section 9.8 of my thesis along with this document,
the sloped lines augmentation experiments can be found in the
[`unhorse`](//github.com/f0k/ismir2015/tree/unhorse) branch.

### dropout variants

For the results comparing individual dropout, channelwise dropout and bandwise
dropout in the convolutional layers, run:
```bash
./eval_all.sh dropout/
```
Channelwise dropout improved results over the baseline for me, the others
deteriorated results. It is possible that the dropout in front of the first
dense layer should be channelwise instead of individual as well.

### bagging

As mentioned in Section 9.9, if just looking for improved results, an easy way
is to bag predictions of multiple networks. For example, to bag the five
networks using spatial (channelwise) dropout, run:
```bash
./eval.py dropout/channels*pred.pkl
```
This gave an error of 0.59 in my experiments.

To bag the ten baseline networks, run:
```bash
./eval.py spectlearn/allfixed*pred.pkl
```
This gave an error of 0.58:
```
thr: 0.69, prec: 0.942, rec: 0.932, spec: 0.950, f1: 0.937, err: 0.058
```
To the best of my knowledge, this improves the state of the art for Jamendo.

To bag everything, run:
```bash
./eval.py */*pred.pkl
```
This also gave an error of 0.58.


About...
--------

### ... the code

This is an extension of the code published for the ISMIR 2015 experiments. It
adds a lot of bells and whistles, but also makes the code more complex to
follow -- use the commit history to better understand the changes.

This code can serve as a template for your own sequence labelling experiments.
Some interesting features are:
* Datasets can easily be added to the `datasets` directory and their name be
  passed as the `--dataset` argument of `train.py`, `predict.py` and `eval.py`.
* With `--load-spectra=on-demand`, `train.py` can efficiently handle datasets
  too large to fit into main memory; when placing the spectrogram cache on a
  SSD, it can be fast enough to not stall the GPU.
* Both `train.py` and `predict.py` accept key-value settings in the form
  `--var key=value`, these can be thought of global environment variables that
  you can access anywhere in the code. The `defaults.vars` files stores
  default key-value settings to be overridden via `--var key=value`, and
  `train.py` stores the settings used for training along with the trained
  network model, automatically obeyed by `predict.py`. This allows to easily
  extend any part of the code without breaking existing experiments: implement
  a new feature conditioned on a key-value setting, add this setting to
  `defaults.vars` such that default behaviour stays the same, add a new line
  to `train_all.sh` that passes different `--var key=value` settings, and run
  the script. See the changeset of commit @c6634d3 for an example.

The advantage of using such a template over creating a more generic experiment
framework is that you have direct control of all dataset preparation, model
creation and training code and a low hurdle to modify things, the disadvantage
is that it becomes harder to reuse code.

### ... the results

Results will vary depending on the random initialization of the networks. Even
with fixed random seeds, results will not be exactly reproducible due to the
multi-threaded data augmentation. Furthermore, when training on GPU with cuDNN,
the backward pass is nondeterministic by default, introducing further noise.

Furthermore, as argued in Section 9.9 of my thesis, a dataset that only
consists of songs containing singing could make it too easy for a singing voice
detector. I used the Jamendo dataset because it is easily and freely available,
but take results with a grain of salt.
