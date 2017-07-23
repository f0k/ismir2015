Exploring Data Augmentation for Improved Singing Voice Detection with Neural Networks
=====================================================================================

This is a reimplementation of the experiments presented in the paper "Exploring
Data Augmentation for Improved Singing Voice Detection with Neural Networks" by
Jan Schlüter and Thomas Grill at the 16th International Society for Music
Information Retrieval Conference (ISMIR 2015).
[[Paper](http://ofai.at/~jan.schlueter/pubs/2015_ismir.pdf),
[BibTeX](http://ofai.at/~jan.schlueter/pubs/2015_ismir.bib)]

For follow-up experiments described in my PhD thesis, see the
[`phd_extra`](//github.com/f0k/ismir2015/tree/phd_extra) branch, and for
experiments on training a network to not be irritated by wiggly lines, see the
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

For preparing the experiments, clone the repository somewhere:
```bash
git clone https://github.com/f0k/ismir2015.git
```
If you do not have `git` available, download the code from
https://github.com/f0k/ismir2015/archive/master.zip and extract it.

The experiments use the public [Jamendo dataset by Mathieu Ramona](www.mathieuramona.com/wp/data/jamendo/).
To download and prepare it, open the cloned or extracted repository in a
bash terminal and execute the following scripts (in this order):
```bash
./datasets/jamendo/audio/recreate.sh
./datasets/jamendo/filelists/recreate.sh
./datasets/jamendo/labels/recreate.sh
```


Experiments
-----------

Table 1 in the paper shows results for Jamendo without data augmentation,
with train-time augmentation (combining pitch-shifting, time-stretching and
random frequency filtering), with test-time augmentation (pitch-shifting only)
and with both train-time and test-time augmentation.

### w/o augmentation

To reproduce results without augmentation, run the following in a terminal in
the cloned or extracted repository:
```bash
cd experiments
python train.py --no-augment --cache=/tmp jamendo_noaugment.npz
python predict.py --cache=/tmp jamendo_noaugment.{,pred.}npz
python eval.py jamendo_noaugment.pred.npz
```
The `--cache=/tmp` option will store the spectrograms in `/tmp` so they do not
have to be recomputed for further runs. You can pass any directory there, or
omit this option to always compute them on-the-fly (this will add less than a
minute to training, and less than half a minute to computing the predictions).
Total space requirements for the spectrograms are about 3.2 GiB.

The training code will produce two files: `jamendo_meanstd.npz`, storing the
statistics needed to standardize the data, computed on the training set, and
`jamendo_noaugment.npz`, storing the weights of the trained network.

The second command reads the network weights, computes predictions for all
files of the validation and test set, and stores them in
`jamendo_noaugment.pred.npz`.

Finally, the third command reads the predictions, preprocesses them, optimizes
the threshold on the validation set and reports results on the test set.

Each command can be run with `--help` for documentation on further options.

### train augmentation

To reproduce results with train-time augmentation, run:
```bash
OMP_NUM_THREADS=1 python train.py --augment --cache=/tmp jamendo_augment.npz
python predict.py --cache=/tmp jamendo_augment{,.pred}.npz
python eval.py jamendo_augment.pred.npz
```

The only change is that `--augment` is activated for training. By default, data
augmentation will happen on CPU in three background threads running in parallel
to the training thread. Change `bg_threads` or `bg_processes` in `train.py` if
this is not what you want (this is not exposed as a command line argument). The
`OMP_NUM_THREADS=1` environment variable setting prevents the background
threads from using multi-threaded BLAS routines, which would slow things down.

### test augmentation

For test-time augmentation, run:
```bash
python predict.py --cache=/tmp --pitchshift=+10 jamendo_noaugment{,_p10.pred}.npz
python predict.py --cache=/tmp --pitchshift=+20 jamendo_noaugment{,_p20.pred}.npz
python predict.py --cache=/tmp --pitchshift=-10 jamendo_noaugment{,_m10.pred}.npz
python predict.py --cache=/tmp --pitchshift=-20 jamendo_noaugment{,_m20.pred}.npz
python eval.py jamendo_noaugment{,_p10,_p20,_m10,_m20}.pred.npz
```

This computes predictions for the first network with files pitch-shifted by
+10, +20, -10 and -20 percent, then bags the predictions (along with the
non-shifted ones) for evaluation.

### train/test augmentation

For train-time and test-time augmentation, run:
```bash
python predict.py --cache=/tmp --pitchshift=+10 jamendo_augment{,_p10.pred}.npz
python predict.py --cache=/tmp --pitchshift=+20 jamendo_augment{,_p20.pred}.npz
python predict.py --cache=/tmp --pitchshift=-10 jamendo_augment{,_m10.pred}.npz
python predict.py --cache=/tmp --pitchshift=-20 jamendo_augment{,_m20.pred}.npz
python eval.py jamendo_augment{,_p10,_p20,_m10,_m20}.pred.npz
```

Similar to the previous step, this uses a network we trained before and
applies it to different pitch-shifted versions.


About...
--------

### ... the code

This is not the code used for the original paper, but a compacted
reimplementation. It is not perfectly identical (e.g., the original experiments
use zero-padding of input files during training, while this implementation
discards the borders for training and only pads for testing), but very close.
It is written to be easy to read and pick out parts for reuse (obeying the
license), not so much as a generic starting point for own experiments.
For a more feature-complete starting point, see the
[`phd_extra`](//github.com/f0k/ismir2015/tree/phd_extra) branch.

### ... the results

Results will vary depending on the random initialization of the networks. Even
with fixed random seeds, results will not be exactly reproducible due to the
multi-threaded data augmentation. Furthermore, when training on GPU with cuDNN,
the backward pass is nondeterministic by default, introducing further noise.
For more reliable comparison between the four variants, each experiment should
be repeated at least five times, to compute averages and confidence intervals.
