Horse Taming for Singing Voice Detection with Neural Networks
=============================================================

This is the implementation for an experiment presented in the "Extensions
and Dead Ends" section of the "Singing Voice Detection" chapter in my Phd
thesis titled "Deep Learning for Event Detection, Sequence Labelling and
Similarity Estimation in Music Signals" (Section 9.8; to appear).

As demonstrated in the [`singing_horse`](//github.com/f0k/singing_horse)
repository, the most salient feature used by the networks to detect singing
voice are sloped or wiggly lines in a spectrogram. While this gives
state-of-the-art results, this is not exactly what singing voice is about, and
thus the classifier could be called a *horse* (a term
[coined by Bob Sturm](https://doi.org/10.1109/TMM.2014.2330697) in reference to ["Clever Hans"](https://en.wikipedia.org/wiki/Clever_Hans)).

The experiment in this repository implements a data augmentation scheme that
scribbles wiggly lines into the mel spectrogram processed by the network, in
order to train it to ignore these lines and possibly develop a better concept
of singing voice than "sloped lines in a spectrogram".

For the baseline experiments of my ISMIR 2015 paper, see the
[`master`](//github.com/f0k/ismir2015) branch, and for other follow-up
experiments of my PhD thesis, see the
[`phd_extra`](//github.com/f0k/ismir2015/tree/phd_extra) branch.


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
`unhorse` branch:
```bash
git clone https://github.com/f0k/ismir2015.git
git checkout unhorse
```
If you do not have `git` available, download the code from
https://github.com/f0k/ismir2015/archive/unhorse.zip and extract it.

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

Compared to the ISMIR 2015 paper and the `master` branch, the experiments use
the improved architecture from my ISMIR 2016 paper ("Learning to Pinpoint
Singing Voice from Weakly-Labeled Examples",
[[Paper](http://ofai.at/~jan.schlueter/pubs/2016_ismir.pdf),
[BibTeX](http://ofai.at/~jan.schlueter/pubs/2016_ismir.bib)), but still applied
to mel spectrograms. Compared to the other follow-up experiments in the
`phd_extra` branch, spectrograms are mel-scaled, standardized and augmented
before they reach the network, not mel-scaled and batch-normalized within the
network (this would have required implementing the augmentation in Theano).
So we first train a baseline, then a version with sloped lines augmentation.

### baseline

To train and evaluate the baseline network with five repetitions, run the
following in a terminal in the cloned or extracted repository:
```bash
cd experiments
mkdir unhorse
for r in {1..5}; do OMP_NUM_THREADS=1 ./train.py --cache=/tmp unhorse/baseline$r.npz; done
for r in {1..5}; do ./predict.py --cache=/tmp unhorse/baseline$r{,.pred}.npz; done
for r in {1..5}; do ./eval.py unhorse/baseline$r.pred.npz; done
```
In my case, this gave a classification error of 7.3(±0.5)%.

### sloped lines augmentation

To augment 10% of examples with artificial sloped lines, run:
```bash
for r in {1..5}; do OMP_NUM_THREADS=1 ./train.py --scribble=0.1 --cache=/tmp unhorse/scribble$r.npz; done
for r in {1..5}; do ./predict.py --cache=/tmp unhorse/scribble$r{,.pred}.npz; done
for r in {1..5}; do ./eval.py unhorse/scribble$r.pred.npz; done
```
In my case, this gave a classification error of 7.5(±0.2)%. So if anything,
results got worse, not better.

### qualitative comparison

To check whether the network actually learned to ignore wiggly lines, we can
have a look at its predictions for altered test examples. For the baseline:
```bash
./predict.py --cache=/tmp unhorse/baseline1.npz /tmp/foo.npz --plot --scribble
```
You will see that the baseline mistakes the artificial wiggly lines for singing
voice, as shown in Figure 9.14b of my thesis. Note that this requires
[matplotlib with an interactive backend](https://matplotlib.org/faq/usage_faq.html#what-is-a-backend)
(if in doubt, it should be fine if you run this on your desktop, but for a
headless server, you will need to change the `plt.show()` calls to
`plt.savefig('yourfilename.png')`, for example).

For a network trained to ignore wiggly lines:
```bash
./predict.py --cache=/tmp unhorse/scribble1.npz /tmp/foo.npz --plot --scribble
```
You should see that the network indeed ignores the wiggly lines, as shown in
Figure 9.14c of my thesis. Alas, it did not force the network to find better
cues for singing voice detection, just to ignore the specific augmentation.


About...
--------

### ... the code

This code is a direct fork from the `master` branch (i.e., the ISMIR 2015
reimplementation) which just changes the architecture and adds another
augmentation method.

### ... the results

Results will vary depending on the random initialization of the networks. Even
with fixed random seeds, results will not be exactly reproducible due to the
multi-threaded data augmentation. Furthermore, when training on GPU with cuDNN,
the backward pass is nondeterministic by default, introducing further noise.
