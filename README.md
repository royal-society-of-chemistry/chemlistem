# ChemListem: Chemical Named Entity Recognition with deep neural networks

ChemListem is a package for chemical named entity recognition, developed for the CEMP task of BioCreative V.5. ChemListem uses
deep learning, as implemented using the [keras](https://keras.io/) package, to do this. ChemListem also uses 
[scikit-learn](http://scikit-learn.org/stable/), [h5py](http://www.h5py.org/) and [numpy](http://www.numpy.org/), and has pre-trained
models that require the use of [TensorFlow](https://www.tensorflow.org/).

ChemListem is written in Python 3, and is known to be compatible with __Python 3.5__ and __Python 3.6__. It has been tested
on Windows 10, and Ubuntu 14.

## Installation

ChemListem may be installed from the PyPI via pip:

	pip install chemlistem
    
Note that this does not install tensorflow by default - you will have to install that yourself. `pip install tensorflow` will do. This
has been left out because there are other libraries that keras can use instead of tensorflow, and a gpu-enhanced version of 
tensorflow, and occasional version compatibility issues...

### Installing dependencies

The dependencies for ChemListem can sometimes be hard to install, especially on Windows. Keras depends upon either 
[Theano](https://github.com/Theano) or [TensorFlow](https://www.tensorflow.org/) which in turn depend upon BLAS.

At the time of writing there is a problem with the latest version of tensorflow - 1.11. We have found that using tensorflow 1.10 

We have found that the [WinPython](http://winpython.github.io/) distribution, which contains pre-built versions of the dependencies,
works well on Windows. The following procedure had previously been found to work, but may need updating:

1. Obtain WinPython 3.5.3.1Qt5-64bit, and install it.
2. Update the keras package, using `pip install --upgrade --no-deps keras`
3. Install ChemListem

The pre-trained models for chemlistem were trained using the TensorFlow backend. If you are already using keras, then ensure
that keras is set up to use TensorFlow - alternatively, if you need to use Theano, consider compiling your own model files (see
"Training" below).

## Usage

ChemListem uses three models - a "traditional" model, a "minimalist" model and an ensemble model that combines the two. The following
example shows how to use the ensemble model:

    from chemlistem import get_ensemble_model
    model = get_ensemble_model()
    results = model.process("The morphine was dissolved in ethyl acetate.")
    print(results)

The output should be as follows:

    [(4, 12, 'morphine', 0.9738021492958069, True), (30, 43, 'ethyl acetate', 0.9788203537464142, True)]

The output is a list of lists, each sub-list corresponding to a chemical named entity.

0. The start character position.
1. The end character position.
2. The string of the entity.
3. The score of the entity - i.e. how confident chemlistem is that the entity is a true entity. 1.0 = maximum confidence, 0.0 = 
minimum confidence.
4. Whether the entity is "dominant" i.e. not overlapping with a higher-score entity.

(There may also be some messages from TensorFlow talking about an "unknown op" - these can usually be ignored.)

ChemListem can be tuned for precision or recall by varying a threshold that confidence scores are tested against. Furthermore,
chemlistem can be set up to report overlapping guesses. For example:

    results = model.process("The morphine was dissolved in ethyl acetate.", 0.00001, False)
    for r in results: print(r)

gives:

    (0, 12, 'The morphine', 0.00017620387734496035, False)
    (4, 12, 'morphine', 0.9738021492958069, True)
    (4, 16, 'morphine was', 0.00012143117555751815, False)
    (4, 26, 'morphine was dissolved', 7.890002598287538e-05, False)
    (4, 43, 'morphine was dissolved in ethyl acetate', 1.1027213076886255e-05, False)
    (4, 44, 'morphine was dissolved in ethyl acetate.', 1.1027213076886255e-05, False)
    (13, 16, 'was', 1.1566731700440869e-05, True)
    (17, 26, 'dissolved', 4.46309641120024e-05, True)
    (17, 43, 'dissolved in ethyl acetate', 1.0192422223553876e-05, False)
    (17, 44, 'dissolved in ethyl acetate.', 1.0192422223553876e-05, False)
    (27, 35, 'in ethyl', 1.8829327018465847e-05, False)
    (27, 43, 'in ethyl acetate', 0.00015375280418084003, False)
    (27, 44, 'in ethyl acetate.', 3.4707042686932255e-05, False)
    (30, 35, 'ethyl', 3.010855562024517e-05, False)
    (30, 43, 'ethyl acetate', 0.9788203537464142, True)
    (30, 44, 'ethyl acetate.', 3.4707042686932255e-05, False)
    (36, 43, 'acetate', 7.422585622407496e-05, False)
    (36, 44, 'acetate.', 1.424002675776137e-05, False)
    
The second argument to `process` is the threshold. The third argument is whether to exclude non-dominant entities or not. For example,
`model.process("The morphine was dissolved in ethyl acetate.", 0.00001, True)` gives:

    (4, 12, 'morphine', 0.9738021492958069, True)
    (13, 16, 'was', 1.1566731700440869e-05, True)
    (17, 26, 'dissolved', 4.46309641120024e-05, True)
    (30, 43, 'ethyl acetate', 0.9788203537464142, True)

To use the traditional and minimalist models, use `get_trad_model` or `get_mini_model` instead of `get_ensemble_model`.

There are fast versions of these models which need to be run with a CUDNN-enabled GPU. To load these models, use
`get_ensemble_model(gpu=True)` etc.

If you wish to process multiple lines quickly, there is a batchprocess method which accepts a list of strings, and gives a list of 
results. Neural networks run faster if they can process several items in parallel so using batch processing can give a speed increase.
For example:

    tm = chemlistem.get_trad_model()
    results = tm.batchprocess(["This is ethyl acetate and ethanol.", "This is codeine and morphine."], 0.5, False)
    for r in results:
        for l in r:
            print(l)
        print("---")

gives:

    (8, 21, 'ethyl acetate', 0.9956316, True)
    (26, 33, 'ethanol', 0.975327, True)
    ---
    (8, 15, 'codeine', 0.98441076, True)
    (20, 28, 'morphine', 0.9720572, True)
    ---
	
## Training

ChemListem is bundled with model files. However, you may wish to train your own. The training data is available
[here](http://www.becalm.eu/pages/biocreative) - you will need to get the GPRO & CEMP training set 2016, and extract the files
`BioCreative V.5 training set.txt` and `CEMP_BioCreative V.5 training set annot.tsv`. Optionally, you may also wish to obtain the
[GloVe](https://nlp.stanford.edu/projects/glove/) pre-trained word vectors - get `glove.6B.zip`, unzip it and find
`glove.6B.300d.txt`.

The traditional model may be trained using the following example:

    from chemlistem import tradmodel
	tm = tradmodel.TradModel()
    tm.train("BioCreative V.5 training set.txt", "CEMP_BioCreative V.5 training set annot.tsv", "D:/glove.6B/glove.6B.300d.txt", "tradtest")

If you have a CUDNN-enabled GPU, you can include the option `gpu=True` in the `tm.train` call. This should speed up training. Note
that models trained in this manner cannot be directly used on non-GPU-enabled systems.

If you do not wish to use GloVe, then instead of `"D:/glove.6B/glove.6B.300d.txt"` write `None`.

Alternatively, in the BitBucket repository there is the file `vectors_patents.zip`. Download it and unzip it and use the text
file therein instead of the GloVe file. This has been prepared specially from pharmaceutical patent abstracts, with ChemListem's
tokenisation and capitalisation controls, and gives better results.
	
This process will take several hours to run. It should eventually produce two files: `tradmodel_tradtest.h5` and
`tradmodel_tradtest.json` (also several files of the form `epoch_*_tradtest.h5`). These are your model files. To use them, follow
this example:

	from chemlistem import tradmodel
	tm = tradmodel.TradModel()
	tm.load("tradmodel_tradtest.json", "tradmodel_tradtest.h5")
	print(tm.process("This test includes morphine.")

Training and loading the minimalist model is similar - however, this takes several days, does not use GloVe, and does not produce a
JSON file. Examples:

    from chemlistem import minimodel
	mm = minimodel.MiniModel()
    mm.train("BioCreative V.5 training set.txt", "CEMP_BioCreative V.5 training set annot.tsv", "minitest")

    from chemlistem import minimodel
	mm = minimodel.MiniModel()
	mm.load("minimodel_minitest.json", "minimodel_minitest.h5")
	print(mm.process("This test includes morphine.")

Once you have produced these two models, then you may produce an ensemble model. Example:

	from chemlistem import tradmodel, minimodel, ensemblemodel
	tm = tradmodel.TradModel()
	tm.load("tradmodel_tradtest.json", "tradmodel_tradtest.h5")
	mm = minimodel.MiniModel()
	mm.load("minimodel_minitest.json", "minimodel_minitest.h5")
	em = ensemblemodel.EnsembleModel(tm, mm)
	print(em.process("This test includes morphine.")
    
The train method here has several methods:

* `gpu` - set this to True for fast training, as per the same option in the traditional system.
* `unsupfile` - give this the filename of a file containing sentences from patent abstracts. In the BitBucket repository there is a
file called `patent_lines.zip` which is good for this.
* `nunsup` - how many lines to use from the file - if this is larger than the number of lines in the file, it will use some or all of
the lines more than once. 0 is no unsupervised learning. -1 is all the lines, once only.
* `unsupcfg` - this contains options to control when the various unsupervised learning techniques take place. See the docstring for
more details, or just leave it unset - there is a good default.

## Differences from published versions.

The system here is as described in a forthcoming full-text journal paper. Due to version compatibility difficulties, the model files
supplied are not exactly the same as those used for publication - they were built using the same code with the same hyperparameters,
but the random initialisation was different, giving slightly different results.

There are also some minor difference from the version used in the original BioCreative V.5 submission.



## BioCreative VI

This repository also contains the source code for our entry in BioCreative VI Task 5, in the subdirectory cl_bc6_chemprot

## Conclusion

ChemListem has been developed by the Data Science group at the Royal Society of Chemistry.

ChemListem is distributed under the MIT License - see License.txt.

There is a paper describing ChemListem - *Peter Corbett, John Boyle. “Chemlistem - chemical named entity recognition using recurrent
neural networks”. Proceedings of the BioCreative V.5 Challenge Evaluation Workshop (2017): 61-68.* published as part of the
BioCreative V.5 proceedings, and there is a journal paper forthcoming. If you use chemlistem in your work, please cite it.

I would like to thank Adam Bernard for the initial Python translation of the tokeniser that chemlistem uses.

Peter Corbett, 2016-2018
The Royal Society of Chemistry