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
	
### Installing dependencies

The dependencies for ChemListem can sometimes be hard to install, especially on Windows. Keras depends upon either 
[Theano](https://github.com/Theano) or [TensorFlow](https://www.tensorflow.org/) which in turn depend upon BLAS.

We have found that the [WinPython](http://winpython.github.io/) distribution, which contains pre-built versions of the dependencies,
works well on Windows. The following procedure has been found to work at the time of writing:

1. Obtain WinPython 3.5.3.1Qt5-64bit, and install it.
2. Update the keras package, using `pip install --upgrade --no-deps keras`
3. Install ChemListem

The pre-trained models for chemlistem were trained using the TensorFlow backend. If you are already using keras, then ensure
that keras is set up to use TensorFlow - alternatively, if you need to use Theano, consider compiling your own model files (see "Training"
below).

## Usage

ChemListem uses three models - a "traditional" model, a "minimalist" model and an ensemble model that combines the two. The following
example shows how to use the ensemble model:

    from chemlistem import get_ensemble_model
    model = get_ensemble_model()
    results = model.process("The morphine was dissolved in ethyl acetate.")
    print(results)


The output should be as follows:

    [[4, 12, 'morphine', 0.96779683232307434, True], [30, 43, 'ethyl acetate', 0.97885456681251526, True]]

The output is a list of lists, each sub-list corresponding to a chemical named entity.

0. The start character position.
1. The end character position.
2. The string of the entity.
3. The score of the entity - i.e. how confident chemlistem is that the entity is a true entity. 1.0 = maximum confidence, 0.0 = minimum
confidence.
4. Whether the entity is "dominant" i.e. not overlapping with a higher-score entity.

(There may also be some messages from TensorFlow talking about an "unknown op" - these can usually be ignored.)

ChemListem can be tuned for precision or recall by varying a threshold that confidence scores are tested against. Furthermore,
chemlistem can be set up to report overlapping guesses. For example:

    results = model.process("The morphine was dissolved in ethyl acetate.", 0.0001, False)
    for r in results: print(r)

gives:

    [4, 12, 'morphine', 0.96779683232307434, True]
    [4, 16, 'morphine was', 0.00048376933409599587, False]
    [4, 26, 'morphine was dissolved', 0.00052144089568173513, False]
    [4, 43, 'morphine was dissolved in ethyl acetate', 0.00011464102863101289, False]
    [5, 12, 'orphine', 0.00051424378762021661, False]
    [13, 16, 'was', 0.00011857352365041152, True]
    [17, 26, 'dissolved', 0.00032986183578032069, True]
    [27, 29, 'in', 0.00010183551421505399, True]
    [27, 43, 'in ethyl acetate', 0.00019362384045962244, False]
    [30, 35, 'ethyl', 0.00015816287850611843, False]
    [30, 43, 'ethyl acetate', 0.97885456681251526, True]
    [30, 44, 'ethyl acetate.', 0.00028939036201336421, False]
    [31, 43, 'thyl acetate', 0.00019865675130859017, False]
    [36, 43, 'acetate', 0.00022335542234941386, False]

The second argument to `process` is the threshold. The third argument is whether to exclude non-dominant entities or not. For example,
`model.process("The morphine was dissolved in ethyl acetate.", 0.0001, True)` gives:

    [4, 12, 'morphine', 0.96779683232307434, True]
    [13, 16, 'was', 0.00011857352365041152, True]
    [17, 26, 'dissolved', 0.00032986183578032069, True]
    [27, 29, 'in', 0.00010183551421505399, True]
    [30, 43, 'ethyl acetate', 0.97885456681251526, True]

To use the traditional and minimalist models, use `get_trad_model` or `get_mini_model` instead of `get_ensemble_model`.
	
## Training

ChemListem is bundled with model files. However, you may wish to train your own. The training data is available
[here](http://www.becalm.eu/pages/biocreative) - you will need to get the GPRO & CEMP training set 2016, and extract the files
`BioCreative V.5 training set.txt` and `CEMP_BioCreative V.5 training set annot.tsv`. Optionally, you may also wish to obtain the
[GloVe](https://nlp.stanford.edu/projects/glove/) pre-trained word vectors - get `glove.6B.zip`, unzip it and find `glove.6B.300d.txt`.

The traditional model may be trained using the following example:

    from chemlistem import tradmodel
	tm = tradmodel.TradModel()
    tm.train("BioCreative V.5 training set.txt", "CEMP_BioCreative V.5 training set annot.tsv", "D:/glove.6B/glove.6B.300d.txt", "tradtest")

If you do not wish to use GloVe, then instead of `"D:/glove.6B/glove.6B.300d.txt"` write `None`.
	
This process will take several hours to run. It should eventually produce two files: `tradmodel_tradtest.h5` and `tradmodel_tradtest.json`
(also several files of the form `epoch_*_tradtest.h5`). These are your model files. To use them, follow this example:

	from chemlistem import tradmodel
	tm = tradmodel.TradModel()
	tm.load("tradmodel_tradtest.json", "tradmodel_tradtest.h5")
	print(tm.process("This test includes morphine.")
	
Training and loading the minimalist model is similar - however, this takes several days, does not use GloVe, and does not produce a JSON
file. Examples:

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
	
## Differences from BioCreative V.5 entry

To make it possible to distribute ChemListem, a few minor changes have been made from the system that generated the entries
for BioCreative V.5, as described in the paper *Chemlistem - chemical named entity recognition using recurrent neural network*.

1. ChemListem now no longer uses a list of names from ChemSpider - this provided a very slight increase in performance of the
traditional model, but created difficulties for distribution.
2. The BioCreative V.5 used scikit-learn's Random Forest implementation both for generating the random forests, and for
generating predictions using the forests. This implementation still uses scikit-learn to generate the random forests, but
uses custom code to extract the model from scikit-learn's data structures and generate predictions. This to make the
random forest model easier to serialize and deserialize.
3. The list of chemical names extracted from ChEBI varies in two regards. Firstly, it uses a later version of chebi.obo (version 150,
rather than 139). Secondly, the original version extracted those names with InChI strings that could be interepreted as chemicals
using RDKit. The current version does not attempt to intepret InChI strings, instead accepting all names with associated InChI strings.
4. The code has been tidied up, comments have been added, and various other changes have been made (e.g. to code organization,
data flow etc.) that should not affect the results.
5. The model files have been regenerated to reflect the changes above, and also changes in keras from version 1.2.0 (used in the
BioCreative V.5 entry) to version 2.0.3 (the current version at time of writing). Furthermore the model files in this release have
been generated using the TensorFlow backend, whereas the BioCreative V.5 models were generated using Theano. Regenerating the
model files creates some natural variability due to the use of randomness in generating the models.
6. There is no model for post-processing the results of the minimalist system: the post-processing was found not to improve
the results in the final evaluation, and at any rate is outperfomed by the ensemble model, so it is not included.
7. In training, we found that our models gave slightly higher F scores than before. The minimalist model gave an F of 0.8678 (c.f. 0.8664
from the paper) and the traditional model gave an F of 0.8725 (c.f. 0.8707 from the paper).

## BioCreative VI

This repository also contains the source code for our entry in BioCreative VI Task 5, in the subdirectory cl_bc6_chemprot

## Conclusion

ChemListem has been developed by the Data Science group at the Royal Society of Chemistry.

ChemListem is distributed under the MIT License - see License.txt.

There is a paper describing ChemListem - *Chemlistem - chemical named entity recognition using recurrent neural network* - to be
published as part of the BioCreative V.5 proceedings. If you use chemlistem in your work, please cite it.

I would like to thank Adam Bernard for the initial Python translation of the tokeniser that chemlistem uses.

Peter Corbett, 2016-2017
The Royal Society of Chemistry