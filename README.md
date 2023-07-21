# RockML

The RockML library is designed to aid in developing, testing, and deploying
machine-learning models for subsurface characterization. The library is written
for Python 3.6 and includes two main namespaces: data preprocessing and learning.
The former provides adapters for common seismic data formats, i.e., SEGY,
well-logs, and horizon geometries. It also provides many transformations and
visualizations for the data. The learning module includes callbacks, data loaders,
metrics, and state-of-the-art models for post-stack segmentation and VA estimation.

## Installation

To install RockML you need our `seisfast` package also installed. Both Seisfast and RockML are developed and tested
using Python 3.6 Pypi or Anaconda, x86 and Power. In this tutorial, we are going to use a conda environment. The first
thing you have to do is to create and activate the new environment.

For intel:

``` shell
conda create -n rockml python=3.6 numpy -y

. activate rockml
```

For Power8 or Power9:

``` shell
conda create -n rockml python=3.6 numpy pydot scikit-image pandas==0.25.3 -y
. activate rockml

conda config --prepend channels https://public.dhe.ibm.com/ibmdl/export/pub/software/server/ibm-ai/conda/
conda install tensorflow-gpu=2.1.0 -y
```

With the new environment activated, you can install `seisfast` by running the following commands:

``` shell
git clone git@github.ibm.com:brl-nrs-ai-subsurface-libraries/fast-seismic-library.git

pip install fast-seismic-library/.
```

Finally, you can install `rockml` with the following commands:

``` shell
git clone git@github.ibm.com:brl-nrs-ai-subsurface-libraries/rockml.git

pip install rockml/.
```

## Testing your installation

If you reach this point, you're probably all set. However, to make sure that everything is working, you can run:

``` shell
cd rockml

pytest tests
```

# Authors

- Daniel Salles Civitarese - sallesd@br.ibm.com
- Daniela Szwarcman - daniela.szw@ibm.com