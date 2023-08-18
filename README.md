# RockML

The RockML library is designed to aid in developing, testing, and deploying
machine-learning models for subsurface characterization. The library is written
for Python 3.10 and includes two main namespaces: data preprocessing and learning.
The former provides adapters for common seismic data formats, i.e., SEGY,
well-logs, and horizon geometries. It also provides many transformations and
visualizations for the data. The learning module includes callbacks, data loaders,
metrics, and state-of-the-art models for post-stack segmentation and VA estimation.

## API structure

Our API is divided in three main groups: data, transformations, and estimators.

![datum.png](docs%2Fimgs%2Fdatum.png)

![transformation.png](docs%2Fimgs%2Ftransformation.png)

![estimator.png](docs%2Fimgs%2Festimator.png)

This is one example of a workflow using RockML for horizon picking:

![overview.png](docs%2Fimgs%2Foverview.png)

## Installation

RockML was developed and tested using Python 3.6 Pypi or Anaconda, x86, and Power, but we tested on Python 3.10.
In this tutorial, we are going to use a conda environment. The first
thing you have to do is to create and activate the new environment.

For intel:

``` shell
conda create -n rockml python=3.10 numpy -y

. activate rockml
```

You can install `rockml` with the following commands:

``` shell
git clone git@github.com:IBM/rockml.git

pip install rockml/.
```

## Testing your installation

If you reach this point, you're probably all set. However, to make sure that everything is working, you can run:

``` shell
cd rockml

pytest tests
```

## Authors

- Daniel Salles Civitarese - sallesd@br.ibm.com
- Daniela Szwarcman - daniela.szw1@ibm.com
- Rodrigo Ferreira

## Citation

Please, consider citing or work if you use RockML. Our first related publication is this one, and you can use it to refer to RockML:

D. S. Chevitarese, D. Szwarcman, E. V. Brazil and B. Zadrozny, "Efficient Classification of Seismic Textures," 2018 International Joint Conference on Neural Networks (IJCNN), Rio de Janeiro, Brazil, 2018, pp. 1-8, doi: 10.1109/IJCNN.2018.8489654.

```
@INPROCEEDINGS{8489654,
  author={Chevitarese, Daniel Salles and Szwarcman, Daniela and Brazil, Emilio Vital and Zadrozny, Bianca},
  booktitle={2018 International Joint Conference on Neural Networks (IJCNN)}, 
  title={Efficient Classification of Seismic Textures}, 
  year={2018},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/IJCNN.2018.8489654}}
```

### The datasets used in our research are Netherlands F3 and Penobscot. You can find the already processed seismic here:

1. Silva, Reinaldo Mozart, et al. "Netherlands dataset: A new public dataset for machine learning in seismic interpretation." arXiv preprint arXiv:1904.00770 (2019).

> ```
> @misc{silva2019netherlands,
>      title={Netherlands Dataset: A New Public Dataset for Machine Learning in Seismic Interpretation}, 
>      author={Reinaldo Mozart Silva and Lais Baroni and Rodrigo S. Ferreira and Daniel Civitarese and Daniela Szwarcman and Emilio Vital Brazil},
>      year={2019},
>      eprint={1904.00770},
>      archivePrefix={arXiv},
>      primaryClass={cs.LG}
>}
>```
>
>**Dataset on Zenodo**: Baroni, Lais, Silva, Reinaldo Mozart, S. Ferreira, Rodrigo, Chevitarese, Daniel, Szwarcman, Daniela, & Vital Brazil, Emilio. (2018). Netherlands F3 Interpretation >Dataset (2.0.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.1471548

2. Baroni, Lais, et al. "Penobscot dataset: Fostering machine learning development for seismic interpretation." arXiv preprint arXiv:1903.12060 (2019).

> ```
> @misc{baroni2019penobscot,
>      title={Penobscot Dataset: Fostering Machine Learning Development for Seismic Interpretation}, 
>      author={Lais Baroni and Reinaldo Mozart Silva and Rodrigo S. Ferreira and Daniel Civitarese and Daniela Szwarcman and Emilio Vital Brazil},
>      year={2019},
>      eprint={1903.12060},
>      archivePrefix={arXiv},
>      primaryClass={physics.geo-ph}
>}
>```
>
>**Dataset on Zenodo**: Baroni, Lais, Silva, Reinaldo Mozart, S. Ferreira, Rodrigo, Chevitarese, Daniel, Szwarcman, Daniela, & Vital Brazil, Emilio. (2020). Penobscot Interpretation >Dataset (3.0.0) [Data set]. https://doi.org/10.5281/zenodo.3924682

### Other related publications that have used RockML

1. Chevitarese, Daniel Salles, et al. "Deep learning applied to seismic facies classification: A methodology for training." Saint Petersburg 2018. Vol. 2018. No. 1. European Association of Geoscientists & Engineers, 2018.
1. Chevitarese, Daniel, et al. "Seismic facies segmentation using deep learning." AAPG Annual and Exhibition (2018).
1. A. B. Mattos et al., "Enabling Robust Horizon Picking From Small Training Sets," in IEEE Transactions on Geoscience and Remote Sensing, vol. 59, no. 6, pp. 5317-5324, June 2021, doi: 10.1109/TGRS.2020.3010124.
1. Souza, Renan, et al. "Provenance data in the machine learning lifecycle in computational science and engineering." 2019 IEEE/ACM Workflows in Support of Large-Scale Science (WORKS). IEEE, 2019.
1. Civitarese, Daniel, et al. "Semantic segmentation of seismic images." arXiv preprint arXiv:1905.04307 (2019).
1. Souza, Renan, et al. "Workflow provenance in the lifecycle of scientific machine learning." Concurrency and Computation: Practice and Experience 34.14 (2022): e6544.
1. Chevitarese, Daniel, et al. "Transfer learning applied to seismic images classification." AAPG Annual and Exhibition (2018).
1. Zadrozny, Bianca, et al. "Estimate ore content based on spatial geological data through 3d convolutional neural networks." U.S. Patent Application No. 16/122,859.
1. Carvalho, BW WSR, et al. "Ore content estimation based on spatial geological data through 3D convolutional neural networks." 81st EAGE Conference and Exhibition 2019 Workshop Programme. Vol. 2019. No. 1. European Association of Geoscientists & Engineers, 2019.
1. Civitarese, Daniel, Daniela Szwarcman, and E. Vital Brazil. "Stratigraphic Segmentation Using Convolutional Neural Networks." 81st EAGE Conference and Exhibition 2019 Workshop Programme. Vol. 2019. No. 1. European Association of Geoscientists & Engineers, 2019.
1. Souza, Renan Francisco Santos, et al. "Managing data traceability in the data lifecycle for deep learning applied to seismic data." AAPG Annual Convention and Exhibition. 2019.
