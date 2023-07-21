""" Copyright 2019 IBM Research. All Rights Reserved.

    - Pipeline class definition.
"""

import multiprocessing as mp
from functools import partial
from typing import List
from warnings import warn

import numpy as np
from rockml.data.transformations import Composer


class Pipeline(object):

    def __init__(self, composer: Composer):
        """ Initialize Pipeline class

        Args:
            composer: Composer object.
        """
        self.composer = composer

    @staticmethod
    def _apply_composer(chunks: List[List[int]], data_adapter, composer: Composer) -> list:
        """ Generate tiles from a list of seismic slices.

        Args:
            chunks: list of seismic slices.
            data_adapter: initialized adapter object.
            composer: Composer object.

        Returns:
            list containing transformed seisfast.
        """

        loaded_data = data_adapter[chunks]
        tiles = composer.apply(loaded_data)

        return tiles

    # TODO: should we define a data_adapter generic type?
    def build_dataset(self, data_adapter, num_blocks: int, cores: int) -> list:
        """  Build the dataset by parallelizing the task in a pool of workers.

        Args:
            data_adapter: initialized adapter object.
            num_blocks: (int) number of blocks to divide the seismic lines into chunks.
            cores: (int) number of cores to use.

        Returns:
            list containing processed seisfast.
        """

        divisions = np.linspace(0, len(data_adapter), num=num_blocks + 1, dtype=np.int32)

        if len(np.unique(divisions)) < len(divisions):
            divisions = np.unique(divisions)
            warn("num_blocks is bigger than data_adapter total size!")

        chunks = [list(range(divisions[i], divisions[i + 1])) for i in range(len(divisions) - 1)]

        worker = partial(self._apply_composer, data_adapter=data_adapter, composer=self.composer)

        with mp.Pool(cores) as pool:
            processed_data = pool.map(worker, chunks)

        # Concatenating results
        processed_data = [r for sublist in processed_data for r in sublist]

        return processed_data
