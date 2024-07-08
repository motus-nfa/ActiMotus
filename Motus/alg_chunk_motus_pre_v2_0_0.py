import scipy
import numpy as np
from functions.Motus_backend import motus_step1


# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 Det National Forskningscenter for Arbejdsmiljø
Dette prograh er skabt på baggrund af Acti4/Motus algoritmen, og udarbejdet til brug i sammenhæng med alg_activity_motus_v1_2_0.

Version: 2.0.0
"""

# from sensapp.math.algorithm.algorithm_manager import AlgorithmBase

fft = scipy.fft


class ChunkMotusPre_v2_0_0:
    """
    Preprocessing step skeleton for Motus. Downsamples from 30Hz (n) to 1Hz (n2)

    Input is:
    - ts_list: Contains sensor timestamps.   numpy.array [n, 1] @ int64 (For motus w. sens sensor it will be a timestamp for every ~22Hz)
    - data_list: Contains sensor xyz values. numpy.array [n, 3] @ int32 (22Hz xyz samples)

    Output is:
    - out_ts:  Contains timestamps of processed data.  numpy.array [n2, 1] @ int64 (For motus it will be for every 1s)
    - out_cat: Contains processed categories.          numpy.array [n2, 1] @ int64 (For a pre-process step it is either 0=nodata, 1=data)
    - out_val: Contains processed values.              numpy.array [n2, 2] @ int64 (Value of each of output_values member. Will probably be more than 2)
    - out_ver: Contains verbose/debugging values.      numpy.array [n2, 0] @ int64 (Empty. Not used here)

    History
    - 2021.02.22 - Initial skeleton from MKJ
    """

    algname = "chunk/motuspre_v2_0_0"
    algtype = "chunk"
    algvers = "2024.02.23"

    place = ["person/any"]
    input = ["acc/3ax/4g"]

    output_categories = ["general/nodata/time", "general/data/time"]

    output_values = [
        "acc/Stdx",
        "acc/Stdy",
        "acc/Stdz",
        "acc/Meanx",
        "acc/Meany",
        "acc/Meanz",
        "acc/hlratio",
        "acc/Iws",
        "acc/Irun",
        "acc/NonWear",
        "acc/xsum",
        "acc/zsum",
        "acc/xSqsum",
        "acc/zSqsum",
        "acc/xzsum",
        "acc/SF12",
    ]

    output_verbose = []

    # THE MAIN FUNCTION
    @classmethod
    def analyse_data_list_new(
        cls, ts_list, data_list, debug_stream=None, debug_chunks=None
    ):

        ts_chunked, out_cat, out_val, out_ver = motus_step1(cls, ts_list, data_list)

        return (
            ts_chunked,
            out_cat,
            (out_val * 1000000).astype(dtype=np.int32),
            (out_ver * 1000000).astype(dtype=np.int32),
        )
