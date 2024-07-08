import numpy
import numpy as np
from functions.Motus_backend import motus_step2_thighonly


# -*- coding: utf-8 -*-
"""
Copyright (C) 2023 Det National Forskningscenter for Arbejdsmiljø
Dette prograh er skabt på baggrund af Acti4/Motus algoritmen, og udarbejdet til brug i sammenhæng med alg_chunk_motus_pre_v1_2_0.

Version: 1.2.0
"""


class ActivityMotus_v1_2_0:
    """
    Acticity/Step classification skeleton for Motus. (Runs on 1s values (n))

    Input is:
    - ts_list: Contains chunk timestamps.      numpy.array [n, 1] @ int64 (For motus it will be a ts for every 1 second)
    - data_list: Contains preprocessed values. numpy.array [n, 2] @ int32 (Output from 'chunk/motuspre', will probably be more than 2)

    Output is:
    - out_ts:  Contains timestamps of processed data.  numpy.array [n, 1] @ int64 (For motus it will be for every 1s)
    - out_cat: Contains processed categories.          numpy.array [n, 1] @ int64 (For a pre-process step it is either 0=nodata, 1=data)
    - out_val: Contains processed values.              numpy.array [n, 2] @ int64 (Value of each of output_values member)
    - out_ver: Contains verbose/debugging values.      numpy.array [n, 0] @ int64 (Empty. Not used here)

    History
    - 2021.02.22 - Initial skeleton from MKJ
    """

    algname = "person/motus_v1_2_0"
    algtype = "categorizer"
    algvers = "2024.05.16"

    place = [
        "person/thigh",
        "person/trunk",
        "person/arm",
        "person/calf",
    ]
    input = ["chunk/motuspre_v2_0_0"]

    output_categories = [
        "general/nodata/time",
        "activity/lying/time",
        "activity/sitting/time",
        "activity/upright_stand/time",
        "activity/upright_move/time",
        "activity/upright_walk/time",
        "activity/upright_run/time",
        "activity/upright_stair/time",
        "activity/cycling/time",
        "activity/row/time",
        "activity/non_wear/time",
    ]
    output_values = ["activity/steps/count"]


    output_verbose = []

    #  THE MAIN FUNCTION
    @classmethod
    def analyse_data_list_new(
        cls, ts_list, data_list, parameters, debug_stream=None, debug_chunks=None
    ):

        ts_chunked, out_cat, out_val, out_ver = motus_step2_thighonly(
            cls, ts_list, data_list, parameters
        )

        # t.end()

        return (
            ts_chunked,
            out_cat.astype(numpy.int32),
            (out_val * 1000).astype(numpy.int32),
            (out_ver * 1000).astype(numpy.int32),
            None,
        )
