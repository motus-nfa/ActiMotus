# Import modules
from ..preprocessing import read_bin
from .alg_chunk_motus_pre_v1_2_0 import ChunkMotusPre_v1_2_0
from .alg_activity_motus_v1_2_0 import ActivityMotus_v1_2_0
import numpy as np
import os
import pandas as pd


def classify(Time, Acc, baseline=False, issens12=False):
    ts_list = [Time]
    data_list = [Acc]
    # Run pre-step
    # Create an instance of the ChunkMotusPre_v1_2_0 class
    chunk_motus_pre = ChunkMotusPre_v1_2_0

    # Call the analyse_data_list_new method and pass the inputs
    out_ts, out_cat, out_val, out_ver, _ = chunk_motus_pre.analyse_data_list_new(
        ts_list, data_list
    )

    # Combine the categorical and value outputs for step 2
    out_cat_val = np.column_stack((out_cat, out_val))

    # Define the required inputs for step 2
    ts_list_step2 = [out_ts]
    data_list_step2 = [out_cat_val]

    # Create an instance of the ActivityMotus_v1_3_3 class
    activity_motus = ActivityMotus_v1_2_0

    # Call the analyse_data_list_new method and pass the inputs
    out_ts_step2, Akt, Fstep, out_ver_step2, _ = activity_motus.analyse_data_list_new(
        ts_list_step2,
        data_list_step2,
        parameters=None,
        debug_stream=None,
        debug_chunks=None,
    )

    return Akt, Fstep, out_val, out_ts


if __name__ == "__main__":
    accpath = os.path.abspath(
        "U:\DI\MOTUS\Validering\Hovedprojekt\Data\Results\Sens12_pol\Training data"
    )
    resultpath = os.path.abspath(
        "U:\DI\MOTUS\Validering\Hovedprojekt\Data\Results\Sens12_pol\Backend"
    )

    # Load referencevinkler
    file = "U:\DI\MOTUS\Validering\Hovedprojekt\Data\Synced\Reference angles from annotated part.csv"
    ref_angles = pd.read_csv(file, index_col=0)
    ref_angles.columns = ["Ax25", "Sens25", "Sens12"]

    # Load og stack annoteringer
    annodict = {}
    for id in range(1, 22):
        anno_file = os.path.join(accpath, str(id), f"{id}_Annotation_1sec.csv")
        if os.path.isfile(anno_file):
            anno = pd.read_csv(anno_file, header=None)
            annodict[id] = anno

    # Weights from OMMSB wihtout move
    weightpath = "U:\DI\MOTUS\Validering\Hovedprojekt\Data\Vægte"
    weights = pd.read_csv(os.path.join(weightpath, "weights.csv"), index_col=0).values
    weights = [i[0] for i in weights]

    runs = ["baseline", "new_pol"]

    for run in runs:
        cm_diags = []

        if run == "baseline":
            sensors = ["Sens25", "Sens12"]
            baseline = True
        else:
            sensors = ["Sens12"]
            baseline = False

        for sensor in sensors:
            Aktdict = {}
            Fstepdict = {}

            for id in range(1, 22):
                # Read Accelerometer data
                Accfile = os.path.join(accpath, str(id), f"{id}_{sensor}.csv")
                if os.path.isfile(Accfile) == False:
                    # print(f'Skip {id=}')

                    continue
                Acc = pd.read_csv(Accfile)
                cutoff = Acc.shape[0] % 30
                Time = (
                    Acc.iloc[:-cutoff, 0].values
                    if cutoff != 0
                    else Acc.iloc[:, 0].values
                )
                Acc = (
                    Acc.iloc[:-cutoff, 1:].values
                    if cutoff != 0
                    else Acc.iloc[:, 1:].values
                )

                issens12 = "Sens12" in sensor

                # Read reference angles
                VrefThigh = np.pi * np.array([0, ref_angles.loc[id, sensor], 0]) / 180

                Akt, Fstep = classify(
                    Time=Time,
                    Acc=Acc,
                    VrefThigh=VrefThigh,
                    baseline=baseline,
                    issens12=issens12,
                )

                Aktdict[id] = Akt
                Fstepdict[id] = Fstep

            # Gather valid ids based on available Akt and Annotation
            anno_ids = set(annodict.keys())
            akt_ids = set(Aktdict.keys())
            valid_ids = list(anno_ids.intersection(akt_ids))

            # Stack Akt
            stacked_Akt = np.concatenate([Aktdict[id] for id in valid_ids])

            stacked_anno = np.concatenate([annodict[id] for id in valid_ids]).flatten()

            ind7 = np.argwhere(stacked_anno == 7)
            ind8 = np.argwhere(stacked_anno == 8)
            stacked_anno[ind7] = 8
            stacked_anno[ind8] = 7

            # Check that stacked annotation and akt have same lengths
            if not len(stacked_Akt) == len(stacked_anno):
                print(f"{len(stacked_Akt) == len(stacked_anno)=}")

            cr, cm_diag, labels = Eval_Akt.Eval_Akt(
                Akt=stacked_Akt,
                Annotation=stacked_anno,
                parameterset_id=run,
                sensor=sensor,
                savepath=resultpath,
                nomove=False,
                weights=weights,
            )

            cm_diags.append(cm_diag)

            cr = pd.DataFrame(cr)
            cr.to_csv(os.path.join(resultpath, f"{sensor}_{run}_cr.csv"))

        diag_df = pd.DataFrame(cm_diags).T
        diag_df.columns = sensors
        diag_df.index = labels
        diag_df.to_excel(os.path.join(resultpath, f"diagonals {run}.xlsx"))
