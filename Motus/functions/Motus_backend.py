# Import Motus files
from .preprocessing import AdjustRawBinData, Flip, Flip_UpsideDown
from .backendfunctions import (
    CycleStepSplit1,
    array,
    chunk_ts,
    ActivityDetectSplit,
    LyingSplit,
    StepSplit2,
    rotate_split,
    ref_angle_auto_thigh_1hz,
    Flipthigh_1hz,
    FlipTrunk_insideout,
    downsample,
)
from .Motus_Core import NotWorn, detect_SF12, BoutFilter

# Import other modules
import numpy as np
import bisect

def find_nearest_timestamp(ts_comb, ts_chunked):
    """
    Find the index of the nearest timestamp in ts_array for each timestamp in ts.
    """
    nearest_indices = []
    for t in ts_comb:
        nearest_index = bisect.bisect_left(ts_chunked, t)
        if nearest_index == 0:
            nearest_indices.append(0)
        elif nearest_index == len(ts_chunked):
            nearest_indices.append(len(ts_chunked) - 1)
        else:
            before = ts_chunked[nearest_index - 1]
            after = ts_chunked[nearest_index]
            if after - t < t - before:
                nearest_indices.append(nearest_index)
            else:
                nearest_indices.append(nearest_index - 1)
    return nearest_indices

def get_ts(ts_list):

    lo = np.min(np.concatenate(ts_list))
    hi = np.max(np.concatenate(ts_list))

    hi += (
        1000 if (lo % 1000) > (hi % 1000) else 0
    )  # Assure that highest ts is included in combined ts
    ts_comb = np.arange(lo, hi, step=1000)
    return ts_comb


def motus_step1(cls, ts_list, data_list, SF=30):
    # Prepare input
    ts = ts_list[0]
    ts = ts.flatten()
    Acc = array(data_list[0])

    # Detect if sampling frequency is 12.5 Hz
    SF12, HardSF = detect_SF12(ts)

    # Adjust data
    Acc, Start, TimeT = AdjustRawBinData(Acc, ts=ts)
    N = len(Acc)
    Tid = TimeT[0] + np.arange(0, (N - 1) / SF) / 86400

    X = Acc[
        :, 0
    ]  # define before rotation to align with offline version (should maybe be after in both)

    # Detect nonweat
    NonWear = NotWorn(Acc, SF, Tid)

    # Downsample to 1 Hz
    Mean, Std, Acc12 = downsample(Acc, SF)

    # Compute values for cycling and step detection
    Iws, Irun, hlratio = CycleStepSplit1(Acc, Mean, Std, SF, X)

    # MKJ: added
    TimeTunix = ((TimeT - 719529.0) * 86400 * 1000).astype(np.int64)
    ts_chunked: np.array = chunk_ts(TimeTunix, SF)

    # Rotation/Calibration pre-step
    AccSum = np.sum(Acc12, axis=0)
    AccSqsum = np.sum(np.square(Acc12), axis=0)

    xsum = AccSum[:, 0]
    zsum = AccSum[:, 2]
    xSqsum = AccSqsum[:, 0]
    zSqsum = AccSqsum[:, 2]

    xzsum = np.sum((Acc12[:, :, 0] * Acc12[:, :, 2]), axis=0)

    # Prepare output arrays
    out_cat = np.zeros([len(ts_chunked), 1], dtype=np.int32)
    out_val = np.zeros([len(ts_chunked), len(cls.output_values)], dtype=np.float32)
    out_ver = np.zeros([len(ts_chunked), len(cls.output_verbose)], dtype=np.float32)

    out_cat[:, 0] = 1

    # 1Hz std values per axis
    out_val[:, 0] = Std[:, 0]
    out_val[:, 1] = Std[:, 1]
    out_val[:, 2] = Std[:, 2]

    # 1Hz mean values per axis
    out_val[:, 3] = Mean[:, 0]
    out_val[:, 4] = Mean[:, 1]
    out_val[:, 5] = Mean[:, 2]

    # 1Hz values for cycle and step computations
    out_val[:, 6] = hlratio
    out_val[:, 7] = Iws
    out_val[:, 8] = Irun

    # 1Hz nonwear booleans
    out_val[:, 9] = NonWear

    # 1Hz values for rotating acc data
    out_val[:, 10] = xsum
    out_val[:, 11] = zsum
    out_val[:, 12] = xSqsum
    out_val[:, 13] = zSqsum
    out_val[:, 14] = xzsum

    # Boolean for whether signal is 12.5 Hz
    out_val[:, 15] = SF12

    return ts_chunked, out_cat, out_val, out_ver


def motus_step2_3sensors(cls, ts_list, data_list, parameters, SF=30):

    ts_comb = get_ts(ts_list)  # If outputs from pre steps are not of equal length

    # Prepare out variables
    out_cat = np.zeros(shape=len(ts_comb), dtype=np.int32)
    # out_val = np.zeros([len(ts_comb), len(cls.output_values)], dtype=np.float32)
    out_val = np.full([len(ts_comb), len(cls.output_values)], fill_value=np.nan)
    out_ver = np.zeros([len(ts_comb), len(cls.output_verbose)], dtype=np.float32)

    FBThigh = np.full(shape=len(ts_comb), fill_value=np.nan)

    if data_list[0] is not None:
        # Thigh process
        out_cat, out_val, FBThigh = thigh_process(
                                            cls,
                                            ts_list,
                                            data_list,
                                            parameters,
                                            out_cat,
                                            out_val,
                                            out_ver,
                                            ts_comb,
                                            FBThigh=FBThigh)

    if data_list[1] is not None:
        # Trunk process
        out_cat, out_val = trunk_process(
                                    cls,
                                    ts_list,
                                    data_list,
                                    parameters,
                                    out_cat,
                                    out_val,
                                    out_ver,
                                    ts_comb,
                                    FBThigh=FBThigh)


    if data_list[2] is not None:
        # Arm process
        out_val = arm_process(cls,
                        ts_list,
                        data_list,
                        parameters,
                        out_cat,
                        out_val,
                        out_ver,
                        ts_comb)

    if data_list[3] is not None:
        # Calf process
        calf_process(cls, ts_list, data_list, parameters)

    out_cat = out_cat.flatten()

    return ts_comb, out_cat, out_val, out_ver

def motus_step2_thighonly(cls, ts_list, data_list, parameters, SF=30):

    ts_comb = ts_list[0]  # If outputs from pre steps are not of equal length

    # Prepare out variables
    out_cat = np.zeros(shape=len(ts_comb), dtype=np.int32)
    # out_val = np.zeros([len(ts_comb), len(cls.output_values)], dtype=np.float32)
    out_val = np.full([len(ts_comb), len(cls.output_values)], fill_value=np.nan)
    out_ver = np.zeros([len(ts_comb), len(cls.output_verbose)], dtype=np.float32)

    FBThigh = np.full(shape=len(ts_comb), fill_value=np.nan)

    if data_list[0] is not None:
        # Thigh process
        out_cat, out_val, FBThigh = thigh_process(
                                            cls,
                                            ts_list,
                                            data_list,
                                            parameters,
                                            out_cat,
                                            out_val,
                                            out_ver,
                                            ts_comb,
                                            FBThigh=FBThigh)

    out_cat = out_cat.flatten()

    return ts_comb, out_cat, out_val, out_ver




def thigh_process(
    cls,
    ts_list,
    data_list,
    parameters,
    out_cat,
    out_val,
    out_ver,
    ts_comb,
    FBThigh
):
    ts_chunked = ts_list[0]
    Datain = data_list[0][:, 1:] * 0.000001

    ts_chunked = ts_chunked[np.argsort(ts_chunked)]
    Datain = Datain[np.argsort(ts_chunked)]

    ts_idx = find_nearest_timestamp(ts_comb, ts_chunked)

    ts_chunked = ts_chunked[ts_idx]
    Datain = Datain[ts_idx]

    startidx = np.argmin(abs(ts_comb - ts_chunked.min()))
    endidx = startidx + len(ts_chunked)

    Stdx = Datain[:, 0]
    Stdy = Datain[:, 1]
    Stdz = Datain[:, 2]
    Std = np.array([Stdx, Stdy, Stdz]).T

    Meanx = Datain[:, 3]
    Meany = Datain[:, 4]
    Meanz = Datain[:, 5]
    Mean = np.array([Meanx, Meany, Meanz]).T

    hlratio = Datain[:, 6]
    Iws = Datain[:, 7]
    Irun = Datain[:, 8]

    NonWear = Datain[:, 9]

    xsum = Datain[:, 10]
    zsum = Datain[:, 11]
    xSqsum = Datain[:, 12]
    zSqsum = Datain[:, 13]
    xzsum = Datain[:, 14]

    SF12 = Datain[0, 15]

    # Flip should go here
    Mean = Flipthigh_1hz(Mean)

    ref_angle = ref_angle_auto_thigh_1hz(Mean, Std, iterative=None)

    Std, Mean = rotate_split(
        ref_angle * (np.pi / 180), xsum, zsum, xSqsum, zSqsum, xzsum, Std, Mean)
    
    if SF12:
        Std = 1.14*Std + 0.02*np.square(Std)
    else:
        Std = 1.03*Std + 0.18*np.square(Std)

    Comb, ThighFB = ActivityDetectSplit(Mean, Std, hlratio)

    # To completely remove short bouts, especially 'move' has not been filtered above
    Comb = BoutFilter(Comb, "row")
    Comb = BoutFilter(Comb, "cycle")
    Comb = BoutFilter(Comb, "stair")
    Comb = BoutFilter(Comb, "run")
    Comb = BoutFilter(Comb, "walk")
    Comb = BoutFilter(Comb, "move")
    Comb = BoutFilter(Comb, "stand")
    Comb = BoutFilter(Comb, "sit")

    Comb = LyingSplit(Mean, Comb)

    Comb = BoutFilter(Comb, "sit")
    Comb = BoutFilter(Comb, "lie")

    Comb[NonWear == 1] = 10

    Fstep = StepSplit2(Comb, Iws, Irun)

    Comb[np.all([Comb == 5, Fstep > 2.5], axis=0)] = 6
    Comb = BoutFilter(Comb, "run")
    Comb = BoutFilter(Comb, "walk")

    out_cat[startidx:endidx] = Comb

    out_val[startidx:endidx, 0] = Fstep

    FBThigh[startidx:endidx] = ThighFB

    return out_cat, out_val, FBThigh, Mean


def trunk_process(cls,
    ts_list,
    data_list,
    parameters,
    out_cat,
    out_val,
    out_ver,
    ts_comb,
    FBThigh):

    #initialize variables from Datain
    ts_chunked = ts_list[1]
    Datain = data_list[1][:, 1:] * 0.000001

    ts_chunked = ts_chunked[np.argsort(ts_chunked)]
    Datain = Datain[np.argsort(ts_chunked)]
    
    startidx = np.argmin(abs(ts_chunked.min()-ts_comb))
    
    ts_idx = (
        find_nearest_timestamp(ts_comb[startidx:], ts_chunked) 
    )

    ts_chunked = ts_chunked[ts_idx]
    Datain = Datain[ts_idx]
    
    endidx = startidx + len(ts_chunked)

    out_cat_copy = np.copy(out_cat)
    out_cat_copy = out_cat_copy[startidx:endidx]

    Stdx = Datain[:, 0]
    Stdy = Datain[:, 1]
    Stdz = Datain[:, 2]
    Std = np.array([Stdx, Stdy, Stdz]).T

    Meanx = Datain[:, 3]
    Meany = Datain[:, 4]
    Meanz = Datain[:, 5]
    Mean = np.array([Meanx, Meany, Meanz]).T

    NonWear = Datain[:, 9]

    # xsum = Datain[:, 10]
    # zsum = Datain[:, 11]
    # xSqsum = Datain[:, 12]
    # zSqsum = Datain[:, 13]
    # xzsum = Datain[:, 14]

    #Flip of trunk. Flip should be extracted from reference position. 
    Mean = array(Mean)

    # #access trunk reference position in parameters 

    # TrunkRefAcc = parameters['TrunkRefAcc']
    # Flip_upsidedown = Flip_UpsideDown(Mean)
    # Flip_insideout = FlipTrunk_insideout(TrunkRefAcc)

    # # Flip accelerometer data based on orientation flags
    # if Flip_insideout and (not Flip_upsidedown):
    #     Mean[:, -3:] = Mean[:, -3:] * np.array([1, -1, -1])

    # elif Flip_insideout and Flip_upsidedown:
    #     Mean[:, -3:] = Mean[:, -3:] * np.array([-1, 1, -1])

    # elif (not Flip_insideout) and Flip_upsidedown:
    #     Mean[:, -3:] = Mean[:, -3:] * np.array([-1, -1, 1])

    
    # Calculate Euclidean length of the modified accelerometer data
    Lng = np.sqrt(array(np.square(Mean[:, 0]) + np.square(Mean[:, 1]) + np.square(Mean[:, 2])))

    # Compute angles from the modified accelerometer data
    Inc = np.arccos(array(np.divide(Mean[:, 0], Lng)))
    FB = -np.arcsin(array(np.divide(Mean[:, 2], Lng)))
    Lat = -np.arcsin(array(np.divide(Mean[:, 1], Lng)))

    angles = np.column_stack((Inc, FB, Lat))
    
    #Reference angle. Iteratively? Check if Trunk ref exists in parameters?
    if parameters['TrunkRef'] is None: #probably an index instead
        if sum(out_cat)>0:
            v2 = np.median(FB[out_cat_copy == 5]) - (np.pi * 6 / 180)
            v3 = np.median(Lat[out_cat_copy == 5])
            Trunkref = np.array([np.arccos(v2) * np.cos(v3), v2, v3])
        else:
            Trunkref = np.pi * np.array([27, 27, 0]) / 180
    else:
        Trunkref = parameters["TrunkRef"]

    
    # Create rotation matrix for trunk data
    Rot1 = np.array(
        [
            [np.cos(Trunkref[1]), 0, np.sin(Trunkref[1])],
            [0, 1, 0],
            [-np.sin(Trunkref[1]), 0, np.cos(Trunkref[1])],
        ]
    )
    Rot2 = np.array(
        [
            [np.cos(Trunkref[2]), np.sin(Trunkref[2]), 0],
            [-np.sin(Trunkref[2]), np.cos(Trunkref[2]), 0],
            [0, 0, 1],
        ]
    )
    Rot = np.matmul(Rot1, Rot2)

    # Rotate the trunk data
    MeanRot = np.matmul(Mean, Rot)
    
    # Compute rotated angles
    VTrunkRot = np.column_stack(
        (
            np.real(np.arccos(array(np.divide(MeanRot[:, 0], Lng)))),
            np.real(-np.arcsin(array(np.divide(MeanRot[:, 2], Lng)))),
            np.real(-np.arcsin(array(np.divide(MeanRot[:, 1], Lng)))),
        )
    )

    TrunkInc = VTrunkRot[:,0] #Trunk Inclination
    TrunkFB = VTrunkRot[:,1] #Trunk FB
    TrunkLat = VTrunkRot[:,2] #Trunk Lateral

    TrunkInc[NonWear == 1] = np.nan
    TrunkFB[NonWear == 1] = np.nan
    TrunkLat[NonWear == 1] = np.nan

    out_val[startidx:endidx, 1] = TrunkInc * (180/np.pi)
    out_val[startidx:endidx, 2] = TrunkFB * (180/np.pi)
    out_val[startidx:endidx, 3] = TrunkLat * (180/np.pi)


    if sum(out_cat)>0: # check if thigh_process has run

        Akt = out_cat_copy
        IncTrunk = out_val[startidx:endidx, 1]
        FBTrunk = out_val[startidx:endidx, 2]
        LatTrunk = out_val[startidx:endidx, 3]
        FBThigh = FBThigh[startidx:endidx]

        TrunkLie = IncTrunk > 65

        Akt[Akt == 1] = 2  # Turn every lie back to sit


        Akt[np.all([Akt == 2, TrunkLie], axis=0)] = 1  # Let only trunk angle determine lying

        Ibackwards45 = np.any(
            [FBTrunk < -45,
                abs(LatTrunk) > 45], axis=0,)

        Akt[np.all([Akt == 2, Ibackwards45 == 1, ~np.isnan(IncTrunk)], axis=0)] = 1
        
        Akt[np.all([Akt == 1,
                    ~np.isnan(IncTrunk),
                    FBTrunk > 0,
                    FBThigh > 45,], 
                    axis=0,)] = 2
        Akt[np.all([Akt == 1,
                    ~np.isnan(IncTrunk),
                    abs(IncTrunk - FBTrunk) < 10,
                    IncTrunk < 65,
                    FBTrunk < 65,
                ],
                axis=0,
            )
        ] = 2
        
        Akt = BoutFilter(Akt, "sit")
        Akt = BoutFilter(Akt, "lie")

        out_cat[startidx:endidx] = Akt

    return out_cat, out_val


def arm_process(cls,
    ts_list,
    data_list,
    parameters,
    out_cat,
    out_val,
    out_ver,
    ts_comb):

    ts_chunked = ts_list[2]
    Datain = data_list[2][:, 1:] * 0.000001

    ts_chunked = ts_chunked[np.argsort(ts_chunked)]
    Datain = Datain[np.argsort(ts_chunked)]
    
    startidx = np.argmin(abs(ts_chunked.min()-ts_comb))
    
    ts_idx = (
        find_nearest_timestamp(ts_comb[startidx:], ts_chunked) 
    )

    ts_chunked = ts_chunked[ts_idx]
    Datain = Datain[ts_idx]
    
    endidx = startidx + len(ts_chunked)

    Stdx = Datain[:, 0]
    Stdy = Datain[:, 1]
    Stdz = Datain[:, 2]
    Std = np.array([Stdx, Stdy, Stdz]).T

    Meanx = Datain[:, 3]
    Meany = Datain[:, 4]
    Meanz = Datain[:, 5]
    Mean = np.array([Meanx, Meany, Meanz]).T

    NonWear = Datain[:, 9]

    # xsum = Datain[:, 10]
    # zsum = Datain[:, 11]
    # xSqsum = Datain[:, 12]
    # zSqsum = Datain[:, 13]
    # xzsum = Datain[:, 14]

    FlipUD_arm = Flip_UpsideDown(Mean)
    if FlipUD_arm:
        Mean[:, -3:] = Mean[:, -3:] * np.array([-1, -1, 1])

    #Reference angle for ARM???

    Lng = np.sqrt(np.square(Mean[:, 0]) + np.square(Mean[:, 1]) + np.square(Mean[:, 2]))

    ArmInc = np.arccos(np.divide(Mean[:, 0], Lng))
    # FB = -np.arcsin(np.divide(Mean[:, 2], Lng))
    # Lat = -np.arcsin(np.divide(Mean[:, 1], Lng))

    # angles = np.column_stack((Inc, FB, Lat))

    ArmInc[NonWear == 1] = np.nan

    out_val[startidx:endidx, 4] = ArmInc * (180/np.pi)

    return out_val
    





def Compute_Exposures(cat, val):
    """
    Compute the exposures needed for the ErgoConnect project, based on the 1Hz data 
    output from step 2. 


    Parameters
    ----------
    cat : numpy array
        length same as ts.
    val : numpy array
        with columns corresponding to
        ["activity/steps/count", "angle/TrunkInc", 
         "angle/TrunkFB", "angle/TrunkLat", "angle/ArmInc"].

    Returns
    -------
    out : TYPE
        DESCRIPTION.

    """

    import numpy as np
    import pandas as pd
    
    out = np.c_[cat, val] 
    out = pd.DataFrame(out, columns=['Activity', 'TrunkInc', "TrunkFB", "TrunkLat", "ArmInc"])

    out['Sedentary_time'] = np.any([out.Activity == 1, out.Activity == 2], axis=0)*1
    out['Standing_time'] = np.any([out.Activity == 3, out.Activity == 4], axis=0)*1
    out['Walking_time'] = (out.Activity == 5)*1


    #Nrisesit
    DiffAkt = (out["Activity"].isin([1,2]) * 1).to_numpy()
    DiffAkt = np.diff(DiffAkt, prepend=DiffAkt[0])
    out['Rise_from_sedentary_number'] = (DiffAkt == -1)*1

    ThresTrunk = np.array([30, 60, 180])
    ThresArm = np.array([30, 60, 90, 180])

    for idx in range(len(ThresTrunk) - 1):
        out[f'Forward_Bending_{ThresTrunk[idx]}_to_{ThresTrunk[idx+1]}_time'] = np.all(
            [
                out.Activity > 2,
                out.Activity < 8,
                out.TrunkFB > 0,
                out.TrunkInc >= ThresTrunk[idx],
                out.TrunkInc <= ThresTrunk[idx + 1],
            ],
            axis=0,
        )*1
    forward45 = np.all(
            [
                out.Activity > 2,
                out.Activity < 8,
                out.TrunkFB > 0,
                out.TrunkInc >= 45,
                out.TrunkInc <= 180,
            ],
            axis=0,
        )

    # out['forward45'] = forward45
    Diff45 = np.diff(forward45, prepend=0)
    # out['diff45'] = Diff45
    # Start = (np.array(np.nonzero(Diff45 == 1)))[0]
    # Slut = (np.array(np.nonzero(Diff45 == -1)) - 1)[0]
    # SSdur = Slut - Start + 1

    # crossing 45 from below 
    #up45 = (np.array(angles[1:]) > 45) & (np.array(angles[:-1]) <= 45)
    # up45 = np.insert(up45, 0, 0)
    #crossing 30 from above
    #down30 = (np.array(angles[1:]) < 30) & (np.array(angles[:-1]) >= 30)
    #down30 = np.insert(down30, 0, 0)

    out['Forward_Bending_45_to_180_number'] = (Diff45 == 1)*1

    for idx in range(len(ThresArm) - 1):
        out[f'Arm_Lifting_{ThresArm[idx]}_to_{ThresArm[idx+1]}_time'] = np.all(
            [
                out.Activity > 2,
                out.Activity < 6,
                out.ArmInc >= ThresArm[idx],
                out.ArmInc <= ThresArm[idx + 1],
            ],
            axis=0,
        )*1

    return out

