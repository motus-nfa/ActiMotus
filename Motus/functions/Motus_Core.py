# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 09:05:52 2024

@author: b061621
"""

from functions.preprocessing import Flip_UpsideDown, AdjustRawBinData, AutoCalibrate

# from functions.Motus_Thigh import ActivityDetect, ref_angle_auto_thigh


def main_multiple(
    AccThigh=None,
    AccArm=None,
    AccTrunk=None,
    AccCalf=None,
    SF=30,
    VrefThigh=None,
    VrefTrunk=None,
    AdjustBin=True,
    Autocalibrate=True,
    SF12=False,
):
    """
    Executes the motus pipeline; from raw data to classified. Accepts Arm and Trunk data as well

    Parameters
    ----------
    AccThigh : array, optional
        Thigh acc data (first column is time). The default is None.
    AccArm : array, optional
        Arm acc data (first column is time). The default is None.
    AccTrunk : array, optional
        Trunk acc data (first column is time). The default is None.
    AccCalf : array, optional
        Calf acc data (first column is time). The default is None.
    SF : int, optional
        System sampling frequncy. The default is 30.
    VrefThigh : array, optional
        Reference angle for thigh. The default is None.
    VrefTrunk : TYPE, optional
        Reference angle for trunk. The default is None.
    AdjustBin : bool, optional
        Whether to adjust raw bin data to 30Hz. The default is True.
    Autocalibrate : bool, optional
        Whether to autocalibrate data. The default is True.
    SF12 : bool, optional
        If the data comes from a 12.5Hz sensor. The default is False.

    Returns
    -------
    Akt : array
        Classified activities encoded as integers.
    Time : array
        time.
    Std : array
        standard deviation.
    Fstep : array
        steps per second.
    TrunkBending : array
        trunk bending based on intervals. Not currently used for anything.
    ArmLifting : array
        arm lifting based on defined intervals. Not currently used for anything.
    VTrunkRot_out : array
        Angle signal of trunk (inc, fb, y-axis angle) after rotation with reference angles.
    Varm_out : array
        Angle signal of arm (inc, fb, y-axis angle).

    """
    import numpy as np

    # After ActivityDetect is called, some additional filtering and classification takes place utilizing StepAnalysis

    Akt, Time, Std, Fstep, TrunkBending, ArmLifting, VTrunkRot_out, Varm_out = (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )

    # Upsample/adjust raw data to G format and autocalibrate
    if AdjustBin:
        if AccThigh is not None:
            AccThigh, Start, TimeThigh = AdjustRawBinData(AccThigh, SF, utc=True)

            if Autocalibrate:
                AccThigh, scale, offset = AutoCalibrate(AccThigh)
        else:
            TimeThigh = None

        if AccArm is not None:
            AccArm, Start, TimeArm = AdjustRawBinData(AccArm, SF, utc=True)

            if Autocalibrate:
                AccArm, scale, offset = AutoCalibrate(AccArm)
        else:
            TimeArm = None

        if AccTrunk is not None:
            AccTrunk, Start, TimeTrunk = AdjustRawBinData(AccTrunk, SF, utc=True)

            if Autocalibrate:
                AccTrunk, scale, offset = AutoCalibrate(AccTrunk)
        else:
            TimeTrunk = None

        if AccCalf is not None:
            AccCalf, Start, TimeCalf = AdjustRawBinData(AccCalf, SF, utc=True)

            if Autocalibrate:
                AccCalf, scale, offset = AutoCalibrate(AccCalf)
        else:
            TimeCalf = None

    # #Synchronize
    (
        TimeCommon,
        AccThigh,
        AccTrunk,
        AccArm,
        AccCalf,
        OffThigh,
        OffTrunk,
        OffArm,
        OffCalf,
    ) = sync_extend(
        AccThigh=AccThigh,
        AccTrunk=AccTrunk,
        AccArm=AccArm,
        AccCalf=AccCalf,
        TimeThigh=TimeThigh,
        TimeTrunk=TimeTrunk,
        TimeArm=TimeArm,
        TimeCalf=TimeCalf,
    )
    # TimeCommon, AccThigh, AccTrunk, AccArm, AccCalf, OffThigh, OffTrunk, OffArm, OffCalf = sync_extend(AccThigh = AccThigh, AccTrunk = AccTrunk, AccArm = AccArm, AccCalf = AccCalf, TimeThigh = TimeThigh, TimeTrunk = TimeTrunk, TimeArm = TimeArm, TimeCalf = TimeCalf)

    if AccThigh is not None:
        AccThigh[OffThigh, :] = 1  # overwriting 99 so std does not go crazy

        # Either the reference angle for thigh is inputted or it is estimated here
        if VrefThigh is None:
            if len(OffThigh) > 0:
                AccThighOn = np.delete(AccThigh, OffThigh, axis=0)
            else:
                AccThighOn = AccThigh
            median_angle_auto = ref_angle_auto_thigh(AccThighOn)
            VrefThigh = (
                np.pi * np.array([0, median_angle_auto, 0]) / 180
            )  # 3d but only the second element is used/only corrected in one direction.
            print(f"{median_angle_auto = }")
        Start = [TimeCommon[0]]

        # Run activity classification
        Akt, Time, FBthigh, Std, Incthigh, STD, Fstep = ActivityDetect(
            AccThigh, SF, Start, VrefThigh, SF12, OffThigh
        )  # activity detection based on thigh

    if AccTrunk is not None:
        # Perform trunk analysis

        # Find nonwear periods in trunk data
        Off_Trunk = NotWorn(AccTrunk, SF, Tid=None)

        # Synchronize off indices
        Off_Trunk[np.fix(Off_Trunk[::30] / 30).astype(int)] = 1

        # Check if the device is flipped upside down
        FlipUD_Trunk = Flip_UpsideDown(AccTrunk)

        # Check if the device is flipped inside out
        FlipIO_Trunk = Flip_InsideOut_Timeless(AccTrunk)

        # Compute trunk angles
        Vtrunk, AccTrunkFilt, Lng = Vinkler(AccTrunk, SF)

        # Estimate reference angles for trunk/back from periods of walking
        if VrefTrunk is None:
            if AccThigh is not None:
                VtrunkAccAP = np.median(
                    np.reshape(Vtrunk[:, 1], (SF, len(Akt)), order="F"), axis=0
                )
                VtrunkAccLat = np.median(
                    np.reshape(Vtrunk[:, 2], (SF, len(Akt)), order="F"), axis=0
                )
                v2 = np.median(
                    VtrunkAccAP[np.all([Akt == 5, Off_Trunk == 0], axis=0)]
                ) - (np.pi * 6 / 180)
                v3 = np.median(VtrunkAccLat[np.all([Akt == 5, Off_Trunk == 0], axis=0)])
                VrefTrunk = np.array([np.arccos(v2) * np.cos(v3), v2, v3])
            else:
                VrefTrunk = np.pi * np.array([27, 27, 0]) / 180

        # Create rotation matrix for trunk data
        Rot1 = np.array(
            [
                [np.cos(VrefTrunk[1]), 0, np.sin(VrefTrunk[1])],
                [0, 1, 0],
                [-np.sin(VrefTrunk[1]), 0, np.cos(VrefTrunk[1])],
            ]
        )
        Rot2 = np.array(
            [
                [np.cos(VrefTrunk[2]), np.sin(VrefTrunk[2]), 0],
                [-np.sin(VrefTrunk[2]), np.cos(VrefTrunk[2]), 0],
                [0, 0, 1],
            ]
        )
        Rot = np.matmul(Rot1, Rot2)

        # Rotate the trunk data
        AccTrunkRot = np.matmul(AccTrunkFilt, Rot)

        # Compute rotated angles (mainly the first is used, which is inclination)
        VTrunkRot = np.column_stack(
            (
                np.real(np.arccos(np.divide(AccTrunkRot[:, 0], Lng))),
                np.real(-np.arcsin(np.divide(AccTrunkRot[:, 2], Lng))),
                np.real(-np.arcsin(np.divide(AccTrunkRot[:, 1], Lng))),
            )
        )

        # Compute trunk bending for various postures
        if AccThigh is not None:
            # Overwrite lying
            Acc12Trunk = Acc60(AccTrunkRot, SF)
            AccMeanTrunk = np.mean(Acc12Trunk, axis=0)
            LngMean = np.sqrt(
                np.square(AccMeanTrunk[:, 0])
                + np.square(AccMeanTrunk[:, 1])
                + np.square(AccMeanTrunk[:, 2])
            )

            Inc_Trunk = (180 / np.pi) * np.arccos(
                np.divide(AccMeanTrunk[:, 0], LngMean)
            )

            TrunkLie = Inc_Trunk > 65

            Akt[Akt == 1] = 2  # Turn every lie back to sit

            Akt[np.all([Akt == 2, TrunkLie], axis=0)] = (
                1  # Let only trunk angle determine lying
            )

            Ibackwards45 = np.any(
                [
                    VTrunkRot[::SF, 1] * (180 / np.pi) < -45,
                    abs(VTrunkRot[::SF, 2]) * (180 / np.pi) > 45,
                ],
                axis=0,
            )

            Akt[np.all([Akt == 2, Ibackwards45 == 1, Off_Trunk == 0], axis=0)] = 1

            Akt[
                np.all(
                    [
                        Akt == 1,
                        Off_Trunk == 0,
                        VTrunkRot[::SF, 1] * (180 / np.pi) > 0,
                        FBthigh > 45,
                    ],
                    axis=0,
                )
            ] = 2
            Akt[
                np.all(
                    [
                        Akt == 1,
                        Off_Trunk == 0,
                        abs(VTrunkRot[::SF, 0] - VTrunkRot[::SF, 1]) * (180 / np.pi)
                        < 10,
                        VTrunkRot[::SF, 0] * (180 / np.pi) < 65,
                        VTrunkRot[::SF, 1] * (180 / np.pi) < 65,
                    ],
                    axis=0,
                )
            ] = 2

            Akt = BoutFilter(Akt, "sit")
            Akt = BoutFilter(Akt, "lie")

            # Forward bending
            IonTrunk = np.logical_not(
                np.reshape(np.tile(Off_Trunk, (SF, 1)), -1, order="F")
            )
            IpositiveU = VTrunkRot[:, 1] > 0
            InotLie = np.logical_not(
                np.reshape(
                    np.tile(np.any([Akt == 0, Akt == 1], axis=0), (SF, 1)),
                    -1,
                    order="F",
                )
            )

            IforwardInc = np.all([IonTrunk, InotLie, IpositiveU], axis=0)

            IsitFwd = np.all(
                [
                    np.reshape(np.tile((Akt == 2) * 1, (SF, 1)), -1, order="F"),
                    IforwardInc,
                ],
                axis=0,
            )
            IstandFwd = np.all(
                [
                    np.reshape(
                        np.tile(np.any([Akt == 3, Akt == 4], axis=0), (SF, 1)),
                        -1,
                        order="F",
                    ),
                    IforwardInc,
                ],
                axis=0,
            )
            IuprightFwd = np.all(
                [
                    np.reshape(
                        np.tile(np.all([Akt >= 3, Akt <= 7], axis=0), (SF, 1)),
                        -1,
                        order="F",
                    ),
                    IforwardInc,
                ],
                axis=0,
            )

            # Loop over thresholds for trunk bending
            ThresTrunk = np.array([20, 30, 60, 90]) * np.pi / 180
            IncTrunk = []
            PctTrunk = []
            PctTrunk_NotLie = []
            ForwardIncTrunk = []
            ForwardIncTrunkSit = []
            ForwardIncTrunkStandMove = []
            ForwardIncTrunkUpright = []

            for count, thrs in enumerate(ThresTrunk):
                # Append time spend with trunk bending over threshold
                IncTrunk.append(sum(VTrunkRot[IonTrunk, 0] >= thrs))
                PctTrunk.append((IncTrunk[count] / len(VTrunkRot[IonTrunk, 0])) * 100)
                ForwardIncTrunk.append(sum(VTrunkRot[IforwardInc, 0] >= thrs) / SF)
                ForwardIncTrunkSit.append(sum(VTrunkRot[IsitFwd, 0] >= thrs) / SF)
                ForwardIncTrunkStandMove.append(
                    sum(VTrunkRot[IstandFwd, 0] >= thrs) / SF
                )
                ForwardIncTrunkUpright.append(
                    sum(VTrunkRot[IuprightFwd, 0] >= thrs) / SF
                )

                PctTrunk_NotLie.append(
                    ((sum(VTrunkRot[InotLie, 0] >= thrs)) / len(VTrunkRot[InotLie, 0]))
                    * 100
                )

            TrunkBending = np.vstack(
                (
                    PctTrunk,
                    PctTrunk_NotLie,
                    ForwardIncTrunk,
                    ForwardIncTrunkSit,
                    ForwardIncTrunkStandMove,
                    ForwardIncTrunkUpright,
                )
            )
            TrunkBending[2:, :] = TrunkBending[2:, :] / 60 / 60

            VTrunkRot_out = VTrunkRot[::SF, :]
            VTrunkRot_out[Off_Trunk == 1, :] = np.nan

    if AccArm is not None:
        # Perform arm analysis

        # Find nonwear periods in arm data
        Off_Arm = NotWorn(AccArm, SF, Tid=None)
        Off_Arm[np.fix(Off_Arm[::30] / 30).astype(int)] = 1

        # Check if the device is flipped upside down
        FlipUD_Arm = Flip_UpsideDown(AccArm)

        # Check if the device is flipped inside out (may not make sense)
        FlipIO_Arm = Flip_InsideOut_Timeless(AccArm)

        print(f"{FlipUD_Arm = } {FlipIO_Arm = }")

        # Compute arm angles
        VArm, AccArmFilt, Lng_arm = Vinkler(AccArm, SF)

        # Isolate different body positions
        Varm = VArm[:, 0]
        IonArm = np.logical_not(np.reshape(np.tile(Off_Arm, (SF, 1)), -1, order="F"))
        InotLie = np.logical_not(
            np.reshape(
                np.tile(np.any([Akt == 0, Akt == 1], axis=0), (SF, 1)), -1, order="F"
            )
        )
        IokArm = np.all([IonArm, InotLie], axis=0)

        Isit = np.all(
            [np.reshape(np.tile((Akt == 2) * 1, (SF, 1)), -1, order="F"), IonArm],
            axis=0,
        )
        Istandmove = np.all(
            [
                np.reshape(
                    np.tile(np.any([Akt == 3, Akt == 4], axis=0), (SF, 1)),
                    -1,
                    order="F",
                ),
                IonArm,
            ],
            axis=0,
        )
        Iupright = np.all(
            [
                np.reshape(
                    np.tile(np.all([Akt >= 3, Akt <= 7], axis=0), (SF, 1)),
                    -1,
                    order="F",
                ),
                IonArm,
            ],
            axis=0,
        )

        # Loop over thresholds
        ThresArm = np.array([30, 60, 90, 120, 150]) * np.pi / 180
        IncArm = []
        PctArm = []
        PctArm_NotLie = []
        IncArmSit = []
        IncArmStandMove = []
        IncArmUpright = []

        for count, thrs in enumerate(ThresArm):
            # Append sum of time over thresholds at different positions
            IncArm.append(sum(Varm[IonArm] >= thrs) / SF)
            PctArm.append((IncArm[count] / (len(Varm[IonArm]) / SF)) * 100)
            IncArmSit.append(sum(Varm[Isit] >= thrs) / SF)
            IncArmStandMove.append(sum(Varm[Istandmove] >= thrs) / SF)
            IncArmUpright.append(sum(Varm[Iupright] >= thrs) / SF)

            PctArm_NotLie.append(
                ((sum(Varm[IokArm] >= thrs)) / (len(Varm[IokArm]))) * 100
            )

        ArmLifting = np.vstack(
            (PctArm, PctArm_NotLie, IncArm, IncArmSit, IncArmStandMove, IncArmUpright)
        )
        ArmLifting[2:, :] = ArmLifting[2:, :] / 60 / 60

        # Arm Speed (angles per second)
        ArmSpeed = np.diff(Varm, prepend=Varm[0]) * SF
        # Either take every SF element or take the average of every second
        NArmS = SF * np.fix(len(ArmSpeed) / SF)
        ArmSpeed_out = np.mean(ArmSpeed[: int(NArmS)].reshape(-1, SF), axis=1)

        Varm_out = Varm[::SF]
        Varm_out[Off_Arm == 1] = np.nan

    if AccCalf is not None:
        # Detect non-wear periods for calf accelerometer
        Off_Calf = NotWorn(AccCalf, SF, Tid=None)
        Off_Calf[np.fix(OffCalf[::30] / 30).astype(int)] = 1

        # Check if the calf accelerometer is flipped upside down or inside out
        FlipUD_Calf = Flip_UpsideDown(AccCalf)
        if FlipUD_Calf:
            AccCalf[:, -3:] = AccCalf[:, -3:] * np.array([-1, -1, 1])
        # FlipIO_Calf = Flip_InsideOut_Timeless(AccCalf)  # Note: The comment suggests this may not make sense, consider reviewing or removing

        MeanCalf, StdCalf, _ = downsample(AccCalf, SF, SF12)
        MeanThigh, StdThigh, _ = downsample(AccThigh, SF, SF12)

        # # Identify periods of non-wear for calf based on a threshold
        # OffCalf = OffCalf | NotWornCalf(Akt, VThigh, AccCalf, SF, calfTh)

        # Detect kneeling
        Akt = KneelSquatDetection(Akt, MeanThigh, MeanCalf)

    return Akt, Time, Std, Fstep, TrunkBending, ArmLifting, VTrunkRot_out, Varm_out


def Acc60(Acc, SF):
    """
    This function creates a tensor in order to compute mean and std of samples for 2 overlapping seconds.
    A built in function (pd.rolling or np.convolve) should be able to replace it.

    Parameters
    ----------
    Acc : numpy array nx3
        Acceleration in 30Hz
    SF : int
        System sampling frequency. Hard wired to 30Hz

    Returns
    -------
    Acc12 : numpy array nx3
        Downsampled acceleration signal

    """
    import numpy as np

    # Rearrange data in Acc matrix into 3-dimensional array

    Acc1 = np.concatenate((Acc[0:SF], Acc), axis=0, dtype=np.float32)  # Pad before
    Acc2 = np.concatenate((Acc, Acc[-SF:]), axis=0, dtype=np.float32)  # Pad after
    Acc12 = np.concatenate(
        (
            np.reshape(Acc1, (SF, -1, 3), order="F"),
            np.reshape(Acc2, (SF, -1, 3), order="F"),
        ),
        axis=0,
        dtype=np.float32,
    )  # Reshape into tensor
    del Acc1, Acc2  # Delete unused matrices
    Acc12 = Acc12[:, 0:-1, :]  # Slice
    return Acc12


def adjust_std(Std, SF12):
    import numpy as np

    # SENS sensor correction (different corrections for 25Hz and 12.5Hz)
    Std = (
        0.18 * np.square(Std) + 1.03 * Std
        if not SF12
        else 0.02 * np.square(Std) + 1.14 * Std
    )
    return Std


def downsample(Acc, SF):
    """
    Takes the 30Hz data down to 1Hz. This function is a collection of code previously present in
    other functions.
    The correctional polynomial on std for Sens and Sens 12.5Hz is applied here

    Parameters
    ----------
    Acc : numpy array Nx3
        Acceleration data.
    SF : int
        System sampling frequency (30Hz).

    Returns
    -------
    Mean : numpy array nx3
        mean of filtered 2 second windows of acceleration.
    Std : numpy array nx3
        standard deviation of filtered 2 second windows of acceleration.
    Acc12 : numpy array nx3
        downsampled acc data.

    """
    import numpy as np
    from scipy import signal

    # Butterworth filtering/ 5 Hz lowpass filtering
    Bl, Al = signal.butter(4, 5 / (SF / 2), "low")  # butterworth filter coefficients
    AccL = signal.lfilter(
        Bl, Al, Acc, axis=0
    )  # filter using coefficients using rational transfer function

    Acc12 = Acc60(AccL, SF)  # reshape Acc into a 3 dimensional array
    del AccL

    Std = np.std(
        Acc12, axis=0, ddof=1
    )  # compute standard deviation of columns. The array is automatically squeezed intop a 2d array

    Mean = np.mean(Acc12, axis=0)  # compute mean of 2 sec intervals

    return Mean, Std, Acc12


def NotWorn(Acc, SF, Tid=None):
    """
    Non-wear detection. For periods of adequate stillness the device is classified as not-worn.

    Parameters
    ----------
    Acc : numpy array
        raw acc.
    SF : int
        sampling frequency 30Hz.
    Tid : numpy array
        time vector.

    Returns
    -------
    OffOut : boolean array
        indices of elements where sensor is off.
    """
    import numpy as np
    from scipy import signal

    # Low pass filter
    Blp, Alp = signal.butter(6, 2 / (SF / 2), "low")
    M = signal.lfilter(Blp, Alp, Acc, axis=0)

    smpls1s = np.arange(0, len(Acc), SF)

    V, M, L = Vinkler(Acc, SF)

    # Sample the Vinkler function at 1 Hz
    V = V[smpls1s, :]

    del M, L

    # Calculate the acceleration at 1 Hz
    Acc12 = Acc60(Acc, SF)
    StdMean = np.mean(np.std(Acc12, axis=0, ddof=1), axis=1)
    StdSum = np.sum(np.std(Acc12, axis=0, ddof=1), axis=1)

    # Mapping of still
    StillAcc = StdMean[:-1] < 0.01
    StillAcc = np.insert(StillAcc, 0, False)
    StillAcc = np.insert(StillAcc, len(StillAcc), False)
    StillAcc = StillAcc * 1

    # Changes in still, sensor is taken on or off
    OffOn = np.diff(StillAcc)
    Off = (np.array(np.nonzero(OffOn == 1)))[0]
    On = (np.array(np.nonzero(OffOn == -1)))[0]
    OffPerioder = On - Off

    del StillAcc

    # Start and end of off periods
    StartOff = Off[OffPerioder > 600]
    SlutOff = On[OffPerioder > 600]

    Ok = np.full(len(StartOff), False)

    # Check for conditions to consider an off period valid
    for i in range(len(StartOff)):
        # Evaluate conditions for off period validity
        # Conditions include checking for a significant increase in standard deviation sum or a long duration (greater than 1.5 hours)
        Ok[i] = (
            max(StdSum[max(StartOff[i] - 15, 1) : max(StartOff[i] - 11, 5)]) > 0.5
            or (SlutOff[i] - StartOff[i]) > 5400
        )

    StartOff = StartOff[Ok]
    SlutOff = SlutOff[Ok]

    # Check for short on periods
    KortOn = ((StartOff[1:] - SlutOff[:-1]) < 60) * 1

    if KortOn.size > 0:
        # If there are short on periods, use np.insert to create boolean masks for filtering SlutOff and StartOff arrays

        # Create a boolean mask for filtering SlutOff:
        # - Negate each element in KortOn (convert 1s to 0s and vice versa)
        # - Insert 1 at the end of the negated array
        SlutOff = SlutOff[[bool(i) for i in np.insert((1 - KortOn), len(KortOn), 1)]]
        # Create a boolean mask for filtering StartOff:
        # - Negate KortOn (convert 1s to 0s and vice versa)
        # - Insert 1 at the beginning of the negated array
        StartOff = StartOff[[bool(i) for i in np.insert((1 - KortOn), 0, 1)]]

    Af = np.zeros(len(V))

    # Mark off periods based on conditions
    for i in range(len(StartOff)):
        Vmean = 180 * np.mean(V[StartOff[i] : SlutOff[i], :]) / np.pi
        # If off period longer than 1.5 hours or mean angles close to 90,90/90,-90, mark as off
        if (
            (SlutOff[i] - StartOff[i] > 5400)
            or np.all(abs(Vmean - np.array([90, 90, 0])) < 5)
            or np.all(abs(Vmean - np.array([90, -90, 0])) < 5)
        ):
            Af[StartOff[i] : SlutOff[i]] = 1

    Af[np.isnan(StdMean)] = 1

    # Call the Night function for nighttime processing if Tid is provided
    if Tid is not None:
        OffOut = Night(Af, Tid)
    else:
        OffOut = Af

    return OffOut


def BoutFilter(Comb, ActType):
    """
    Filter short bouts of activities out
    If a certain type of activity is below the bout threshold, the activity is
    overwritten with the activity on either side.

    Parameters
    ----------
    Comb : numpy array nx1
        encoded activities.
    ActType : string
        specific activity type (eg. walk, sit etc).

    Returns
    -------
    CombNew : numpy array nx1
        short bouts filtered out for given acttype.

    """
    # input: Comb = vector of different activities every second
    #       ActType = string of activity type
    import numpy as np

    # additional filtering of activities, where short bouts of activities are 'erased' if they are shorter than the thresholds below

    # default bout lengths
    bout_row = 15
    bout_stair = 5
    bout_run = 2
    bout_sit = 5
    bout_lie = 5
    bout_walk = 2
    bout_move = 2
    bout_stand = 2
    bout_cycle = 15

    # Set bout based on input
    if ActType == "lie":
        bout = bout_lie
        No = 1
    if ActType == "sit":
        bout = bout_sit
        No = 2
    if ActType == "stand":
        bout = bout_stand
        No = 3
    if ActType == "move":
        bout = bout_move
        No = 4
    if ActType == "walk":
        bout = bout_walk
        No = 5
    if ActType == "walkslow":
        bout = bout_walk
        No = 5.1
    if ActType == "walkfast":
        bout = bout_walk
        No = 5.2
    if ActType == "run":
        bout = bout_run
        No = 6
    if ActType == "stair":
        bout = bout_stair
        No = 7
    if ActType == "cycle":
        bout = bout_cycle
        No = 8
    if ActType == "row":
        bout = bout_row
        No = 9

    CombNew = Comb.copy()

    # Find instances of activity changes in Comb
    Akt = np.zeros(len(Comb))
    Akt[Comb == No] = 1
    DiffAkt = Akt.copy()
    DiffAkt = np.insert(DiffAkt, len(DiffAkt), 0)
    DiffAkt = np.insert(DiffAkt, 0, 0)
    DiffAkt = np.diff(DiffAkt)

    # Start and end indices of bouts of activities and the ones below the given bout length
    Start = (np.array(np.nonzero(DiffAkt == 1)))[0]
    Slut = (np.array(np.nonzero(DiffAkt == -1)) - 1)[0]
    Korte = np.squeeze(np.array(np.nonzero((Slut - Start) < bout)))

    SS = np.column_stack((Start[Korte], Slut[Korte]))

    for i in range(len(SS)):
        if (
            i == 0 and SS[i, 0] == 0
        ):  # special cases for a bout at the very beginning and end of Comb
            CombNew[SS[i, 0] : SS[i, 1] + 1] = Comb[SS[i, 1] + 1]
        elif (i == (len(SS) - 1)) and (SS[i, 1] == (len(Akt) - 1)):
            CombNew[SS[i, 0] : SS[i, 1]] = Comb[SS[i, 0] - 1]
        else:
            # for short bouts, the middle index is found and elements before this index are set to the activity found just before the bout
            # and elements after are set set to the activity seen after the short bout
            Midt = int(np.fix(np.mean(SS[i, :])))
            CombNew[SS[i, 0] : Midt + 1] = Comb[SS[i, 0] - 1]
            CombNew[Midt + 1 : (SS[i, 1] + 1)] = Comb[SS[i, 1] + 1]

    return CombNew


def Night(Af, Tid):
    """
    Subfunction to NotWorn. For nighttime periods, there exist a different criteria for
    notworn as it is expected that the sensor is more still here. The notworn function works without this as well.


    Parameters
    ----------
    Af : array
        indices of off elements.
    Tid : array
        time.

    Returns
    -------
    OffOut : array
        refined indices.

    """
    import numpy as np

    T = np.fmod(
        Tid, 1
    )  # Compute remainder after division by 1, which corresponds to time of day

    # Define nighttime period
    NightStart = 22 / 24
    NightEnd = 8 / 24
    InNight = np.any([T > NightStart, T < NightEnd], axis=0)

    # Initialize OffOut with the original activity data
    OffOut = Af.copy()

    if InNight.size > 0:  # Check if some nighttime periods exist
        # Calculate differences to find start and end indices of nighttime periods
        Idiff = np.insert(InNight, 0, False)
        Idiff = np.insert(Idiff, len(Idiff), False) * 1
        Idiff = np.diff(Idiff)
        InNightS = (np.array(np.nonzero(Idiff == 1)))[0]
        InNightE = (np.array(np.nonzero(Idiff == -1)) - 1)[0]

        lenNight = np.zeros(len(InNightS))

        # Iterate over identified nighttime periods
        for i in range(len(lenNight)):
            # Calculate length of the nighttime period
            lenNight[i] = InNightE[i] - InNightS[i] + 1

            # If less than half of the period is active, set the activity to 0
            if sum(Af[InNightS[i] : InNightE[i]] == 1) / lenNight[i] < 0.5:
                OffOut[InNightS[i] : InNightE[i]] = 0

    return OffOut


def synchronize_bytime(Acc1, Acc2, Time1, Time2):
    """
    Synchronize two acceleration signals by padding, such that the earliest timestamp starts the common time
    and the latest ends it.

    Parameters
    ----------
    Acc1 : TYPE
        DESCRIPTION.
    Acc2 : TYPE
        DESCRIPTION.
    Time1 : TYPE
        DESCRIPTION.
    Time2 : TYPE
        DESCRIPTION.

    Returns
    -------
    Acc1 : TYPE
        DESCRIPTION.
    Acc2 : TYPE
        DESCRIPTION.
    TimeCommon : TYPE
        DESCRIPTION.
    Off1 : TYPE
        DESCRIPTION.
    Off2 : TYPE
        DESCRIPTION.

    """
    import numpy as np

    # Determine the common time interval and identify overlapping regions
    start = min(Time1[0], Time2[0])
    end = max(Time1[-1], Time2[-1])

    Off1 = np.array([])
    Off2 = np.array([])

    # Check for non-overlapping time intervals
    if (Time1[0] > Time2[-1]) or (Time1[-1] < Time2[0]):
        print("No overlap")
        return

    fillvalue = 99

    # Identify common time interval and adjust accelerometer data accordingly
    if Time1[0] == start:
        TimeCommon = Time1
        start2 = np.argwhere(Time1 >= Time2[0])[0][0]
        Acc2 = np.concatenate((np.full((start2, 3), fillvalue), Acc2), axis=0)
        Off2 = np.arange(0, start2)

        if Time1[-1] == end:
            end2 = np.argwhere(Time2[-1] <= Time1)[0][0]
            Acc2 = np.concatenate(
                (Acc2, np.full((len(Acc1) - len(Acc2), 3), fillvalue)), axis=0
            )
            Off2 = np.append(Off2, np.arange(end2, len(Acc1)))

        elif Time2[-1] == end:
            end2 = np.argwhere(Time1[-1] <= Time2)[0][0]
            TimeCommon = np.append(TimeCommon, Time2[end2:])
            Acc1 = np.concatenate(
                (Acc1, np.full((len(Acc2) - len(Acc1), 3), fillvalue)), axis=0
            )
            Off1 = np.arange(end2, len(Acc1))

    elif Time2[0] == start:
        TimeCommon = Time2
        start2 = np.argwhere(Time2 >= Time1[0])[0][0]
        Acc1 = np.concatenate((np.full((start2, 3), fillvalue), Acc1), axis=0)
        Off1 = np.arange(0, start2)

        if Time2[-1] == end:
            end2 = np.argwhere(Time1[-1] <= Time2)[0][0]
            Acc1 = np.concatenate(
                (Acc1, np.full((len(Acc2) - len(Acc1), 3), fillvalue)), axis=0
            )
            Off1 = np.append(Off1, np.arange(end2, len(Acc2) - 1))

        elif Time1[-1] == end:
            end2 = np.argwhere(Time2[-1] <= Time1)[0][0]
            TimeCommon = np.append(TimeCommon, Time1[end2:])
            Acc2 = np.concatenate(
                (Acc2, np.full((len(Acc1) - len(Acc2), 3), fillvalue)), axis=0
            )
            Off2 = np.arange(end2, len(Acc2))

    else:
        print("This should not happen")

    return Acc1, Acc2, TimeCommon, Off1, Off2


def sync_extend(
    AccThigh=None,
    AccTrunk=None,
    AccArm=None,
    AccCalf=None,
    TimeThigh=None,
    TimeTrunk=None,
    TimeArm=None,
    TimeCalf=None,
    SF=30,
):
    """
    Synchronize up to four sensors

    Parameters
    ----------
    AccThigh : TYPE, optional
        DESCRIPTION. The default is None.
    AccTrunk : TYPE, optional
        DESCRIPTION. The default is None.
    AccArm : TYPE, optional
        DESCRIPTION. The default is None.
    AccCalf : TYPE, optional
        DESCRIPTION. The default is None.
    TimeThigh : TYPE, optional
        DESCRIPTION. The default is None.
    TimeTrunk : TYPE, optional
        DESCRIPTION. The default is None.
    TimeArm : TYPE, optional
        DESCRIPTION. The default is None.
    TimeCalf : TYPE, optional
        DESCRIPTION. The default is None.
    SF : TYPE, optional
        DESCRIPTION. The default is 30.

    Returns
    -------
    TimeCommon : TYPE
        DESCRIPTION.
    AccThigh : TYPE
        DESCRIPTION.
    AccTrunk : TYPE
        DESCRIPTION.
    AccArm : TYPE
        DESCRIPTION.
    AccCalf : TYPE
        DESCRIPTION.
    OffThigh : TYPE
        DESCRIPTION.
    OffTrunk : TYPE
        DESCRIPTION.
    OffArm : TYPE
        DESCRIPTION.
    OffCalf : TYPE
        DESCRIPTION.

    """
    import numpy as np

    OffThigh = np.array([])
    OffTrunk = np.array([])
    OffArm = np.array([])
    OffCalf = np.array([])
    TimeCommon = np.array([])

    if AccThigh is not None:
        ThighOn = 1
        TimeCommon = TimeThigh
        if AccTrunk is not None:
            TrunkOn = 1
            AccThigh, AccTrunk, TimeCommon, OffThigh, OffTrunk = synchronize_bytime(
                AccThigh, AccTrunk, TimeCommon, TimeTrunk
            )
            TimeTrunk2 = TimeCommon
            if AccArm is not None:
                ArmOn = 1
                AccThigh, AccArm, TimeCommon, OffThigh, OffArm = synchronize_bytime(
                    AccThigh, AccArm, TimeCommon, TimeArm
                )
                TimeArm2 = TimeCommon

                AccThigh, AccTrunk, TimeCommon, OffThigh, OffTrunk = synchronize_bytime(
                    AccThigh, AccTrunk, TimeCommon, TimeTrunk2
                )
                TimeTrunk3 = TimeCommon

                if AccCalf is not None:
                    CalfOn = 1
                    (
                        AccThigh,
                        AccCalf,
                        TimeCommon,
                        OffThigh,
                        OffCalf,
                    ) = synchronize_bytime(AccThigh, AccCalf, TimeCommon, TimeCalf)

                    (
                        AccThigh,
                        AccTrunk,
                        TimeCommon,
                        OffThigh,
                        OffTrunk,
                    ) = synchronize_bytime(AccThigh, AccTrunk, TimeCommon, TimeTrunk3)
                    AccThigh, AccArm, TimeCommon, OffThigh, OffArm = synchronize_bytime(
                        AccThigh, AccArm, TimeCommon, TimeArm2
                    )
                else:
                    CalfOn = 0

            else:
                ArmOn = 0
                if AccCalf is not None:
                    CalfOn = 1
                    (
                        AccThigh,
                        AccCalf,
                        TimeCommon,
                        OffThigh,
                        OffCalf,
                    ) = synchronize_bytime(AccThigh, AccCalf, TimeCommon, TimeCalf)

                    (
                        AccThigh,
                        AccTrunk,
                        TimeCommon,
                        OffThigh,
                        OffTrunk,
                    ) = synchronize_bytime(AccThigh, AccTrunk, TimeCommon, TimeTrunk2)
                else:
                    CalfOn = 0

        else:
            TrunkOn = 0
            if AccArm is not None:
                ArmOn = 1
                AccThigh, AccArm, TimeCommon, OffThigh, OffArm = synchronize_bytime(
                    AccThigh, AccArm, TimeCommon, TimeArm
                )
                TimeArm2 = TimeCommon
                if AccCalf is not None:
                    CalfOn = 1
                    (
                        AccThigh,
                        AccCalf,
                        TimeCommon,
                        OffThigh,
                        OffCalf,
                    ) = synchronize_bytime(AccThigh, AccCalf, TimeCommon, TimeCalf)

                    AccThigh, AccArm, TimeCommon, OffThigh, OffArm = synchronize_bytime(
                        AccThigh, AccArm, TimeCommon, TimeArm2
                    )
                else:
                    CalfOn = 0

            else:
                ArmOn = 0
                if AccCalf is not None:
                    CalfOn = 1
                    (
                        AccThigh,
                        AccCalf,
                        TimeCommon,
                        OffThigh,
                        OffCalf,
                    ) = synchronize_bytime(AccThigh, AccCalf, TimeCommon, TimeCalf)
                else:
                    CalfOn = 0
    else:
        # acc thigh off
        ThighOn = 0
        if AccTrunk is not None:
            TrunkOn = 1
            TimeCommon = TimeTrunk
            if AccArm is not None:
                ArmOn = 1
                AccTrunk, AccArm, TimeCommon, OffTrunk, OffArm = synchronize_bytime(
                    AccTrunk, AccArm, TimeCommon, TimeArm
                )
                TimeArm2 = TimeCommon
                if AccCalf is not None:
                    CalfOn = 1
                    (
                        AccTrunk,
                        AccCalf,
                        TimeCommon,
                        OffThigh,
                        OffCalf,
                    ) = synchronize_bytime(AccTrunk, AccCalf, TimeCommon, TimeCalf)

                    AccThigh, AccArm, TimeCommon, OffThigh, OffArm = synchronize_bytime(
                        AccThigh, AccArm, TimeCommon, TimeArm2
                    )
                else:
                    CalfOn = 0

            else:
                ArmOn = 0
                if AccCalf is not None:
                    CalfOn = 1
                    (
                        AccTrunk,
                        AccCalf,
                        TimeCommon,
                        OffThigh,
                        OffCalf,
                    ) = synchronize_bytime(AccTrunk, AccCalf, TimeCommon, TimeCalf)
                else:
                    CalfOn = 0

        else:
            TrunkOn = 0
            if AccArm is not None:
                ArmOn = 1
                TimeCommon = TimeArm
                if AccCalf is not None:
                    CalfOn = 1
                    AccArm, AccCalf, TimeCommon, OffThigh, OffCalf = synchronize_bytime(
                        AccArm, AccCalf, TimeCommon, TimeCalf
                    )
                else:
                    CalfOn = 0

            else:
                ArmOn = 0
                if AccCalf is not None:
                    CalfOn = 1
                    TimeCommon = TimeCalf
                    # AccThigh, AccCalf, TimeCommon, OffThigh, OffCalf = synchronize_bytime(AccThigh, AccCalf, TimeCommon, TimeCalf)
                else:
                    CalfOn = 0

    # print([ThighOn, TrunkOn, ArmOn, CalfOn])

    N = SF * np.fix(
        len(TimeCommon) / SF
    )  # make sure that number of samples corresponds to integer number of seconds,

    TimeCommon = TimeCommon[: int(N)]

    if ThighOn:
        AccThigh = AccThigh[: int(N), :]
        OffThigh = np.argwhere(AccThigh[:, 0] == 99).flatten()
    if ArmOn:
        AccArm = AccArm[: int(N), :]
        OffArm = np.argwhere(AccArm[:, 0] == 99).flatten()
    if TrunkOn:
        AccTrunk = AccTrunk[: int(N), :]
        OffTrunk = np.argwhere(AccTrunk[:, 0] == 99).flatten()
    if CalfOn:
        AccCalf = AccCalf[: int(N), :]
        OffCalf = np.argwhere(AccCalf[:, 0] == 99).flatten()

    return (
        TimeCommon,
        AccThigh,
        AccTrunk,
        AccArm,
        AccCalf,
        OffThigh,
        OffTrunk,
        OffArm,
        OffCalf,
    )


def Vinkler(Acc, SF=30):
    """
    Compute angles of acceleration signal.
    Inclination: 0-180 degrees, based on x-axis
    Forward backward/U: -90 to 90 degrees, based on z-axis (denoted U)
    Sideways/V: sideways tilt, based on y-axis

    Parameters
    ----------
    Acc : array
        raw acceleration.
    SF : int, optional
        sampling frequency of signal. The default is 30.

    Returns
    -------
    angles : nx3
        angle signals.
    M : array nx3
        filtered acc .
    Lng : array
        magnitude of acc.

    """
    import numpy as np
    from scipy import signal

    # Apply low-pass Butterworth filter to the accelerometer data
    Bl, Al = signal.butter(6, 2 / (SF / 2), "low")
    M = signal.lfilter(Bl, Al, Acc, axis=0)

    # Calculate Euclidean length of the modified accelerometer data
    Lng = np.sqrt(np.square(M[:, 0]) + np.square(M[:, 1]) + np.square(M[:, 2]))

    # Compute angles from the modified accelerometer data
    Inc = np.arccos(np.divide(M[:, 0], Lng))
    U = -np.arcsin(np.divide(M[:, 2], Lng))
    V = -np.arcsin(np.divide(M[:, 1], Lng))

    angles = np.column_stack((Inc, U, V))

    return angles, M, Lng


def detect_SF12(ts):
    """
    Detect the sampling frequency of the input time series.

    Parameters
    ----------
    ts : numpy.ndarray
        Input time series.

    Returns
    -------
    is_SF12 : bool
        True if the sampling frequency is 12.5 Hz, False if it's 25 Hz.
    detected_freq : float
        Detected frequency of the input time series.
    """
    import numpy as np

    # Calculate the frequency based on the middle 1000 data points
    freq = 1000 / np.mean(np.diff(ts[int(len(ts) / 2) : int(len(ts) / 2) + 1000]))

    # Check if the detected frequency is close to 12.5 Hz
    is_SF12 = np.isclose(freq, 12.5, atol=2)

    # Check if the detected frequency is close to 25 Hz
    if is_SF12:
        return True, freq
    elif np.isclose(freq, 25, atol=2):
        return False, freq
    else:
        # Print a warning if the detected frequency is neither 12.5 Hz nor 25 Hz
        print("Hardware frequency seems to be neither 12.5 Hz nor 25 Hz")
        print(f"{freq=}")
        return False, freq


def ref_angle_from_ref_position(Acc):
    """
    Compute reference angle based on 20 second standing still with arms down the side. Use this function for all placements (arm, thigh, trunk, calf).


    Parameters
    ----------
    Acc : np array nx3 or nx4
        Length of acc corresponds to 20 seconds (more or less) of standing still in the reference position. So trim acc before passinbg to function.

    Returns
    -------
    Ref angle

    """
    import numpy as np

    # Check if Acc has 4 columns and remove the first column if true
    if Acc.shape[1] == 4:
        Acc = Acc[:, 1:]

    # Calculate angles and length from acceleration
    angles, M, Lng = Vinkler(Acc)
    Inc = angles[:, 0]
    FB = angles[:, 1]

    # Calculate the reference angle as the median of the forward/backward angles converted to degrees
    ref_angle = np.median(FB) * 180 / np.pi

    # Return the reference angle
    return ref_angle
