"""
    File name:              backendfunctions.py
    Author:                 Sebastian Sode Hørlück
    Contributors:           Jon Roslyng Larsen, Sebastian Sode Hørlück
    Copyright:              2024 - Det Nationale Forskningscenter for Arbejdsmiljø
    License:                TBD
    Date created:           April 30th 2023
    Date last modified:     January 3rd 2024
    Python version:         3.11

    Description:
        backendfunctions contains functions related specifically to running the backend version of
        Motus. The backend version runs in two steps, such that data is downsampled to 1 Hz in the
        first step and activity detection is run in the second step. Functions in this file are
        altered functions from offline Motus, where offline functions rely on using downsampled
        and raw data at the same time, which is not possible in the backend version. 

    Functions
        array
            changes array types
        chunk_ts
            chunks timestamps based on size (sz)
        CycleStepSplit1
            Function to detect cycling and steps, to be run in the first part of the backend pipeline
        CycleSplit2
            Function to detect cycling in the second step of the backend pipeline
        StepSplit2
            Function to compute steps in the second step of the backend pipeline
        ActivityDetectSplit
            Activity detection to be run in the second step of the backend pipeline
        LyingSplit
            Lying detection to be run in the second step of the backend pipeline
        WalkSlowFast
            Function to seperate slow and fast walking
        rotate_split
            Rotation to be used in second step of backend pipeline
        ref_angle_auto_thigh_1hz
            Function to compute reference angle based on 1Hz data.
         
        
"""

import numpy as np

def array(*args, **kwargs):
    """
    Create a NumPy array with default dtype set to np.float32.

    Parameters
    ----------
    *args : array_like
        Positional arguments passed to np.array.
    **kwargs : dict
        Keyword arguments passed to np.array.

    Returns
    -------
    np.ndarray
        NumPy array with the specified arguments and default dtype.
    """
    kwargs.setdefault("dtype", np.float32)
    return np.array(*args, **kwargs)


def chunk_ts(ts, sz):
    """
    Chunk a time series into equal-sized segments.

    Parameters
    ----------
    ts : array_like
        Time series data.
    sz : int
        Size of each chunk.

    Returns
    -------
    np.ndarray
        NumPy array containing chunks of the time series.
    """
    skip_count = len(ts) % sz
    return ts[0 : len(ts) - skip_count : sz]


def CycleStepSplit1(Acc, Mean, Std, SF, X):
    """
    Perform cycle and step detection with pre-processing steps.

    Parameters
    ----------
    Acc : np.ndarray
        Acceleration data.
    Mean : np.ndarray
        Mean values.
    Std : np.ndarray
        Standard deviation values.
    SF : int
        Sampling frequency.
    X : np.ndarray
        Input data.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Indices of step frequency, indices of run frequency, and HL ratio.
    """
    import numpy as np
    import scipy
    from scipy import signal

    fft = scipy.fft

    # High-pass filtering
    BhC, AhC = signal.butter(3, 1 / (SF / 2), "high")
    AccH = signal.filtfilt(BhC, AhC, Acc[:, 2], axis=0)

    # Low-pass filtering
    BlC, AlC = signal.butter(3, 1 / (SF / 2), "low")
    AccL = signal.filtfilt(BlC, AlC, Acc[:, 2], axis=0)

    # Trinanalyse + cycle classification pre-step
    hlratio = np.zeros(len(Mean))

    # Low-pass filtering (2.5 Hz)
    Bc, Ac = signal.butter(6, 2.5 / (SF / 2), "low")
    Xc = signal.lfilter(Bc, Ac, X, axis=0)

    # High-pass filtering (1.5-2.5 Hz) for walking
    # Bw, Aw = signal.butter(6, 1.3 / (SF / 2), 'high')
    Bw, Aw = signal.butter(6, 1.5 / (SF / 2), "high")  # original parameter

    Xw = signal.lfilter(Bw, Aw, Xc, axis=0)

    N = len(X)
    del Xc, X, Acc

    # Extra high-pass filtering (3 Hz) for running
    Br, Ar = signal.butter(6, 3 / (SF / 2), "high")
    Xr = signal.lfilter(Br, Ar, Xw)

    Iws = np.zeros(len(Mean))
    Irun = np.zeros(len(Mean))

    # Activity detection
    Some_Activity = 0.09 <= Std[:, 0]
    Some_Activity = np.convolve(Some_Activity, [1, 1, 1], mode="same")

    for i in range(len(Mean)):
        # Check if there is some activity (acceleration standard deviation is above a threshold)
        if Some_Activity[i]:
            # Extract a window of data around the current second
            ii = np.arange(max(0, (i + 1) * SF - 64), min((i + 1) * SF + 64, N))

            # Calculate the High-Low (HL) ratio for the window
            HLratio = np.mean(abs(AccH[ii])) / np.mean(abs(AccL[ii]))
            hlratio[i] = HLratio

            # Detrend and Fourier transform the windowed data for walking
            xws = signal.detrend(
                Xw[ii]
            )  # Detrend the high-pass filtered data for walking
            Yws = fft.fft(xws, 512)  # Apply Fourier transform to the detrended data
            Aws = 2 * abs(Yws[:256])  # Extract the amplitude spectrum
            Iws[i] = np.argmax(
                Aws
            )  # Store the index of the maximum amplitude frequency, representing the step frequency for walking

            # If standard deviation indicates running, detrend and Fourier transform the windowed data for running
            if Std[i, 0] > 0.60:
                xrun = signal.detrend(
                    Xr[ii]
                )  # Detrend the extra high-pass filtered data for running
                Yrun = fft.fft(
                    xrun, 512
                )  # Apply Fourier transform to the detrended data for running
                Arun = 2 * abs(Yrun[:256])  # Extract the amplitude spectrum for running
                Irun[i] = np.argmax(
                    Arun
                )  # Store the index of the maximum amplitude frequency, representing the step frequency for running

    del AccL, AccH, Xw, Xr, Some_Activity

    return Iws, Irun, hlratio


def CycleSplit2(MaybeCycle, SCth, FB, HLratio):
    """
    Split potential cycles based on conditions.

    Parameters
    ----------
    MaybeCycle : np.ndarray
        Array indicating potential cycles.
    SCth : float
        Threshold for cycle splitting.
    FB : np.ndarray
        Forward/backward angles.
    HLratio : np.ndarray
        HL ratio values.

    Returns
    -------
    np.ndarray
        Binary array indicating cycles.
    """
    import numpy as np
    from scipy import signal

    Cycle = np.zeros(len(MaybeCycle))
    MaybeCycle = signal.medfilt(MaybeCycle, 9)

    for i in range(len(MaybeCycle)):
        # Check conditions for cycle
        if MaybeCycle[i]:
            if (HLratio[i] < 0.5) or (SCth < FB[i]):
                Cycle[i] = 1

    return Cycle


def StepSplit2(Akt, Iws, Irun, SF=30):
    """
    Compute steps based on data from pre step.

    Parameters
    ----------
    Akt : np.ndarray
        Activity array.
    SF : int
        Sampling frequency.
    Iws : np.ndarray
        Indices related to walking activity.
    Irun : np.ndarray
        Indices related to running activity.

    Returns
    -------
    np.ndarray
        Frequency of steps.
    """
    import numpy as np
    from scipy import signal

    # pre-allocate
    Fstep = np.zeros(len(Akt))
    Walk = np.zeros(len(Akt))
    Run = np.zeros(len(Akt))
    Stairs = np.zeros(len(Akt))

    Walk[Akt == 5] = 1
    Run[Akt == 6] = 1
    Stairs[Akt == 7] = 1
    Alle = Walk + Run + Stairs

    f = SF / 2 * np.linspace(0, 1, 256)  # frequency scale

    posIws = Iws[Iws > 0]

    # Loop over all seconds
    for i in range(len(Akt)):
        # If walk, run or stairs
        if Alle[i] == 1:
            # Step frequency computed in step 1
            I = Iws[i]
            # If no frequency available, use mean of all positive frequencies
            if I == 0:
                # I = np.random.choice(posIws)
                I = np.mean(posIws)

            # If run find frequencies from run
            if Run[i] == 1:
                I = Irun[i]
                if I == 0:
                    I = np.mean(Irun[Irun > 0])
            Fstep[i] = f[int(I)]

    # Filter frequencies
    Fstep = signal.medfilt(Fstep, 3)

    return Fstep


def ActivityDetectSplit(Mean, Std, hlratio):
    """
    Core function for activity detection using thigh-worn accelerometry.
    The function is translated from matlab based Acti4 (developed by Jørgen Skotte/NFA originally).
    From raw acceleration input, 9 activities are classified based on a simple decision tree using
    angles and standard deviation of sliding windows of two seconds (with one second overlap).

    Finally short classified periods are filtered out, if they are shorter than a defined boutlength.

    Parameters
    ----------
    Mean : numpy.ndarray
        Mean acceleration.
    Std : numpy.ndarray
        Standard deviation of the acceleration.
    hlratio : numpy.ndarray
        Leg angle ratios.

    Returns
    -------
    Comb : numpy.ndarray
        Combined activity labels.
    """
    from scipy import signal

    # Threshold values and bout lengths
    Ath = 0.1  # acceleration threshold for stand/move
    WRth = 0.72  # walk/run
    STth = 45  # Sit/stand angle threshold
    SCth = 40  # stair/cycle angle threshold

    bout_row = 15
    bout_stair = 5
    bout_run = 2
    bout_sit = 5
    # bout_lie = 5
    bout_walk = 2
    # bout_move = 2
    bout_stand = 2
    bout_cycle = 15

    Lng = np.sqrt(np.square(Mean[:, 0]) + np.square(Mean[:, 1]) + np.square(Mean[:, 2]))
    Inc = (180 / np.pi) * np.arccos(
        np.divide(Mean[:, 0], Lng)
    )  # inclination of leg  (0-180 degrees)
    FB = -(180 / np.pi) * np.arcsin(
        np.divide(Mean[:, 2], Lng)
    )  # forward/backward angle of leg
    STD = np.max(Std, axis=1)

    Th = 4
    Vstair = Th + np.median(
        FB[np.all([0.25 < Std[:, 0], Std[:, 0] < WRth, FB < 25], axis=0)]
    )  # walk/stair threshold (degrees) (FB<10 used before rotation was included)

    # initialize activities
    Row = np.zeros(len(Inc))
    Cycle = np.zeros(len(Inc))
    Stair = np.zeros(len(Inc))
    Run = np.zeros(len(Inc))
    Walk = np.zeros(len(Inc))
    Sit = np.zeros(len(Inc))
    Stand = np.zeros(len(Inc))

    # Row
    Row[np.all([90 < Inc, Ath < Std[:, 0]], axis=0)] = (
        1  # elements within the thresholds
    )
    Row = signal.medfilt(Row, 2 * bout_row - 1)  # median filter with bout length
    Row = signal.medfilt(Row, 2 * bout_row - 1)  # twice?
    Etter = Row

    # Cycle
    # Cycle[np.all([SCth<FB, Inc<90, Ath<Std[:,0]], axis=0)] = 1
    MaybeCycle = np.zeros(len(Cycle))
    MaybeCycle[np.all([(SCth - 15) < FB, Inc < 90, Ath < Std[:, 0]], axis=0)] = 1
    Cycle = CycleSplit2(MaybeCycle, SCth, FB, hlratio)
    Cycle = signal.medfilt(Cycle, 2 * bout_cycle - 1)  # median filter with bout length
    Cycle = signal.medfilt(Cycle, 2 * bout_cycle - 1)  # twice?

    Cycle = Cycle * (1 - Etter)
    Etter = Cycle + Etter

    # Stair
    Stair[
        np.all(
            [Vstair < FB, FB < SCth, Ath < Std[:, 0], Std[:, 0] < WRth, Inc < STth],
            axis=0,
        )
    ] = 1  # elements within the thresholds
    Stair = signal.medfilt(Stair, 2 * bout_stair - 1)  # median filter with bout length
    Stair = signal.medfilt(Stair, 2 * bout_stair - 1)  # twice?
    Stair = Stair * (1 - Etter)
    Etter = Stair + Etter

    # run
    Run[np.all([Std[:, 0] > WRth, Inc < STth], axis=0)] = (
        1  # elements within the thresholds
    )
    Run = signal.medfilt(Run, 2 * bout_run - 1)  # median filter with bout length
    Run = signal.medfilt(Run, 2 * bout_run - 1)  # twice?
    Run = Run * (1 - Etter)
    Etter = Run + Etter

    # walk
    Walk[
        np.all([Ath < Std[:, 0], Std[:, 0] < WRth, FB < Vstair, Inc < STth], axis=0)
    ] = 1  # elements within the thresholds
    Walk = signal.medfilt(Walk, 2 * bout_walk - 1)  # median filter with bout length
    Walk = signal.medfilt(Walk, 2 * bout_walk - 1)  # twice?
    Walk = Walk * (1 - Etter)
    Etter = Walk + Etter

    # Stand
    Stand[np.all([Inc < STth, STD < Ath], axis=0)] = 1
    Stand = signal.medfilt(Stand, 2 * bout_stand - 1)  # median filter with bout length
    Stand = signal.medfilt(Stand, 2 * bout_stand - 1)  # twice?
    Stand = Stand * (1 - Etter)
    Etter = Stand + Etter

    # Sit
    Sit[np.all([Inc > STth], axis=0)] = 1
    Sit = signal.medfilt(Sit, 2 * bout_sit - 1)  # median filter with bout length
    Sit = signal.medfilt(Sit, 2 * bout_sit - 1)  # twice?
    Sit = Sit * (1 - Etter)
    Etter = Sit + Etter

    # move is defined by what is not classified into the other categories
    Move = 1 - Etter
    Comb = (
        2 * Sit
        + 3 * Stand
        + 4 * Move
        + 5 * Walk
        + 6 * Run
        + 7 * Stair
        + 8 * Cycle
        + 9 * Row
    )

    return Comb, FB


def LyingSplit(meanAcc, Akt):
    """
    Detect lying in second step of backend pipeline.
    Function uses mean accelerometer data.
    Based on a combination of thigh angle and rotations across thresholds, lying is detected.

    Parameters
    ----------
    meanAcc : numpy.ndarray
        Mean acceleration.
    Akt : numpy.ndarray
        Activity labels.

    Returns
    -------
    Akt : numpy.ndarray
        Updated activity labels.
    """

    import numpy as np

    # Thresholds and parameters
    thrshld_anglH = 65
    thrshld_anglL = 64
    noise_margin = 0.05
    minLieTime = 1

    # Calculate thigh angle from mean accelerometer data
    thigh_angle = abs(
        (180 / np.pi)
        * np.arcsin(
            meanAcc[:, 1] / np.sqrt(np.square(meanAcc[:, 1]) + np.square(meanAcc[:, 2]))
        )
    )

    thigh_angle = np.insert(thigh_angle, 0, thigh_angle[0])

    # Find points where thigh angle crosses thresholds
    rotCrossPtsH = np.diff((thigh_angle > thrshld_anglH) * 1) > 0
    rotCrossPtsL = np.diff((thigh_angle < thrshld_anglL) * 1) > 0

    # Apply noise index to exclude small variations
    noiseIndex = abs(np.diff(thigh_angle)) >= noise_margin
    rotCrossPtsH = np.all([rotCrossPtsH, noiseIndex], axis=0)
    rotCrossPtsL = np.all([rotCrossPtsL, noiseIndex], axis=0)

    # Identify sitting points
    sitPts = (Akt == 2) * 1

    # Label sitting sections
    SitSections = np.zeros(len(sitPts))
    label = 0
    numSitS = 0
    for i in range(len(sitPts)):
        if sitPts[i] == 1:
            if sitPts[i - 1] == 0 or i == 0:
                label += 1
                numSitS += 1
            SitSections[i] = label

    # Refine activity labels based on thigh angle and sitting sections
    for section in range(1, numSitS + 1):
        slctPts = np.nonzero(SitSections == section)[0]

        if len(slctPts) > minLieTime:
            ptsH = np.nonzero(rotCrossPtsH[slctPts])[0]
            ptsL = np.nonzero(rotCrossPtsL[slctPts])[0]

            if len(ptsH) >= 1 and len(ptsL) >= 1:
                Akt[slctPts] = 1

    return Akt


def WalkSlowFast(Comb, Fstep, SFWth):
    """
    Calculate the duration of slow and fast/moderate walking based on step frequency.

    Parameters
    ----------
    Comb : numpy.ndarray
        Combined activity labels.
    SFWth : float
        Step frequency threshold.

    Returns
    -------
    WalkSlow : float
        Duration of slow walking in hours.
    WalkFast : float
        Duration of fast/moderate walking in hours.
    """
    WalkSlow = (
        sum(np.all([Fstep < SFWth / 60, Comb == 5], axis=0)) / 3600
    )  # duration of slow walking in hours
    WalkFast = (
        sum(np.all([Fstep > SFWth / 60, Comb == 5], axis=0)) / 3600
    )  # duration of fast/moderate walking in hours

    return WalkSlow, WalkFast


def rotate_split(ref_angle, xsum, zsum, xSqsum, zSqsum, xzsum, Std, Mean, SF=30):
    """
    Rotate the mean and standard deviation in step 2 of the backend pipeline.

    Parameters
    ----------
    ref_angle : float
        Reference angle.
    xsum : float
        Sum of x values.
    zsum : float
        Sum of z values.
    xSqsum : float
        Sum of squared x values.
    zSqsum : float
        Sum of squared z values.
    xzsum : float
        Sum of product of x and z values.
    Std : numpy.ndarray
        Standard deviation values.
    Mean : numpy.ndarray
        Mean values.
    SF : float
        Sampling frequency.

    Returns
    -------
    StdT : numpy.ndarray
        Transformed standard deviation.
    MeanT : numpy.ndarray
        Transformed mean.
    """
    # Rotation matrix
    Rot = np.array(
        [
            [np.cos(ref_angle), 0, np.sin(ref_angle)],
            [0, 1, 0],
            [-np.sin(ref_angle), 0, np.cos(ref_angle)],
        ]
    )
    MeanT = np.matmul(Mean, Rot)

    theta = ref_angle

    # Calculate terms for x-axis
    sumtermsx = (
        (np.sin(theta) ** 2) * zSqsum
        + (np.cos(theta) ** 2) * xSqsum
        + 2 * SF * np.square(MeanT[:, 0])
        + 2 * np.sin(theta) * MeanT[:, 0] * zsum
        - 2 * np.sin(theta) * np.cos(theta) * xzsum
        - 2 * np.cos(theta) * MeanT[:, 0] * xsum
    )
    # Ensure non-negativity
    sumtermsx = np.array([x if x > 0 else 0 for x in sumtermsx])
    # Calculate transformed standard deviation for x-axis
    Stdx = np.sqrt(1 / (2 * SF - 1) * sumtermsx)

    # Calculate terms for z-axis
    sumtermsz = (
        (np.sin(theta) ** 2) * xSqsum
        + (np.cos(theta) ** 2) * zSqsum
        + 2 * SF * np.square(MeanT[:, 2])
        + 2 * np.sin(theta) * np.cos(theta) * xzsum
        - 2 * np.sin(theta) * MeanT[:, 2] * xsum
        - 2 * np.cos(theta) * MeanT[:, 2] * zsum
    )
    # Ensure non-negativity
    sumtermsz = np.array([z if z > 0 else 0 for z in sumtermsz])
    # Calculate transformed standard deviation for z-axis
    Stdz = np.sqrt(1 / (2 * SF - 1) * sumtermsz)

    # Extract y-axis standard deviation
    Stdy = Std[:, 1]

    # Combine transformed standard deviations
    StdT = np.array([Stdx, Stdy, Stdz]).T

    return StdT, MeanT


def ref_angle_auto_thigh_1hz(Mean, Std, iterative=None):
    """
    Estimate the reference angle for thigh accelerometer placement.

    The estimation is based on walking data, assuming less variation compared to standing.
    It computes a weighted reference angle if the previous reference angle and support are provided.

    Parameters
    ----------
    Mean : np.ndarray
        Matrix of mean acceleration data.
    Std : np.ndarray
        Matrix of standard deviation of acceleration data.
    iterative : list, optional
        A list containing the previous reference angle and the number of samples it was based on.
        The default is None.

    Returns
    -------
    float
        Estimated reference angle.

    """
    import numpy as np

    # import pandas as pd

    # Calculate the Euclidean length of the mean vectors
    Lng = np.sqrt(np.square(Mean[:, 0]) + np.square(Mean[:, 1]) + np.square(Mean[:, 2]))
    # Calculate the inclination of the leg (0-180 degrees)
    Inc = (180 / np.pi) * np.arccos(np.divide(Mean[:, 0], Lng))
    # Calculate the forward/backward angle of the leg
    FB = -(180 / np.pi) * np.arcsin(np.divide(Mean[:, 2], Lng))

    # Identify walking samples based on specified conditions
    Walk = np.all([0.1 < Std[:, 0], Std[:, 0] < 0.7, FB < 10, Inc < 45], axis=0)
    walk_supp = sum(Walk)

    # Calculate the median forward/backward angle during walking
    medianFB_walk = np.median(FB[Walk]) - 6

    # Initialize the median forward/backward angle
    # medianFB = medianFB_walk

    # correction based on 400 subjects of refangle computed based on raw data vs 1Hz.
    # Good linear association, probably introduced in filtering.
    medianFB = medianFB_walk * 0.725 - 5.569

    if iterative is not None:
        # If iterative information is provided, update the median forward/backward angle
        old_ref = iterative[0]
        old_supp = iterative[1]
        total_supp = walk_supp + old_supp
        # weighted average of ref angles
        medianFB = medianFB * (walk_supp / total_supp) + old_ref * (
            old_supp / total_supp
        )

    # fail safe: if ref angle is beyond the acceptable range (based on gathered data and rounded a bit)
    if (medianFB < -30) or (medianFB > 15):
        medianFB = -16

    return medianFB


def Flipthigh_1hz(Mean):
    """
    Check whether sensor is attached inside out or upside down.
    After checks, the data is flipped and returned.

    Parameters
    ----------
    Mean : numpy array
        Mean acc array 1 hz.

    Returns
    -------
    None.

    """
    # Calculate the Euclidean length of mean acc vectors
    Lng = np.sqrt(np.square(Mean[:, 0]) + np.square(Mean[:, 1]) + np.square(Mean[:, 2]))

    # Calculate inclination angles in degrees
    Inc = (180 / np.pi) * np.arccos(np.divide(Mean[:, 0], Lng))

    # Initialize list to store indices of not valid data points
    not_valid_datapoints = []

    # Define indices for each hour in the data
    hourind = np.arange(0, len(Mean) - (60 * 180), 60 * 60)

    count = 0

    # Loop through each hour to check inclination values
    for i in hourind:
        inctmp = Inc[i : i + (60 * 180)]

        # Check if the 2nd percentile of inclination values is above 45 degrees
        if np.percentile(inctmp, 2) > 45:
            not_valid_datapoints.extend(np.arange(i, i + (60 * 180)))
        count += 1

    # Create modified arrays by removing not valid data points
    AccMod = Mean
    IncMod = Inc
    not_valid_datapoints = np.unique(not_valid_datapoints)

    if len(not_valid_datapoints) > 0:
        AccMod = np.delete(AccMod, not_valid_datapoints, axis=0)
        IncMod = np.delete(IncMod, not_valid_datapoints, axis=0)

    # Extract all observations for the z-axis where the inclination of the thigh is above 45 degrees
    SitLieZ = AccMod[IncMod > 45, 2]

    # Calculate the median value of SitLieZ
    median_value_ISO = np.median(SitLieZ)

    # If the median value is positive, it indicates that the device might be attached flipped
    if median_value_ISO > 0:
        Flip_insideout = 1
    else:
        Flip_insideout = 0

    # Upside down
    # extract all observations for the x-axis where the inclination of the thigh is below 45 degrees.
    StandX = Mean[np.any([Inc < 45, Inc > 135], axis=0), 0]

    median_value_USD = np.median(StandX)

    # if the median value is positive it indicates that the device might be attached flipped (could be troublesome if they spend all night sleeping on their stomach (fixed if Time is added as input))
    if median_value_USD < 0:
        Flip_upsidedown = 1
    else:
        Flip_upsidedown = 0

    Mean_Flipped = Mean

    # Flip accelerometer data based on orientation flags
    if Flip_insideout and (not Flip_upsidedown):
        Mean_Flipped = Mean_Flipped * np.array([1, -1, -1])

    elif Flip_insideout and Flip_upsidedown:
        Mean_Flipped = Mean_Flipped * np.array([-1, 1, -1])

    elif (not Flip_insideout) and Flip_upsidedown:
        Mean_Flipped = Mean_Flipped * np.array([-1, -1, 1])

    return Mean_Flipped


def FlipTrunk_insideout(Acc):
    """"
    Insert acc data from reference position. Check if, when bending forward, that the sensor is not attached insideout
    """
    
    Z_median = np.median(Acc[:,-1])

    #we maybe need to doublecheck if this is correct 
    if Z_median > 0:
        Flip = 1
    else:
        Flip = 0
    

    return Flip

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