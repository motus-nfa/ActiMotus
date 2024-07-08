"""
    File name:              preprocessing.py
    Author:                 Jon Roslyng Larsen
    Contributors:           Jon Roslyng Larsen, Sebastian Sode Hørlück
    Copyright:              2024 - Det Nationale Forskningscenter for Arbejdsmiljø
    License:                TBD
    Date created:           Primo 2021
    Date last modified:     January 3rd 2024
    Python version:         3.11

    Description:
        Preprocessing contains functions that prepares data for classification in Motus.
        Functions are related to reading raw bin-data, and forming it into arrays.
        Further it upsamples data to 30 Hz, such that no matter the actual sampling frequency
        the output of preprocessing is in 30 Hz.

    Functions
        read_bin
            Reads raw bin data and outputs arrays
        AdjustRawBinData
            Adjusts output from read_bin and upsamples to 30 Hz
        Rotate 
            Rotates signal based on reference angle
        Flip_InsideOut
            Detects whether thigh sensor is worn inside out
        Flip_InsideOut_Timeless
            Same as above but without time as input
        Flip_UpsideDown
            Detects whether thigh sensor is worn upside down
        Flip
            Calls all relevant flip functions and outputs non-flipped data
        AutoCalibrate
            Calibrates sensor inputs (to 1g) through iterative process
        DiaryFun
            Diary function
        find_ref_angle_unix
            Detects reference angle (deprecateds)
"""

# Imports
from functions.backendfunctions import array


def read_bin(name):
    """
    Read .bin file of raw accelerometer data downloaded from web app

    Parameters
    ----------
    name : .bin
        Name/path of file.

    Returns
    -------
    4xn array of values (time, x, y, z)

    """
    import os
    import numpy as numpy
    import struct

    #     n1 = name.split("/")[1]
    #     stream_type = n1.split("_")[1]
    if name is not None:
        var_count, struct_str, bytes_per_sample = (3, "hhh", 6 + 6)
        filesize = os.path.getsize(name)
        samples = int(filesize / bytes_per_sample)

        values = numpy.empty((samples, 1 + var_count), dtype=numpy.int64)
        s = struct.Struct(">q" + struct_str)
        i = 0

        # Read the binary file and unpack the data
        with open(name, "rb") as f:
            bytes = f.read(bytes_per_sample)
            while bytes:
                values[i] = s.unpack(b"\0\0" + bytes)
                i += 1
                bytes = f.read(bytes_per_sample)

        # print(values)
        return values
    else:
        return None


def AdjustRawBinData(Acc, ts=None):
    """
    The downloaded raw data must be mulitplied by -4/500 (after calling read_bin) before being inputted into Motus, in order to have the correct orientation of axes and convert from integer.
    Furthermore it is upsampled to 30Hz.

    Parameters
    ----------
    Acc : TYPE
        DESCRIPTION.
    SF : TYPE
        DESCRIPTION.
    utc : TYPE
        DESCRIPTION.

    Returns
    -------
    Acc : TYPE
        DESCRIPTION.
    Start : TYPE
        DESCRIPTION.
    TimeT : TYPE
        DESCRIPTION.

    """
    import numpy as np

    # Check if the timestamp (ts) is provided, otherwise use the first column of accelerometer data (Acc) as timestamps
    if ts is None:
        ts = Acc[:, 0]
        idx = 1
    else:
        idx = 0

    # Add timexone difference if not utc
    
    Tsens = 719529.0 + ts / 1000 / 86400
    Start = [719529.0 + ts[0] / 1000 / 86400]

    #upsampling frequency
    SF = 30

    # Create a time vector (TimeT) with regular intervals based on the timestamps and sample frequency (SF)
    TimeT = np.arange(Tsens[0], Tsens[-1], 1 / (86400 * SF))

    # Interpolate accelerometer data to match the regular time intervals
    Acc = np.array(
        np.column_stack(
            (
                np.interp(TimeT, Tsens, Acc[:, idx]),
                np.interp(TimeT, Tsens, Acc[:, idx + 1]),
                np.interp(TimeT, Tsens, Acc[:, idx + 2]),
            )
        )
    )

    # Ensure that the number of samples corresponds to an integer number of seconds
    N = SF * np.fix(len(Acc) / SF)
    Acc = Acc[: int(N), :]
    TimeT = TimeT[: int(N)]

    # Remove unnecessary variables
    del ts, Tsens

    # Normalize accelerometer data by scaling it
    Acc = Acc * (-4.0 / 512)

    # Return the adjusted accelerometer data, start timestamp, and time vector
    return Acc, Start, TimeT


def Rotate(Acc, VrefThigh):
    """
    Rotate the acceleration vector 'Acc' based on the rotation angle provided in 'VrefThigh'.

    Parameters
    ----------
    Acc : numpy.ndarray
        3D acceleration vector represented as a NumPy array [Ax, Ay, Az].
    VrefThigh : numpy.ndarray
        Rotation angles. Only one axes [1] is used.

    Returns
    -------
    numpy.ndarray
        Rotated acceleration vector after applying the specified rotation.

    Notes
    -----
    The function uses a 3x3 rotation matrix to rotate the input acceleration vector 'Acc'
    based on the angles specified in 'VrefThigh'. The rotation is applied in the pitch axis.
    The resulting rotated acceleration vector is returned.
    """
    import numpy as np

    # Create a rotation matrix using the provided thigh angle (VrefThigh)
    Rot = np.array(
        [
            [np.cos(VrefThigh[1]), 0, np.sin(VrefThigh[1])],
            [0, 1, 0],
            [-np.sin(VrefThigh[1]), 0, np.cos(VrefThigh[1])],
        ]
    )

    # Rotate the accelerometer data (Acc) using matrix multiplication
    Acc = np.matmul(Acc, Rot)

    # Return the rotated accelerometer data
    return Acc


def Flip_InsideOut(Acc, Time=None):
    """
    A simple algorithm for detecting if the participant attached the device inside out. If time is used as input, only day time is considered.
    For all observations where the inclination of the leg is above 45, the median value of the z-axis is computed. The idea being that if the leg is
    inclined is is likely while sitting (especially during the day), and then it is possible to detect what side of the accelerometer is facing up by
    looking at the z-axis.

    Parameters
    ----------
    Acc : TYPE
        DESCRIPTION.
    Time : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    Flip : binary
        If the device is likely flipped or not.

    """
    # assume 3 columns of [x,y,z] in acc (so without time column)
    import numpy as np

    if Time is not None:
        T = np.fmod(
            Time, 1
        )  # datenumbers. computes remainder after division by 1 which corresponds to time of day
        NightStart = 22 / 24
        NightEnd = 8 / 24
        NotNight = np.all([T < NightStart, T > NightEnd], axis=0)
        Acc = Acc[NotNight, :]

    # Find inclination of leg for samples in daytime (to avoid stomach lying)
    Lng = np.sqrt(
        np.square(Acc[:, 0]) + np.square(Acc[:, 1]) + np.square(Acc[:, 2])
    )  # euclidean length of acc vectors
    Inc = (180 / np.pi) * np.arccos(np.divide(Acc[:, 0], Lng))

    # extract all observations for the z-axis where the inclination of the thigh is above 45 degrees.
    SitLieZ = Acc[Inc > 45, 2]
    # print(f'mean: {np.mean(SitLieZ)}')
    # print(f'median: {np.median(SitLieZ)}')
    median_value = np.median(SitLieZ)
    # if the median value is positive it indicates that the device might be attached flipped (could be troublesome if they spend all night sleeping on their stomach (fixed if Time is added as input))
    if median_value > 0:
        Flip = 1
    else:
        Flip = 0

    return Flip


def Flip_InsideOut_Timeless(Acc):
    """
    A simple algorithm for detecting if the participant attached the device inside out.
    For all observations where the inclination of the leg is above 45, the median value of the z-axis is computed. The idea being that if the leg is
    inclined is is likely while sitting (especially during the day), and then it is possible to detect what side of the accelerometer is facing up by
    looking at the z-axis.

    Timeless: Excludes periods of 3 hours if 98% of the time is spent in an angle above 45 degrees. Rolling window with half hour steps

    Assumes SF=30

    Parameters
    ----------
    Acc : TYPE
        DESCRIPTION.

    Returns
    -------
    Flip : binary
        If the device is likely flipped or not.

    """
    # Assume 3 columns of [x, y, z] in acc (without a time column)

    import numpy as np

    # Calculate the Euclidean length of acc vectors
    Lng = np.sqrt(np.square(Acc[:, -3]) + np.square(Acc[:, -2]) + np.square(Acc[:, -1]))

    # Calculate inclination angles in degrees
    Inc = (180 / np.pi) * np.arccos(np.divide(Acc[:, -3], Lng))

    # Initialize list to store indices of not valid data points
    not_valid_datapoints = []

    # Define indices for each hour in the data
    hourind = np.arange(0, len(Acc) - (30 * 60 * 180), 30 * 60 * 60)

    count = 0

    # Loop through each hour to check inclination values
    for i in hourind:
        inctmp = Inc[i : i + (30 * 60 * 180)]

        # Check if the 2nd percentile of inclination values is above 45 degrees
        if np.percentile(inctmp, 2) > 45:
            not_valid_datapoints.extend(np.arange(i, i + (30 * 60 * 180)))
        count += 1

    # Create modified arrays by removing not valid data points
    AccMod = Acc
    IncMod = Inc
    not_valid_datapoints = np.unique(not_valid_datapoints)

    if len(not_valid_datapoints) > 0:
        AccMod = np.delete(AccMod, not_valid_datapoints, axis=0)
        IncMod = np.delete(IncMod, not_valid_datapoints, axis=0)

    # Extract all observations for the z-axis where the inclination of the thigh is above 45 degrees
    SitLieZ = AccMod[IncMod > 45, 2]

    # Calculate the median value of SitLieZ
    median_value = np.median(SitLieZ)

    # If the median value is positive, it indicates that the device might be attached flipped
    if median_value > 0:
        Flip = 1
    else:
        Flip = 0

    # Return the calculated flip indicator
    return Flip


def Flip_UpsideDown(Acc, Time=None):
    """
    A simple algorithm for detecting if the participant attached the device upside down. If time is used as input, only day time is considered.

    For all observations where the inclination of the leg is either below 45 og above 135 degrees, the median value of the x-axis is computed. The idea being that if the leg is
    vertical it is likely while standing (especially during the day), and then it is possible to detect what orientation the accelerometer is attached by looking at the value of the x-axis.

    Parameters
    ----------
    Acc : TYPE
        DESCRIPTION.
    Time : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    Flip : binary
        If the device is likely flipped or not.

    """
    # assume 3 columns of [x,y,z] in acc (so without time column)
    import numpy as np

    if Time is not None:
        T = np.fmod(
            Time, 1
        )  # datenumbers. computes remainder after division by 1 which corresponds to time of day
        NightStart = 22 / 24
        NightEnd = 8 / 24
        NotNight = np.all([T < NightStart, T > NightEnd], axis=0)
        Acc = Acc[NotNight, :]

    # Find inclination of leg for samples in daytime (to avoid stomach lying)
    Lng = np.sqrt(
        np.square(Acc[:, -3]) + np.square(Acc[:, -2]) + np.square(Acc[:, -1])
    )  # euclidean length of acc vectors
    Inc = (180 / np.pi) * np.arccos(np.divide(Acc[:, -3], Lng))

    # extract all observations for the x-axis where the inclination of the thigh is below 45 degrees.

    StandX = Acc[np.any([Inc < 25, Inc > 155], axis=0), 0]

    median_value = np.median(StandX)

    # if the median value is positive it indicates that the device might be attached flipped (could be troublesome if they spend all night sleeping on their stomach (fixed if Time is added as input))
    if median_value < 0:
        Flip = 1
    else:
        Flip = 0

    return Flip


def Flip(Acc, SF=30, ts=None):
    """
    Uses other flip functions to check whether sensor is attached inside out or upside down.
    After checks, the data is flipped and returned.

    Parameters
    ----------
    Acc : numpy.ndarray
        Raw accelerometer data represented as a 4xn array of values [time, x, y, z].
    SF : float
        Scale factor for data adjustment.
    ts : int, optional
        Timestamps from sensor data (default is None).

    Returns
    -------
    numpy.ndarray
        Adjusted and flipped accelerometer data.

    Notes
    -----
    This function adjusts the raw accelerometer data by calling the 'AdjustRawBinData' function.
    It then checks orientation flags 'Flip_insideout' and 'Flip_upsidedown' to determine
    whether flipping is required. Flipping is performed by multiplying the corresponding
    columns in the acceleration data by appropriate scaling factors.

    If 'Flip_insideout' is True and 'Flip_upsidedown' is False:
    - The last three columns of the acceleration data are multiplied by [1, -1, -1].

    If both 'Flip_insideout' and 'Flip_upsidedown' are True:
    - The last three columns of the acceleration data are multiplied by [-1, 1, -1].

    If 'Flip_insideout' is False and 'Flip_upsidedown' is True:
    - The last three columns of the acceleration data are multiplied by [-1, -1, 1].

    The adjusted and flipped accelerometer data is then returned.
    """
    import numpy as np

    # Adjust raw accelerometer data
    Accref, Start, TimeT = AdjustRawBinData(Acc, SF, utc=True, ts=ts)

    # Check orientation flags
    Flip_insideout = Flip_InsideOut_Timeless(Accref)
    Flip_upsidedown = Flip_UpsideDown(Accref)

    # Flip accelerometer data based on orientation flags
    if Flip_insideout and (not Flip_upsidedown):
        Acc[:, -3:] = Acc[:, -3:] * np.array([1, -1, -1])

    elif Flip_insideout and Flip_upsidedown:
        Acc[:, -3:] = Acc[:, -3:] * np.array([-1, 1, -1])

    elif (not Flip_insideout) and Flip_upsidedown:
        Acc[:, -3:] = Acc[:, -3:] * np.array([-1, -1, 1])

    return Acc


def AutoCalibrate(Acc, t_win=10, t_step=10, seed=281597):
    """
    Van Hees algorithm for calibration of accelerometer data.
    For observations with low activity (moving std < 0.013 / still data), a linear regression is made for the three axes under the assumption that the
    magnitude (sqrt(x^2+y^2+z^2)) equals 1. Scale and offset parameters are found for each axis in order to shift all data.

    Parameters
    ----------
    Acc : numpy.ndarray
        Raw accelerometer data represented as a 4xn array of values [time, x, y, z].
    t_win : float, optional
        Time window for rolling statistics and filtering (default is 10).
    t_step : float, optional
        Time step between consecutive samples (default is 10).
    seed : int, optional
        Seed for randomization during point selection (default is 281597).

    Returns
    -------
    numpy.ndarray
        Calibrated accelerometer data.
    numpy.ndarray
        Scale factors for each axis.
    numpy.ndarray
        Offset values for each axis.

    Notes
    -----
    This function performs automatic accelerometer calibration using a robust method. It first filters
    the raw accelerometer data using a rolling window to compute the mean and standard deviation.
    Points with low variability are selected, and a random subset is chosen for calibration.
    The calibration is performed using linear regression, and the process is repeated iteratively
    with weight adjustments until convergence.

    If successful, the calibrated accelerometer data 'Acc', scale factors 'scale', and offset
    values 'offset' are returned. If the calibration is not successful or does not converge,
    appropriate messages are printed.

    """

    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    # Calibration parameters
    SF = 30
    actThresh = 0.013
    maxIter = 1000
    convCrit = 1e-9
    ptsPerAxis = 500

    # Default scale and offset values
    scale = np.ones(3)
    offset = np.zeros(3)

    # Sample indices based on time step
    smplsTS = np.arange(0, len(Acc), SF * t_step)

    # Create DataFrame for easier data manipulation
    Accdf = pd.DataFrame(Acc, columns=["X", "Y", "Z"])

    # Calculate filtering window
    FiltWin = SF * t_win

    # Compute rolling mean and standard deviation
    movMeanAcc = Accdf.rolling(FiltWin, center=True, min_periods=1).mean()
    movStdAcc = Accdf.rolling(FiltWin, center=True, min_periods=1).std()

    # Select points with low variability
    movMeanAcc = movMeanAcc.loc[smplsTS]
    movStdAcc = movStdAcc.loc[smplsTS]
    movMeanAcc = movMeanAcc[(movStdAcc <= actThresh).sum(axis=1) == 3]

    # Convert to numpy array for further processing
    movMeanAcc = movMeanAcc.to_numpy()

    # Select positive and negative valid points for each axis
    ptsPosValidX = np.nonzero(movMeanAcc[:, 0] >= 0.3)[0]
    ptsPosValidY = np.nonzero(movMeanAcc[:, 1] >= 0.3)[0]
    ptsPosValidZ = np.nonzero(movMeanAcc[:, 2] >= 0.3)[0]
    ptsNegValidX = np.nonzero(movMeanAcc[:, 0] <= -0.3)[0]
    ptsNegValidY = np.nonzero(movMeanAcc[:, 1] <= -0.3)[0]
    ptsNegValidZ = np.nonzero(movMeanAcc[:, 2] <= -0.3)[0]

    # Get lengths of valid points
    lNegX = len(ptsNegValidX)
    lPosX = len(ptsPosValidX)
    lNegY = len(ptsNegValidY)
    lPosY = len(ptsPosValidY)
    lNegZ = len(ptsNegValidZ)
    lPosZ = len(ptsPosValidZ)

    # Check if there are enough valid points for calibration
    validX = (lNegX > 1) and (lPosX > 1)
    validY = (lNegY > 1) and (lPosY > 1)
    validZ = (lNegZ > 1) and (lPosZ > 1)

    if validX and validY and validZ:
        # Randomize order of points
        np.random.seed(seed)
        ptsPosValidX = ptsPosValidX[np.random.permutation(lPosX)]
        ptsNegValidX = ptsNegValidX[np.random.permutation(lNegX)]

        ptsPosValidY = ptsPosValidY[np.random.permutation(lPosY)]
        ptsNegValidY = ptsNegValidY[np.random.permutation(lNegY)]

        ptsPosValidZ = ptsPosValidZ[np.random.permutation(lPosZ)]
        ptsNegValidZ = ptsNegValidZ[np.random.permutation(lNegZ)]

        # Select 500 or fewer points for each axis
        ptsPosValidX = ptsPosValidX[: min(ptsPerAxis, lPosX)]
        ptsNegValidX = ptsNegValidX[: min(ptsPerAxis, lNegX)]

        ptsPosValidY = ptsPosValidY[: min(ptsPerAxis, lPosY)]
        ptsNegValidY = ptsNegValidY[: min(ptsPerAxis, lNegY)]

        ptsPosValidZ = ptsPosValidZ[: min(ptsPerAxis, lPosZ)]
        ptsNegValidZ = ptsNegValidZ[: min(ptsPerAxis, lNegZ)]

        # Combine and remove duplicate points
        ptsValid = np.concatenate(
            (
                ptsPosValidX,
                ptsNegValidX,
                ptsPosValidY,
                ptsNegValidY,
                ptsPosValidZ,
                ptsNegValidZ,
            )
        )
        ptsValid = np.unique(ptsValid)

        # Selected valid points for calibration
        D_in = movMeanAcc[ptsValid, :]
        N = len(ptsValid)
        weights = np.ones(N)

        # Iteratively perform calibration until convergence or maximum iterations
        for i in range(maxIter):
            # Calculate calibrated data and support vector magnitudes
            data = np.tile(offset, (N, 1)) + D_in * np.tile(scale, (N, 1))
            svm = np.sqrt(np.sum(data**2, axis=1))

            # Normalize data to have unit magnitude
            target = data / np.reshape(svm, (N, -1))

            # Initialize arrays for gradient and offset
            gradient = np.zeros(3)
            off = np.zeros(3)

            # Perform linear regression per input axis to estimate scale and offset
            for j in range(3):
                reg = LinearRegression().fit(
                    data[:, j].reshape((-1, 1)), target[:, j], sample_weight=weights
                )

                off[j] = reg.intercept_
                gradient[j] = reg.coef_[0]

            # Update scale and offset values
            scaleOld = scale
            scale = scale * gradient
            offset = offset + off

            # Compute errors and update weights for next iteration
            errors = abs(svm - 1)
            weights = np.min(np.column_stack((1 / errors, np.ones(N) * 100)), axis=1)

            # Check convergence based on scale changes
            convgE = sum(abs(scale - scaleOld))
            converged = convgE < convCrit

            # Calculate mean error
            meanE = np.mean(errors)

            # Break the loop if convergence is achieved
            if converged:
                break

        # Check if calibration was successful or if there was convergence
        if meanE > 0.02:
            # Calibration not successful
            print("Calibration not successful")
        else:
            # Apply calibrated scale and offset to the original accelerometer data
            Acc = offset + Acc * scale
            if i == maxIter:
                # No convergence within maximum iterations
                print("No convergence")
    else:
        # No valid data for calibration
        print("No valid data for calibration")

    # Return the calibrated accelerometer data, scale factors, and offset values
    return Acc, scale, offset


# def ref_angle_auto(Acc, SF=30):
#     """
#     Description
#     ------------
#     Estimates the reference angle for thigh accelerometer placement.
#     Estimation idea is based on finding occasions of participant standing still, assuming that this is the reference point for other movement.
#         1.	Compute average axes rotations for each second based on data from two connecting seconds (average from acc60)
#         2.	Compute standard deviations for each second based on data from two connecting secongs (standard deviation from acc60)
#         3.	Find maximum standard deviation across all three axes each second
#         4.	Sample z-axis rotations based on to conditions
#             a.	Secondwise x-axis rotations are less than 25 degrees (threshold picked by comparing 25, 30, 35 and 40 degrees)
#             b.	Secondwise maximal standard deviation is less than 0.1 (same threshold as used for stand/move in ActivityDetect)
#         5.	Get observed median z-axis rotations from the sampled z-axis rotations
#         6.  Add mean deviation from manual reference angles from training data

#     Parameters:
#     -----------
#         Acc (n*3 np.array): Matrix of axes rotations
#         SF (int): Number of observations per second (upsampled from either 25 or 12.5 Hz)

#     Returns
#     --------
#         corr_median_value (float): Corrected median z-axis rotation
#     """

#     import numpy as np

#     Acc12 = Acc60(Acc, SF)
#     Std = np.std(Acc12, axis=0)
#     Mean = np.mean(Acc12, axis=0)
#     Lng = np.sqrt(np.square(Mean[:,0]) + np.square(Mean[:,1]) + np.square(Mean[:,2])) #euclidean length of mean vectors

#     STD = np.max(Std, axis=1)

#     Inc = (180/np.pi)*np.arccos(np.divide(Mean[:,0],Lng)) #inclination of leg  (0-180 degrees)
#     FB = -(180/np.pi)*np.arcsin(np.divide(Mean[:,2],Lng)) #forward/backward angle of leg

#     standstill = FB[((Inc<25)&(STD<0.1))]

#     median_value = np.median(standstill)
#     corr_median_value = median_value - 1.0685

#     return corr_median_value


def DiaryFun(Diaryfile, timezone):
    """
    Read diary data from a CSV file and create a DataFrame of intervals.

    Parameters
    ----------
    Diaryfile : str
        Path to the CSV file containing diary data.
    timezone : str
        Timezone to be used for timestamp conversion.

    Returns
    -------
    pd.DataFrame or None
        DataFrame containing intervals and their types. Returns None if there are missing end times for periods.
    """

    from datetime import timedelta
    import pandas as pd

    # Read diary data from CSV file
    Diary = pd.read_csv(Diaryfile, compression="infer", on_bad_lines="skip")
    Dperiods = Diary[Diary[" entry_type"] == " period"]

    # Initialize DataFrame to store intervals and types
    IntervalDB = pd.DataFrame(columns=["Interval", "Type"])
    intervals = []
    types = []
    maxslut = 0

    for index, row in Dperiods.iterrows():
        if row[" end_time"] == " ":
            # Handle missing end time for intervals
            typ = row[" type"]
            print(f"Missing end time of {typ} interval")
            return None

        start = pd.Timestamp(row.date + row[" start_time"], tz=timezone)
        slut = pd.Timestamp(row.date + row[" end_time"], tz=timezone)

        # Adjust start time if going to bed after midnight or working night shift
        if start.hour > slut.hour:
            start = start - timedelta(days=1)

        # Create interval and add to lists
        inval = pd.Interval(start, slut, closed="both")
        intervals.append(inval)
        types.append(row[" type"])

        # Update maximum hour for AM/PM check
        if max(slut.hour, start.hour) > maxslut:
            maxslut = max(slut.hour, start.hour)

    # Update DataFrame with intervals, types, and duration
    IntervalDB.Interval = intervals
    IntervalDB.Type = types
    IntervalDB["Duration"] = [interval.length for interval in IntervalDB.Interval]

    # Check for AM/PM inconsistency
    if maxslut < 13:
        if " work" not in types:
            pass
        else:
            return None

    return IntervalDB


def find_ref_angle_unix(AccAutoCal, Actout, TimeT, timezone):
    """
    Calculate reference angles based on accelerometer data and calibration timestamp.

    Parameters
    ----------
    AccAutoCal : numpy.ndarray
        Processed accelerometer data after calibration.
    Actout : pd.DataFrame
        Processed data containing calibration information.
    TimeT : numpy.ndarray
        Timestamps associated with the accelerometer data.
    timezone : str
        Timezone to be used for timestamp conversion.

    Returns
    -------
    float, float
        Latitude angle and median angle calculated based on the accelerometer data around the calibration timestamp.
        Returns NaN for both angles if calibration is before the start of raw data.
    """

    import pandas as pd
    import numpy as np

    # Convert the calibration timestamp to the specified timezone
    cal_start_time = pd.to_datetime(Actout.utc[0], utc=True).tz_convert(timezone)
    print(f"Calibration time: {cal_start_time:%d/%m/%Y %H:%H:%S}")

    # Round the calibration timestamp for comparison with raw data timestamps
    cal_start_time = np.round(Actout[" unixts"][0] / 1000)

    # Select a suitable time window from the raw data for analysis
    timeT = TimeT[:3888000]  # 36 hours
    rawtime = np.round((timeT - 719529) * 86400)

    # Check if calibration is before the start of raw data
    if cal_start_time < rawtime[0]:
        print("Calibration is before the start of raw data")
        return np.nan, np.nan

    startcalRaw = rawtime[cal_start_time == rawtime]

    # Check if there is a timestamp matching the calibration start time
    if len(startcalRaw) < 1:
        return np.nan, np.nan
    else:
        startcalRaw = startcalRaw[0]

    # Find the index of the calibration start time in the raw data
    calstartindex = np.where(rawtime == startcalRaw)[0][0]

    # Extract a window of accelerometer data around the calibration timestamp
    Acccal = AccAutoCal[calstartindex - 25 * 15 : calstartindex, :]

    # Calculate mean and median angles based on the accelerometer data
    [x, y, z] = np.mean(Acccal, axis=0)
    [x1, y1, z1] = np.median(Acccal, axis=0)

    # Calculate median angle, latitude angle, and return the values
    median_angle = -(180 / np.pi) * np.arcsin(z1 / np.sqrt(x1**2 + y1**2 + z1**2))
    LatAngle = -(180 / np.pi) * np.arcsin(y1 / np.sqrt(x1**2 + y1**2 + z1**2))

    return LatAngle, median_angle
