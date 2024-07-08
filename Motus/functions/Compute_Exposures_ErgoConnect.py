# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:10:03 2024

@author: b061621
"""

# def Compute_Exposures(Akt):
#     """
#     Compute the exposures needed for the ErgoConnect project, based on the 1Hz data 
#     output from step 2. 

#     Parameters
#     ----------
#     Akt : numpy array
#         length same as ts
#         .

#     Returns
#     -------
#     out : TYPE
#         DESCRIPTION.

#     """
#     import numpy as np
#     out = Akt.copy()

#     out['Sedentary_time'] = np.any([out.Activity == 1, out.Activity == 2], axis=0)
#     out['Standing_time'] = np.any([out.Activity == 3, out.Activity == 4], axis=0)
#     out['Walking_time'] = out.Activity == 5


#     #Nrisesit
#     DiffAkt = (out["Activity"].isin([1,2]) * 1).to_numpy()
#     DiffAkt = np.diff(DiffAkt, prepend=DiffAkt[0])
#     out['Rise_from_sedentary_number'] = (DiffAkt == -1)

#     ThresTrunk = np.array([30, 60, 180])
#     ThresArm = np.array([30, 60, 90, 180])

#     for idx in range(len(ThresTrunk) - 1):
#         out[f'Forward_Bending_{ThresTrunk[idx]}_to_{ThresTrunk[idx+1]}_time'] = np.all(
#             [
#                 out.Activity > 2,
#                 out.Activity < 8,
#                 out.TrunkFB > 0,
#                 out.TrunkInc >= ThresTrunk[idx],
#                 out.TrunkInc <= ThresTrunk[idx + 1],
#             ],
#             axis=0,
#         )
#     forward45 = np.all(
#             [
#                 out.Activity > 2,
#                 out.Activity < 8,
#                 out.TrunkFB > 0,
#                 out.TrunkInc >= 45,
#                 out.TrunkInc <= 180,
#             ],
#             axis=0,
#         )

#     # out['forward45'] = forward45
#     Diff45 = np.diff(forward45, prepend=0)
#     # out['diff45'] = Diff45
#     # Start = (np.array(np.nonzero(Diff45 == 1)))[0]
#     # Slut = (np.array(np.nonzero(Diff45 == -1)) - 1)[0]
#     # SSdur = Slut - Start + 1

#     # crossing 45 from below 
#     #up45 = (np.array(angles[1:]) > 45) & (np.array(angles[:-1]) <= 45)
#     # up45 = np.insert(up45, 0, 0)
#     #crossing 30 from above
#     #down30 = (np.array(angles[1:]) < 30) & (np.array(angles[:-1]) >= 30)
#     #down30 = np.insert(down30, 0, 0)

#     out['Forward_Bending_45_to_180_number'] = Diff45 == 1

#     for idx in range(len(ThresArm) - 1):
#         out[f'Arm_Lifting_{ThresArm[idx]}_to_{ThresArm[idx+1]}_time'] = np.all(
#             [
#                 out.Activity > 2,
#                 out.Activity < 6,
#                 out.ArmInc >= ThresArm[idx],
#                 out.ArmInc <= ThresArm[idx + 1],
#             ],
#             axis=0,
#         )

#     return out



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

