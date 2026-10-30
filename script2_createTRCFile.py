# ---------------------------------------------------------------------------- #
#                           script2_createTRCFile.py                           #
# ---------------------------------------------------------------------------- #
# Copyright (c) 2022-2026 University of Ottawa and the Authors                 #
# Author(s): Ayham AlAkhras, Firas Baklouti, Thomas Uchida                     #
#                                                                              #
# Licensed under the Apache License, Version 2.0 (the "License"); you may not  #
# use this file except in compliance with the License. You may obtain a copy   #
# of the License at http://www.apache.org/licenses/LICENSE-2.0.                #
#                                                                              #
# Unless required by applicable law or agreed to in writing, software          #
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT    #
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the     #
# License for the specific language governing permissions and limitations      #
# under the License.                                                           #
#                                                                              #
# This work has been made possible in part by grant 2020-225400-5022 from the  #
# Chan Zuckerberg Initiative Donor-Advised Fund (DAF), an advised fund of      #
# Silicon Valley Community Foundation.                                         #
#                                                                              #
# Please cite the following paper in work that uses this script:               #
#   AlAkhras A, Baklouti F, Bailey CA, Graham RG, Uchida TK (2026).            #
#   Calibration and verification of OpenSim models from Theia3D markerless     #
#   motion capture.                                                            #
# ---------------------------------------------------------------------------- #

# This script generates a .trc file containing position information on joint
# origins from Theia3D for use in OpenSim. The input to this script must first
# be generated using script1_parseTheiaData. If you want to use only orientation
# data from Theia3D in this workflow, set the flag 'justRootOrigins' to 'True'.
#
# This script was built using OpenSim 4.5 and Python 3.11 and tested with data
# from Theia3D 2022.2.
#
# version 1.0

import pickle


# ===========================  P A R A M E T E R S  ============================

# File name of the body segment origin location file from script1 (*.pkl).
originFileName = 'Output/theiaOrigins.pkl'

# Output file name containing body position information from Theia3D (*.trc).
outputTRCFileName = 'Output/theiaOrigins.trc'

# Set the capture rate of the Theia3D data (Hz).
dataRate = 60

# Set the root bodies to stay where they were at a specified frame number (in
# base-0) of the motion. Note that this time-step is relative to the start frame
# defined in script1. This also ensures that all other bodies are in their
# correct position relative to the root bodies throughout the motion. Set this
# to `True` if you would like the model to not translate in 3D space (common for
# orientation only workflows such as IMU inverse kinematics in OpenSim).
staticRootOrigins = False
staticFrame = 0

# Create a file with only root body information; all other bodies are omitted.
# Set this to `True` if you would like to map only orientation data from Theia3D
# into OpenSim.
# NOTE: Orientation information is still processed for all bodies.
justRootOrigins = False


# ===============================  S C R I P T  ================================

torsoBodies = ['l_hand',
                'l_larm',
                'l_uarm',
                'r_hand',
                'r_larm',
                'r_uarm']
pelvisBodies = ['l_foot',
                'l_shank',
                'l_thigh',
                'l_toes',
                'r_foot',
                'r_shank',
                'r_thigh',
                'r_toes',
                'pelvis_shifted']
rootBodies = ['head',
                'torso',
                'pelvis']
offset = {}
originVectorStatic = {}

# Load the origin dict.
originFile = open(originFileName, 'rb')
originDict = pickle.load(originFile)
originFile.close()

# Obtain the number of frames.
frameCount = len(originDict)
bodyList = list(originDict[list(originDict.keys())[0]].keys())

# Remove bodies if necessary.
if (justRootOrigins):
    tempList = bodyList.copy()
    for i, name in enumerate(bodyList):
        if (name=='head' or name=='torso' or name=='pelvis'):
            pass
        else:
            tempList.remove(name)
            for t in originDict:
                originDict[t].pop(name)
    bodyList = tempList.copy()

# Obtain the number of frames.
markerCount = len(bodyList)

# Create the trc file.
trcFile = open(outputTRCFileName, 'w')

# Write the header lines.
trcFile.write(f'PathFileType\t4\t(X/Y/Z)\t{outputTRCFileName}\n')
trcFile.write('DataRate\tCameraRate\tNumFrames\tNumMarkers\t'
               'Units\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n')
trcFile.write(f'{dataRate:.6f}\t{dataRate:.6f}\t{frameCount}\t{markerCount}\t'
               f'm\t{dataRate:.6f}\t0\t{frameCount}\n')
trcFile.write('Frame#\tTime\t')
for i, name in enumerate(bodyList):
    if (i < markerCount-1):
        trcFile.write(name + '_m\t\t\t')
    else:
        trcFile.write(name + '_m\n')
trcFile.write('\t\t')
for i in range(markerCount):
    if (i < markerCount-1):
        trcFile.write(f'X{i+1}\tY{i+1}\tZ{i+1}\t')
    else:
        trcFile.write(f'X{i+1}\tY{i+1}\tZ{i+1}\t\n')
trcFile.write('\n')

# Write the origin data from line 7 onwards.
for timeStep, t in enumerate(originDict, 1):
    # Obtain the dict for this frame of data.
    dataLine = originDict[t]
    dataLineList = list(dataLine)
    
    # Write the frame number and time.
    trcFile.write(f'{timeStep}\t{t}\t')
    
    # Obtain the offset of the root bodies for each time-step if necessary.
    if (staticRootOrigins):
        for bodyName in rootBodies:
            offset[bodyName] = \
                originDict[list(originDict.keys())[staticFrame]][bodyName] - \
                                                              dataLine[bodyName]
            originVectorStatic[bodyName] = \
                originDict[list(originDict.keys())[staticFrame]][bodyName]
    
    # Loop through the dataLine dict to obtain data for each body individually.
    for bodyName in dataLine:
        # Obtain the position of the current body for the current time-step.
        originVector = dataLine[bodyName]
        
        # Modify the origin vector to keep the child bodies consistent with the
        # roots if necessary.
        if (staticRootOrigins):
            if (bodyName in torsoBodies):
                originVector = originVector + offset['torso']
            elif (bodyName in pelvisBodies):
                originVector = originVector + offset['pelvis']
            elif (bodyName in rootBodies):
                originVector = originVectorStatic[bodyName]
            else:
                raise Exception(f'ERROR: Unrecognized body {bodyName}')
        
        # Write the origin vector to the file.
        originString = (f'{originVector[0]}\t'
                        f'{originVector[1]}\t'
                        f'{originVector[2]}')
        if (bodyName == dataLineList[-1]):
            trcFile.write(originString + '\n')
            break
        else:
            trcFile.write(originString + '\t')

# Close the file and print done message.
trcFile.close()
print(f'Done! TRC file created: {outputTRCFileName}')
