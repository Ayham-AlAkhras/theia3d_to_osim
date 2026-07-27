# ---------------------------------------------------------------------------- #
#                          script1_parseTheiaData.py                           #
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

# This script processes a .txt from Theia3D containing transformation matrices
# and generates a .pkl that contains movement data. This file can then be
# written to a .trc file containing joint center locations (origins) using
# script2_createTRCFile. The parser has the following tests to ensure that the
# file is in the correct format and that the information makes sense:
#
#   1. The first row must contain a number of entries that is a multiple of 16.
#   2. The first four rows must have the same number of entries.
#   3. There must be 16 unique, consecutive transformation matrix indices for
#      each body segment.
#   4. Each line of data must be of the same length.
#   5. The value of each entry must be a floating-point number.
#   6. There cannot be any perspective projections.
#   7. All rotation matrices must be orthogonal.
#       - The matrix multiplied by its transpose must result in an identity
#         matrix.
#       - The determinant of a matrix must equal 1.
#       - The inverse of a matrix must be equal to its transpose.
#   8. The world-frame rotation matrix must be an identity matrix throughout the
#      motion.
#
# The data used for this script is not the native output from Theia3D. If you
# have .c3d files from Theia3D, convert them to .txt files using Visual3D. Each
# .txt file will contain 4×4 transforms in row-major form for each body segment
# (columns) and time step (rows).
#
# This script was built using OpenSim 4.5 and Python 3.11 and tested with data
# from Theia3D 2022.2.
#
# version 1.0

import time
import numpy as np
import pickle


# ===========================  P A R A M E T E R S  ============================

# Set file name of input file (*.txt). If you have .c3d files from Theia3D,
# convert them to a .txt file using Visual3D. Note that OpenSim 4.4 cannot
# process the .c3d files from Theia3D. The .txt should contain 4x4 transforms in
# row-major for each body segment (columns) and time-step (rows). Even though
# the scripts can address this given the correct parameters, it's best if the
# Theia3D data is spatially and temporally synchronized to the marker data.
theiaFileName = 'SampleData/P01_tr03_normal_4x4.txt'

# Set output file names (*.pkl).
outputOriginFileName = 'Output/theiaOrigins.pkl'
outputOrientationFileName = 'Output/theiaOrientations.pkl'

# Set start and end time-steps (in base-0, inclusive) for the script to parse
# (set to 0 and -1, respectively, to parse all lines in the file).
startTimeStep = 2716
endTimeStep = 3540

# Set the capture rate of the Theia3D data (Hz).
dataRate = 60

# Exclude these bodies from parsing.
excludedBodies = ['worldbody','pelvis_shifted']#,'vlb'

# Flag to enable/disable orthogonality test (check 7).
# NOTE: This check is computationally expensive and greatly increases runtime.
testOrthogonality = True

# Flag to enable/disable world reference frame rotation matrix test (check 8).
testWorldIdentity = True

# Specify the rotation matrix that is multiplied by the Theia3D data to rotate
# the model such that the model is upright (Y-axis should point up). A
# world-fixed rotation of -90 degrees about the x-axis then a world-fixed
# rotation of 180 degrees about the y-axis is the default assignment.
# NOTE: There's no check for this; we trust it's a valid rotation.
rotationTheiaToOsim = np.array([[-1,0,0],
                                [0,0,1],
                                [0,1,0]])

# Set the tolerances that account for machine precision in the orthogonality 
# check (check 7).
TOL_MAT = 10e-5  # The tolerance for the difference in rotation matrix elements.
TOL_DET = 10e-5  # The tolerance for the difference in determinants of matrices.

# Set the level of logging required according to OpenSim's convention:
#   3 (OFF):    Do not log any messages. Useful when running an optimization or
#               automated pipeline.
#   
#   2 (INFO):   Default.
#   
#   1 (DEBUG):  Log information that may be useful when debugging the operation
#               of the software to investigate unexpected results.
#   
#   0 (TRACE):  Log as much as possible, including messages that describe the
#               software's behavior step by step.
LOG_LEVEL = 2


# ===============================  S C R I P T  ================================

if (LOG_LEVEL <= 2):
    timerStartEntire = time.time() # Timer for the entire script.
if (LOG_LEVEL <= 1):
    timerStartRead = time.time() # Timer for reading in the file.

# Read in the file according to the number of lines specified.
with open(theiaFileName) as file:
    startIndex = startTimeStep+5
    if (endTimeStep > 0):
        endIndex = endTimeStep+5+1
    elif (endTimeStep == -1):
        endIndex = None
    dataReadAll = file.readlines()
    dataReadAll = dataReadAll[0:5] + dataReadAll[startIndex:endIndex]
    numberOfLines = len(dataReadAll)

if (LOG_LEVEL <= 1):
    timerEndRead = time.time() # Timer for reading in the file.
    print(f'Time taken to read file = {timerEndRead - timerStartRead} seconds')

# Convert the rotation matrix used to go from Theia's world frame to OpenSim's
# to a transformation with no translation or perspective warp.
transformTheiaToOsim         = np.zeros((4,4))
transformTheiaToOsim[:3, :3] = rotationTheiaToOsim
transformTheiaToOsim[3, 3]   = 1

# ---------------------------------- CHECK 1 --------------------------------- #
# Check that the first row contains a number of entries that is a multiple of
# 16 (there are 16 elements in a transformation matrix).
dataReadLine1Length = dataReadAll[1].count('\t')
if (dataReadLine1Length%16==0 and dataReadLine1Length!=0):
    if (LOG_LEVEL <= 1):
        print('SUCCESS: First line contents are a multiple of 16.')
else:
    if (LOG_LEVEL <= 2):
        print('WARNING: First line contents are not a multiple of 16.')
# ---------------------------------------------------------------------------- #

# ---------------------------------- CHECK 2 --------------------------------- #
# Check that the first four rows have the same number of entries.
for iteration, currentDataLine in enumerate(dataReadAll[:4]):
    if (dataReadLine1Length != currentDataLine.count('\t')):
        if (LOG_LEVEL <= 2):
            print(f'WARNING: Length of line {iteration+1} differs from length '
                   'of first line.')
if (LOG_LEVEL <= 1):
    print('SUCCESS: All title lines are of the same length.')
# ---------------------------------------------------------------------------- #

if (LOG_LEVEL <= 1):
    timerStartLoop = time.time() # Timer for the entire main loop

# NOTE: In the main loop, 6 lines of data are ignored (first 5 and last 1).
if (LOG_LEVEL <= 1):
    timerStoreIteration = [0] * (numberOfLines-5-startTimeStep)

# ---------------------------------- CHECK 3 --------------------------------- #
# Check that there are 16 unique, consecutive matrix indices for each body
# segment.
# NOTE: Indices are stored in the fifth line of the file.
dataReadLine4 = dataReadAll[4].split()
dataReadLine4Length = len(dataReadLine4)
indexShouldBe = 0

# NOTE: We iterate over length-1 entries because the first column is for
# indexing the frames only.
for index in range(0, dataReadLine4Length-1):
    indexActuallyIs = int(dataReadLine4[index+1])
    indexShouldBe = index - 16*int(index/16)
    if (indexShouldBe != indexActuallyIs):
        if (indexShouldBe-1 == indexActuallyIs):
            raise Exception(f'ERROR: Repeated item at index {index} on line 4.')
        else:
            raise Exception(f'ERROR: Item numbering inconsistent at index '
                                                          f'{index} on line 4.')
# ---------------------------------------------------------------------------- #

# Defining dicts for the main loop.
orientDict     = {} # The main orientation dict storing all information.
originDict     = {} # The main origin dict storing all information.
lineOrientDict = {} # The orientation dict storing a single line at a time.
lineOriginDict = {} # The origin dict storing a single line at a time.

# Pre-allocating arrays for the main loop.
projectionRow   = np.empty(4,     dtype=float)
translationCol  = np.empty(3,     dtype=float)
rotationMatrix  = np.empty((3,3), dtype=float)
transformMatrix = np.empty((4,4), dtype=float)
projectionCheck = np.array([0,0,0,1])
identity        = np.eye(3)

# Defining some misc. variables for the main loop.
bodyNames = dataReadAll[1].split()
lineDeleted = False

# Main parsing loop, the loop skips the first five lines as they hold no
# movement data.
for iteration, currentDataLine in enumerate(dataReadAll[5:]):
    # Calculate the line number in the file based on the iteration in this loop.
    # +1 to change from zero-based to one-based and +5 to skip the first 5 lines
    # of the file.
    lineNumber = iteration + 1 + 5 + (startTimeStep)
    
    # Create a list for the current line of data.
    currentDataList = currentDataLine.split()
    
    # Check that the index of each line makes sense.
    # NOTE: -1 because the item column is in base-1, we use base-0
    if (int(currentDataList[0]) != (iteration+startTimeStep+1)):
        print(f'WARNING: Inconsistent line index on line {lineNumber},'
              f'index should be {iteration+startTimeStep+1}, but actually is '
              f'{currentDataList[0]}.')
    
    # Remove the indexing column and count the number of elements in this line.
    currentDataList.pop(0)
    lineLength = len(currentDataList)
    
    if (LOG_LEVEL <= 1):
        timerStartIteration = time.time() # Timer for each iteration.
    
    # -------------------------------- CHECK 4 ------------------------------- #
    # CHECK 4: Check that every line of data is of the same length.
    # NOTE: This ignores incomplete lines only if they are at the end of the
    # file.
    if (dataReadLine1Length != lineLength):
        if (LOG_LEVEL <= 2):
            print(f'WARNING: Line {lineNumber} omitted because it is '
                   'incomplete.')
        lineDeleted = True
        continue
    else:
        if not (lineDeleted):
            if (LOG_LEVEL <= 0):
                print(f'Line {lineNumber} is complete.')
        else:
            raise Exception('ERROR: Lines of data are of inconsistent lengths '
                     f'at line {lineNumber}, item {iteration+startTimeStep+1}.')
    # ------------------------------------------------------------------------ #
    
    lineOrientDict = {}
    lineOriginDict = {}
    
    # Loop to check and set the values from the transformation matrices of all
    # bodies for a single time-step.
    for lineElement in range(0, lineLength):
        
        # Theia stores the transformation matrices of all bodies as a single
        # line of numbers in row-major. Therefore, we require a counter to keep
        # track of the current element in the matrix.
        transformElement = (lineElement) % 16
        
        # Obtain the row and column in base-zero for each element in its
        # rotation matrix.
        # The order of the 16 entries of each transformation matrix from Theia
        # is in row-major. Therefore, the indices of a transformation matrix are
        # defined as:
        # Transformation matrix indices = 00 01 02 03
        #                                 04 05 06 07
        #                                 08 09 10 11
        #                                 12 13 14 15
        row = transformElement // 4
        col = transformElement % 4
        
        # Remove '_4x4' and '_4X4' from the current body name.
        bodyName = bodyNames[lineElement].lower().replace('_4x4','')
        
        # ------------------------------ CHECK 5 ----------------------------- #
        # Check that the value of each entry is a float.
        currentDataElementString = currentDataList[lineElement]
        try:
            currentDataElementFloat = float(currentDataElementString)
        except:
            raise Exception(f'ERROR: Invalid data type at index {lineElement} '
                            f'on line {lineNumber}. Row {row} column {col} '
                            f'= {currentDataElementString}')
        # -------------------------------------------------------------------- #
        
        transformMatrix[row][col] = currentDataElementFloat
        
        # If statement to check that the transformation matrix is done being
        # filled for this body.
        if (transformElement == 15):
            # Rotate the transformation such that the model points up.
            transformMatrix = np.dot(transformTheiaToOsim, transformMatrix)
            
            # Split the transformation matrix.
            projectionRow = transformMatrix[3,:]
            translationCol = transformMatrix[:3,3]
            rotationMatrix = transformMatrix[:3,:3]
            
            # ---------------------------- CHECK 6 --------------------------- #
            # Check that there is no perspective projection.
            if not (np.all(np.equal(projectionRow, projectionCheck))):
                raise Exception(f'ERROR: projection row not [0,0,0,1] on line '
                                f'{lineNumber}.')
            # ---------------------------------------------------------------- #
            
            # Assign the rotation and translation information to a dict with
            # body names as keys.
            lineOrientDict[bodyName] = rotationMatrix
            lineOriginDict[bodyName] = translationCol
            
            # ---------------------------- CHECK 7 --------------------------- #
            # Check that every rotation matrix is orthogonal.
            if (testOrthogonality):
                # CHECK 7.1: Check that the matrix multiplied by its transpose
                # results in an identity matrix.
                rotationMatrixTp = np.transpose(rotationMatrix)
                productMat = np.dot(rotationMatrix, rotationMatrixTp)
                if not (np.allclose(productMat, identity,
                                    rtol=0, atol=TOL_MAT)):
                    raise Exception(f'ERROR: Rotation matrix for {bodyName} '
                                    f'on line {lineNumber} is not orthogonal.')
                
                # CHECK 7.2: Check that the determinant of the matrix equals 1.
                # NOTE: Determinant must be 1 (never -1) because a right-handed.
                # coordinate system is used in Theia.
                rotMatDet = np.linalg.det(rotationMatrix)
                if (1-TOL_DET > (rotMatDet) > TOL_DET+1):
                    print('WARNING: Determinant is not within the tolerance.'
                         f'({TOL_DET}), on line {iteration+startTimeStep+1}.')
                
                # CHECK 7.3: Check that the inverse of the matrix is equal to
                # its transpose.
                rotationMatrixInv = np.linalg.inv(rotationMatrix)
                if not (np.allclose(rotationMatrixTp, rotationMatrixInv,
                                    rtol=0, atol=TOL_MAT)):
                    print('WARNING: values of the rotation matrix inverse and '
                         f'rotation matrix transpose do not match on line '
                         f'{iteration+startTimeStep+1}.')
            # ---------------------------------------------------------------- #
            
            # ---------------------------- CHECK 8 --------------------------- #
            # Check that the world frame rotation matrix an identity matrix in
            # every frame of the motion.
            if (testWorldIdentity):
                if (bodyName == 'worldbody'):
                    # NOTE: Here, we use the rotation from Theia to OpenSim (not
                    # identity) because the current rotation matrix is already
                    # rotated to the OpenSim frame.
                    if not (np.allclose(rotationMatrix, rotationTheiaToOsim,
                                        rtol=0, atol=TOL_MAT)):
                        print('WARNING: Rotation matrix for world frame on '
                             f'line {lineNumber} is not an identity matrix in '
                              'the Theia frame.')
            # ---------------------------------------------------------------- #
            
            # Re-declare the transform array.
            # NOTE: Arrays must be re-declared such that the dict values are
            # unique. The dict essentially points to the assigned array, so not
            # declaring the array here means that all entries of the dict for
            # all bodies would reference a single array.
            transformMatrix = np.empty((4,4), dtype=float)
    
    for bodyToRemove in excludedBodies:
        lineOrientDict.pop(bodyToRemove)
        lineOriginDict.pop(bodyToRemove)
    
    # After an iteration is complete, the resulting dictionary is added to the
    # main dictionary with the key being the time at the current time-step.
    timeWrite = iteration/dataRate + startTimeStep/dataRate
    orientDict[timeWrite] = lineOrientDict
    originDict[timeWrite] = lineOriginDict
    
    if (LOG_LEVEL <= 0):
        timerEndIteration = time.time() # Timer for each iteration.
        print('Time for iteration '
              f'{iteration + 1} = {timerEndIteration - timerStartIteration} '
              'seconds.')
    if (LOG_LEVEL <= 1):
        timerEndIteration = time.time() # Timer for each iteration.
        timerStoreIteration[iteration]=(timerEndIteration - timerStartIteration)

if (LOG_LEVEL <= 1):
    # Average time for all steps in the main loop.
    timerIterationAvg = sum(timerStoreIteration)/len(timerStoreIteration)
    print(f'Average time per iteration = {timerIterationAvg} seconds.')
    print(f'Longest iteration took {max(timerStoreIteration)} seconds.')
if (LOG_LEVEL <= 1):
    timerEndLoop = time.time() # Timer for the entire main loop.
    print(f'Loop time = {timerEndLoop - timerStartLoop} seconds.')
    
# Export the data.
writeOrients = open(outputOrientationFileName, 'wb')
pickle.dump(orientDict,writeOrients)
writeOrients.close()
writeOrigins = open(outputOriginFileName, 'wb')
pickle.dump(originDict,writeOrigins)
writeOrigins.close()

if (LOG_LEVEL <= 2):
    timerEndEntire = time.time() # Timer for the entire script.
    print(f'\nTotal time: {timerEndEntire - timerStartEntire} seconds.')

# Done message.
if (LOG_LEVEL <= 2):
    print(f'\nDone! Two PKL files created:\n',
          outputOriginFileName,'\n',
          outputOrientationFileName)
