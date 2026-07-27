# ---------------------------------------------------------------------------- #
#                     script4_runFusedInverseKinematics.py                     #
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

# The Theia3D data give us 4×4 transforms containing the frame origin location
# and orientation for each body segment at each time step. Specifically, each
# "origin" is the location of a joint center from Theia3D, which is equivalent
# to the location of a joint's child-body frame in the TLO model. Each origin
# has a corresponding marker attached to the body segment with relative position
# of (0,0,0). Each "orientation" describes the orientation of a body-segment
# frame. This script runs inverse kinematics (IK) on the Theia3D data, using
# just the origins or using both the origins and the orientations.
#
# This script was built using OpenSim 4.5 and Python 3.11 and tested with data
# from Theia3D 2022.2.
#
# version 1.0

import opensim as osim
import numpy as np
import pickle
import pandas
import os


# ===========================  P A R A M E T E R S  ============================

# File name of the TLO model (*.osim).
modelFileName = 'Output/tloModel.osim'

# File name of the origin file (*.trc).
originsFileNameTRC = 'Output/theiaOrigins.trc'

# File name of the origin file (*.pkl).
originsFileNamePKL = 'Output/theiaOrigins.pkl'

# File name of the orientation file (*.pkl).
orientsFileNamePKL = 'Output/theiaOrientations.pkl'

# Define output motion filename (*.mot).
outputCoordFileName = 'Output/tloCoordinates.mot'

# Locations filename (*.csv).
outputLocationsFileName = 'Output/tloLocations.csv'


# Set the inverse kinematics mode:
#   Origins (False): Perform inverse kinematics using origin data: the 
#                    translation vectors representing the position of the origin
#                    of each body frame from Theia3D.
#
#   Fused (True):    Perform inverse kinematics using both origin and 
#                    orientation data.
#
#   *Orientations:   If you would like to perform inverse kinematics using
#                    orientation data, use the flags in Script 2 to generate a
#                    .trc file containing the origin data of the root bodies
#                    only. You can also enable the 'staticRootBodies' flag,
#                    which fixes the root origins to a specified time-step for
#                    the entire motion. You can then run the 'fused' inverse
#                    kinematics mode in this script to generate the motion. Be
#                    sure to disable the result checking flag when using static
#                    origin information.
includeOrientations = True

# Define the weights for the origin tracking. The global origin weight is
# multiplied by all the origin weight values which is useful when weighting
# origins and orientations.
originWeightGlobal = 1.0
originWeightValues = {'head'       : 1.0,
                      'torso'      : 1.0,
                      'r_clavicle' : 1.0,
                      'l_clavicle' : 1.0,
                      'r_uarm'     : 1.0,
                      'l_uarm'     : 1.0,
                      'r_larm'     : 1.0,
                      'l_larm'     : 1.0,
                      'r_hand'     : 1.0,
                      'l_hand'     : 1.0,
                      'pelvis'     : 1.0,
                      'r_thigh'    : 1.0,
                      'l_thigh'    : 1.0,
                      'r_shank'    : 1.0,
                      'l_shank'    : 1.0,
                      'r_foot'     : 1.0,
                      'l_foot'     : 1.0,
                      'r_toes'     : 1.0,
                      'l_toes'     : 1.0}

# Define the weights for the orientation tracking. The global orientation weight
# is multiplied by all the orientation weight values which is useful when
# weighting origins and orientations.
orientWeightValues = {}
orientWeightGlobal = 1.0
orientWeightValues = {'head'       : 1.0,
                      'torso'      : 1.0,
                      'r_clavicle' : 1.0,
                      'l_clavicle' : 1.0,
                      'r_uarm'     : 1.0,
                      'l_uarm'     : 1.0,
                      'r_larm'     : 1.0,
                      'l_larm'     : 1.0,
                      'r_hand'     : 1.0,
                      'l_hand'     : 1.0,
                      'pelvis'     : 1.0,
                      'r_thigh'    : 1.0,
                      'l_thigh'    : 1.0,
                      'r_shank'    : 1.0,
                      'l_shank'    : 1.0,
                      'r_foot'     : 1.0,
                      'l_foot'     : 1.0,
                      'r_toes'     : 1.0,
                      'l_toes'     : 1.0}

# Check that the final results match the orientation and origin .pkl files.
# Note that this check can throw warnings if:
#   1. The .trc file generated contains static origin information (only
#      orientations will be correct)
#   2. IK is run with origins only (only origins will be correct)
checkResults = True

# Set the accuracy for the result checking.
TOL_POS = 10e-5 # For position vector entries
TOL_ROT = 10e-5 # For rotation matrix entries

# Set the accuracy for InverseKinematicsSolver.
TOL_IK = 10e-8

# ==========================  D E C L A R A T I O N  ===========================

# Method to compare two lists, make sure they're of the same size, and have the
# same elements. If not, a warning is printed.
def compareLists(listA,listB,nameA='listA',nameB='listB'):
    sizeA = len(listA)
    sizeB = len(listB)
    if (sizeA != sizeB):
        print(f"WARNING: List '{nameA}' (size {sizeA}) and list '{nameB}' "
              f"(size {sizeB}) not of same length.")
    for item in listA:
        if (item not in listB):
            print(f"WARNING: Item '{item}' from list '{nameA}' not in list "
                  f"'{nameB}'.")
    for item in listB:
        if (item not in listA):
            print(f"WARNING: Item '{item}' from list '{nameB}' not in list "
                  f"'{nameA}'.")

# Extension to convert an osim Rotation to a numpy array.
def convertRotationToArray(self):
    identityRotation = osim.Rotation()
    rotationNoInv = identityRotation.multiply(self)
    npRotation = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            npRotation[i][j] = rotationNoInv.get(i,j)
    return npRotation
osim.Rotation.to_numpy = convertRotationToArray
osim.InverseRotation.to_numpy = convertRotationToArray

# Read motion file and store in a DataFrame.
def readMotFile(aFilename, aPrintStatus = False):
    
    # Read file header and column headers.
    numHeadRows = 1
    numDataRows = 0
    numDataCols = 0
    inDegrees = -1
    foundEndHeader = False
    
    f = open(aFilename, 'r')
    
    while f:
        line = f.readline()
        if ('nRows' in line):
            numDataRows = int( line[line.index('=')+1:] )
        if ('nColumns' in line):
            numDataCols = int( line[line.index('=')+1:] )
        if ('inDegrees' in line):
            if '=yes' in line:
                inDegrees = True
            elif '=no' in line:
                inDegrees = False
        if ('endheader' in line):
            foundEndHeader = True
            continue
        if (not foundEndHeader):
            numHeadRows += 1
        else:
            line = line.strip()  #remove trailing line break
            colHeaders = line.split('\t')
            break
    
    f.close()
    
    # Report file info (for debugging).
    if (aPrintStatus):
        print('** Header rows detected: {}'.format(numHeadRows))
        print('** Data table size: {} rows, {} columns'
              .format(numDataRows, numDataCols))
        print('** Columns:')
        for i in range(numDataCols):
            print('   [{}]: {}'.format(i, colHeaders[i]))
    
    # Read data into DataFrame.
    df = pandas.read_csv(aFilename, delimiter='\t', skiprows=numHeadRows)
    if ((df.shape[0] != numDataRows) or (df.shape[1] != numDataCols)):
        raise Exception('DataFrame dimensions inconsistent with file header')
    
    # Make sure inDegrees is defined, assign it to df metadata.
    if inDegrees == -1:
        raise Exception('Unknown if file is in degrees or radians')
    else:
        df.inDegrees = inDegrees
    
    return df

class MotFileGenerator:
    def __init__(self,
                 fileName   : str,
                 nTimeSteps : int,
                 model      : osim.Model,
                 title = ''):
        
        self.coordFile = open(fileName, 'w')
        self.model = model
        nCoords = self.model.getNumCoordinates()
        coordSet = self.model.getCoordinateSet()
        
        if title == '':
            title = os.path.basename(fileName)
        
        self.coordFile.write(
                        os.path.splitext(title)[0]+'\n'
                         'version=1\n'
                        f'nRows={nTimeSteps}\n'
                        f'nColumns={nCoords+1}\n'
                         'inDegrees=yes\n'
                         '\n'
                         'Units are S.I. units (second, meters, Newtons, ...)\n'
                         "If the header above contains a line with 'inDegrees',"
                         ' this indicates whether rotational values are in'
                         ' degrees (yes) or radians (no).\n'
                         '\n'
                         'endheader\n')
        
        # Write column labels for the IK results.
        string = 'time\t'
        for coordIndex in range(nCoords):
            coord = coordSet.get(coordIndex)
            coordName = coord.getName()
            stringEnd = '\t' if (coordIndex < nCoords-1) else '\n'
            string = string + coordName + stringEnd
        self.coordFile.write(string)
        
    def writeTimeStep(self,
                      state : osim.State):
        self.model.realizePosition(state)
        nCoords = self.model.getNumCoordinates()
        coordSet = self.model.getCoordinateSet()
        time = state.getTime()
        timeFormatted = '{:.8f}'.format(time)
        string = f'      {timeFormatted}\t     '
        for coordIndex in range(nCoords):
            coord = coordSet.get(coordIndex)
            coordValue = coord.getValue(state)
            if (coord.getMotionType()==1):
                    coordValue = coordValue * (180 / np.pi)
            coordValueFormatted = '{:.8f}'.format(coordValue)
            stringEnd = '\t      ' if (coordIndex < nCoords - 1) else '\n'
            string = string + str(coordValueFormatted) + stringEnd
        self.coordFile.write(string)
    
    def close(self):
        self.coordFile.close()


# ===============================  S C R I P T  ================================

# Load the TLO model.
tloModel = osim.Model(modelFileName)
kinematicsReporter = osim.Kinematics()
kinematicsReporter.setRecordAccelerations(False)
kinematicsReporter.setInDegrees(True)
tloModel.addAnalysis(kinematicsReporter)

# Obtain sets from the TLO model and their sizes.
bodySetModel   = tloModel.getBodySet()
nBodiesModel   = tloModel.getNumBodies()
markerSetModel = tloModel.getMarkerSet()
nMarkersModel  = tloModel.getNumMarkers()
coordSet       = tloModel.getCoordinateSet()
nCoords        = tloModel.getNumCoordinates()

# Assign the body names from the TLO model to a Python list.
bodyNamesModel = [''] * nBodiesModel
for bodyIndex in range(nBodiesModel):
    bodyName = bodySetModel.get(bodyIndex).getName()
    bodyNamesModel[bodyIndex] = bodyName

# ---------------------------------- ORIGINS --------------------------------- #
# Populate origin table.
originTable = osim.TimeSeriesTableVec3(originsFileNameTRC)

originFile = open(originsFileNamePKL,'rb')
originDict = pickle.load(originFile)

# Create body/marker name lists.
nBodiesOrigin = originTable.getNumColumns()
markerNamesOrigin = [''] * nBodiesOrigin
bodyNamesOrigin   = [''] * nBodiesOrigin
for bodyIndex in range(nBodiesOrigin):
    # NOTE: the trc file contains 'marker' information, so the markers have
    # suffixes (_m) indicating that they are markers.
    markerName = originTable.getColumnLabel(bodyIndex)
    bodyName = markerName.replace('_m','')
    markerNamesOrigin[bodyIndex] = markerName
    bodyNamesOrigin[bodyIndex] = bodyName

# Compare the body name lists between the model and the origin .trc file.
compareLists( bodyNamesOrigin,   bodyNamesModel,
             'bodyNamesOrigin', 'bodyNamesModel')

# Assign the origin weights to a set.
smw = osim.SetMarkerWeights()

for bodyIndex, bodyName in enumerate(bodyNamesOrigin):
    mw = osim.MarkerWeight(markerNamesOrigin[bodyIndex], 
                           originWeightGlobal*originWeightValues[bodyName])
    smw.cloneAndAppend(mw)

originsRef = osim.MarkersReference(originTable, smw)

# Declare ikSolver for the origin mode.
if not (includeOrientations):
    coordRefArray = osim.SimTKArrayCoordinateReference()
    ikSolver = osim.InverseKinematicsSolver(tloModel,
                                            originsRef,
                                            coordRefArray)
# ---------------------------------------------------------------------------- #

# ------------------------------- ORIENTATIONS ------------------------------- #
if (includeOrientations):
    # Load the orientation data.
    orientFile = open(orientsFileNamePKL,'rb')
    orientDict = pickle.load(orientFile)
    
    # Obtain the list of bodies from the orientation dict and the number of 
    # bodies in it.
    bodyNamesOrient = list(orientDict[list(orientDict.keys())[0]].keys())
    nBodiesOrient = len(bodyNamesOrient)
    
    # Set up the orientation table.
    orientTable = osim.TimeSeriesTableRotation()
    orientTable.setColumnLabels(bodyNamesOrient)
    
    # Populate the orientation table.
    emptyRotation = osim.Rotation()
    for timeSec in orientDict:
        # Get the dict of body orientations for the current time.
        dataLine = orientDict[timeSec]
        
        # Create a rotation row vector.
        rowVectorRotation = osim.RowVectorRotation(nBodiesOrient, emptyRotation)
        
        # Loop through the bodies in this time-step.
        for index, bodyName in enumerate(bodyNamesOrient):
            rotationArray = dataLine[bodyName]
            rotationMatrix = rowVectorRotation.updElt(0, index)
            for row in range(3):
                for col in range(3):
                    rotationMatrix.set(row, col, rotationArray[row][col])
        
        # Append the row vector to the orientation table.
        orientTable.appendRow(timeSec, rowVectorRotation)
    
    # Create body name list.
    nBodiesOrient = orientTable.getNumColumns()
    bodyNamesOrient   = [''] * nBodiesOrient
    for bodyIndex in range(nBodiesOrient):
        bodyName = orientTable.getColumnLabel(bodyIndex)
        bodyNamesOrient[bodyIndex] = bodyName
    
    compareLists( bodyNamesOrient,   bodyNamesModel,
                 'bodyNamesOrient', 'bodyNamesModel')
    
    # Assign the weights to a set.
    # NOTE: typo in the OpenSim API
    sow = osim.SetOientationWeights()
    for bodyName in bodyNamesOrient:
        ow = osim.OrientationWeight(bodyName,
                            orientWeightGlobal*orientWeightValues[bodyName])
        sow.cloneAndAppend(ow)
    
    orientsRef = osim.OrientationsReference(orientTable, sow)
    
    # Declare ikSolver for the fused mode.
    coordRefArray = osim.SimTKArrayCoordinateReference()
    ikSolver = osim.InverseKinematicsSolver(tloModel,
                                            originsRef,
                                            orientsRef,
                                            coordRefArray)
# ---------------------------------------------------------------------------- #




# Initialize ikSolver.
ikSolver.setAccuracy(TOL_IK)
state = tloModel.initSystem()
state.setTime(list(originDict.keys())[0])
ikSolver.assemble(state)
kinematicsReporter.begin(state)

# Check that the number of time-steps is the same for origins and orientations.
if not (includeOrientations):
    nTimeSteps = originTable.getNumRows()
else:
    if (orientTable.getNumRows() != originTable.getNumRows()):
        print(f'WARNING: Orientation (rows:{orientTable.getNumRows()}) and '
              f'origin (rows:{originTable.getNumRows()}) files do not have the '
               'same number of rows.', flush=True)
    nTimeSteps = max([orientTable.getNumRows(), originTable.getNumRows()])

tloMFG = MotFileGenerator(fileName=outputCoordFileName,
                          nTimeSteps=nTimeSteps,
                          model = tloModel)

# Write column labels for the marker locations.
locations = np.zeros((nTimeSteps,nMarkersModel*3))
nMarkersIK = ikSolver.getNumMarkersInUse()
locationsFile = open(outputLocationsFileName, 'w')
xyz = ['x','y','z']
for markerIndex in range(nMarkersModel):
    name = markerSetModel.get(markerIndex).getName()
    name = name.replace('_m','')
    for coord in range(3):
        locationsFile.write(f'{name}_{xyz[coord]},')
locationsFile.write('\n')

# Track coordinates using ikSolver.
for timeStep, timeSec in enumerate(originDict):
    state.setTime(timeSec)
    ikSolver.track(state)
    tloModel.realizePosition(state)
    
    tloMFG.writeTimeStep(state)
    
    # Write marker locations.
    for markerIndex in range(nMarkersModel):
        marker = markerSetModel.get(markerIndex)
        markerLocation = marker.getLocationInGround(state)
        for k in range(3):
            locationsFile.write(f'{markerLocation.get(k)},')
    locationsFile.write('\n')
    
    kinematicsReporter.step(state,timeStep)

tloMFG.close()
locationsFile.close()

# Check that the coordinate data match the orientation and origin data.
ikResults = readMotFile(outputCoordFileName)
if (checkResults):
    for timeStep, timeSec in enumerate(originDict):
        # state.setTime(timeSec)
        
        # Set the tlo model coordinates at this time-step according to the IK
        # results.
        for c in range(nCoords):
            coord = coordSet.get(c)
            coordName = coord.getName()
            jointName = coord.getJoint().getName()
            coordNameFull = '/jointset/'+jointName+'/'+coordName+'/value'
            coordVal = ikResults.at[timeStep,coordName]
            # Convert rotational coordinates from degrees to radians.
            if (coord.getMotionType()==1):
                coordVal = coordVal * (np.pi/180)
            tloModel.setStateVariableValue(state, coordNameFull, coordVal)
        tloModel.realizePosition(state)
        
        # Check the position and orientation of each body.
        for b in range(nBodiesModel):
            body = bodySetModel.get(b)
            bodyName = body.getName()
            
            posFromFile = originDict[timeSec][bodyName]
            posFromIK = body.getPositionInGround(state)
            if not (np.allclose(posFromFile, posFromIK.to_numpy(),
                                atol=TOL_POS, rtol=0)):
                print(f'WARNING: Incorrect IK solution at time {timeSec} '
                      f'due to the orientation of {bodyName}.\n'
                      f'Position from PKL:\n{posFromFile}\n'
                      f'Position from IK solution:\n{posFromIK.to_numpy()}')
            
            if (includeOrientations):
                rotFromFile = orientDict[timeSec][bodyName]
                rotFromIK = body.getRotationInGround(state)
                if not (np.allclose(rotFromFile, rotFromIK.to_numpy(),
                                    atol=TOL_ROT, rtol=0)):
                    print(f'WARNING: Incorrect IK solution at time {timeSec} '
                          f'due to the orientation of {bodyName}.\n'
                          f'Rotation from PKL:\n{rotFromFile}\n'
                          f'Rotation from IK solution:\n{rotFromIK.to_numpy()}')

print(f'Done!\nMotion saved to: {outputCoordFileName}')
