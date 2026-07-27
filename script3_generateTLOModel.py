# ---------------------------------------------------------------------------- #
#                         script3_generateTLOModel.py                          #
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

# This script generates an OpenSim musculoskeletal model that is designed to
# imitate Theia3D's proprietary model; we call this model the "Theia-like
# OpenSim" (TLO) model. The TLO model is an intermediary model that allows us to
# visualize and verify the data according to Theia's Default Model in their
# documentation
# (https://docs.theiamarkerless.com/apollo-v2024/theia3d-documentation/theia-model-description).
# Note that this script generates the TLO model according to the Theia3D data at
# a specified time step. The state of the model at this time step is the TLO
# model's default pose.
#
# This script was built using OpenSim 4.5 and Python 3.11 and tested with data
# from Theia3D 2022.2.
#
# version 1.0

import numpy as np
import opensim as osim
import pickle
import os.path


# ===========================  P A R A M E T E R S  ============================

# File names of the origin and orientation files created by the parser (*.pkl).
originPKLFileName = 'Output/theiaOrigins.pkl'
orientPKLFileName = 'Output/theiaOrientations.pkl'

# Output file name of the TLO model (*.osim).
outputModelFileName = 'Output/tloModel.osim'

# The frame number (in base-0) that captures the default pose of the model.
# NOTE: this time-step is relative to the start frame defined in script1.
definitionFrame = 0

# Flag to enable custom ankle coordinate names.
customAnkleCoordNames = True
ankleCoordNames = {'ankle_angle_r'    : 'ankle_dorsiflexion_r',
                   'subtalar_angle_r' : 'ankle_inversion_r',
                   'ankle_angle_l'    : 'ankle_dorsiflexion_l',
                   'subtalar_angle_l' : 'ankle_inversion_l'}


# ==========================  D E C L A R A T I O N  ===========================

# This class stores transformation matrices of bodies in a binary tree. This
# structure allows for the automation of obtaining the transform of each body
# relative to its parent (Theia's transforms are given relative to ground but
# joint construction requires transforms relative to the parent body).
class BinaryTreeTransform:
    
    # The default constructor.
    def __init__(self, name, transformMatrix):
        # Name of the body this transformation matrix is attached to.
        self.name = name
        # Transformation matrix (as a numpy Array) in the world frame.
        self.transformInWorld = transformMatrix
        # Transformation matrix in the body's parent frame.
        # NOTE: transformInParent is initialized to the transform in the world.
        # frame because this variable is unmodified for frames connected to
        # ground (the root nodes).
        self.transformInParent = transformMatrix
        # Left child of this transform.
        self.lChild = None
        # Right child of this transform.
        self.rChild = None
    
    # Method to obtain the rotation matrix of the current frame in its parent
    # (as a 3x3 numpy Array).
    def getRotationInParent(self):
        return self.transformInParent[:3, :3]
    
    # Method to obtain the translation vector of the current frame in its parent
    # (as a 3x1 numpy Array).
    def getTranslationInParent(self):
        return self.transformInParent[:3, 3]
    
    # Method to obtain the inverse of the transformation matrix.
    def getInverseInWorld(self):
        # temp = self.transformInWorld.copy()
        osimTransform = convertArrayToTransform(self.transformInWorld)
        osimTransformInv = osimTransform.invert()
        inverseFromOsim = convertTransformToArray(osimTransformInv)
        inverseFromNumpy = np.linalg.inv(self.transformInWorld)
        # NOTE: the tolerance for error checking was set experimentally.
        if not (np.allclose(inverseFromOsim,inverseFromNumpy,rtol=0,atol=1e-4)):
            print(f'WARNING: NUMPY AND SIMBODY INVERSES DO NOT AGREE FOR BODY:\
                    {self.name}.\nThe error is:\n\
                    {np.subtract(inverseFromOsim, inverseFromNumpy)}\n\
                    Osim inverse:\n{inverseFromOsim}\n\
                    Numpy inverse:\n{inverseFromNumpy}')
        return (inverseFromOsim)

# A method to finalize the transformation matrix tree structure by multiplying
# the necessary matrices to obtain the transformation matrix of each child in
# its parent frame. Theia gives us a transformation matrix in the world frame,
# T_w_c, where c denotes the child frame this transformation represents and w
# represents its reference, the world frame. Likewise, parent frames are
# represented by the transformation, T_w_p. To construct the joints of the
# model, the transformation matrix of the child in reference to the parent,
# T_p_c, is obtained by: T_p_c = inv(T_w_p) * T_w_c
def finalizeTree(myNode):
    if type(myNode.rChild) == BinaryTreeTransform:
        myNode.rChild.transformInParent = np.matmul(myNode.getInverseInWorld(),
                                                 myNode.rChild.transformInWorld)
        finalizeTree(myNode.rChild)
    if type(myNode.lChild) == BinaryTreeTransform:
        myNode.lChild.transformInParent = np.matmul(myNode.getInverseInWorld(),
                                                 myNode.lChild.transformInWorld)
        finalizeTree(myNode.lChild)

# Convert a 3x3 numpy Array (rotation matrix) to an osim Vec3 of body-fixed
# Euler angles in radians for rotations in XYZ order.
def convertArrayToBodyFixedXYZ(rotationArray):
    bodyFixedXYZ = osim.Vec3()
    myRotation   = osim.Rotation()
    myMatrix     = osim.Mat33()
    for i in range(3):
        for j in range(3):
            rotationElement = float(rotationArray[i][j])
            myMatrix.set(i,j,rotationElement)
    myRotation.setRotationFromApproximateMat33(myMatrix)
    bodyFixedXYZ = myRotation.convertRotationToBodyFixedXYZ()
    return bodyFixedXYZ

# Convert a 4x4 numpy Array to an osim Transform.
def convertArrayToTransform(transformArray):
    translation = osim.Vec3()
    rotation = osim.Rotation()
    for i in range(3):
        for j in range(3):
            rotationElement = float(transformArray[i][j])
            rotation.set(i, j, rotationElement)
    for i in range(3):
        translation.set(i, transformArray[i][3])
    transform = osim.Transform(rotation, translation)
    return transform

# Converts an osim Transform (or InverseTransform) to a numpy Array.
# NOTE: I don't extend the Transform class because I'd want to extend the 
# InverseTransform class as well but it's inaccessible.
def convertTransformToArray(transform):#.setToZero()
    identityTransform = osim.Transform()
    # NOTE: this part is a workaround (it's the only way that works in Python).
    transformNoInv = identityTransform.compose(transform)
    rotation = transformNoInv.R()
    translation = transformNoInv.p()
    transformArray = np.zeros([4,4])
    for i in range(3):
        for j in range(3):
            transformArray[i][j] = rotation.get(i, j)
    for i in range(3):
        transformArray[i][3] = translation.get(i)
    transformArray[3][3] = 1.0
    # NOTE: transformArray is initialized to zeros, no need to set
    # transformArray[:3][3] to 0.
    return transformArray

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


# ===============================  S C R I P T  ================================

# Define a vector and inertia tensor of zeroes for efficiency.
VEC3_ZERO = osim.Vec3(0)
INERTIA_ZERO = osim.Inertia(0)

# Load the model.
model = osim.Model()
MODEL_GROUND = model.getGround()

# Load the dictionaries (dicts) from the origin and orientation .pkl files
# generated by the parser script.
originFile = open(originPKLFileName, 'rb')
orientFile = open(orientPKLFileName, 'rb')
originDict = pickle.load(originFile)
orientDict = pickle.load(orientFile)
originFile.close()
orientFile.close()

# Grab the origin positions at the definition time-step from the dict.
originDefinition = originDict[list(originDict.keys())[definitionFrame]]
orientDefinition = orientDict[list(originDict.keys())[definitionFrame]]
bodyNameList = list(originDefinition.keys())
if bodyNameList != list(orientDefinition.keys()):
    print('WARNING: origin and orientation body names do not agree (they may '
          'not be in the same order).')

# Initialize dicts.
bodyDict              = {}
eulerAngleDict        = {}
translationDict       = {}
definitionTransforms  = {}
jointDict             = {}
jointTypeDict         = {}

# Construct the bodies.
for bodyName in bodyNameList:
    bodyDict[bodyName] = osim.Body(bodyName,
                                   1.0,
                                   VEC3_ZERO,
                                   INERTIA_ZERO)

# Load body positions and orientations. This places the bodies at the origin of
# each joint.
for bodyName in bodyNameList:
    # Initialize the array that stores the transformation matrix before it is
    # stored in the definitionTransforms dict.
    bodyTransform = np.zeros([4,4])
    
    # The order of the 16 entries of a transformation matrix from Theia3D is
    # in row-major. The entries in the file are ordered: a11, a12, a13, a14,
    # a21, a23, a24, a31, etc., where a12, for example, corresponds to the entry
    # on the first row and second column. Therefore, for a transformation matrix
    # the labels are:
    # Transformation matrix labels = a11 a12 a13 a14
    #                                a21 a22 a23 a24
    #                                a31 a32 a33 a34
    #                                a41 a42 a43 a44
    
    # Add the elements of the rotation matrix and translation vector to a single
    # transformation matrix. This is done to perform the necessary
    # transformations on both origins and orientations at once.
    bodyTransform[3,3] = 1
    bodyTransform[:3,:3] = orientDefinition[bodyName]
    bodyTransform[:3,3] = originDefinition[bodyName]
    # NOTE: bodyTransform is initialized to zeros, no need to set
    # bodyTransform[:3][3] to 0.
    
    # Assign the transformations to a dict of binary tree nodes storing the
    # transformation matrices (as np arrays) of each segment's state at the
    # definition pose.
    definitionTransforms[bodyName] = BinaryTreeTransform(bodyName,
                                                         bodyTransform)

# Setting up the binary tree structure, define the child bodies for each parent.
# NOTE: the head has no child bodies.
definitionTransforms['torso'].rChild      = definitionTransforms['r_clavicle']
definitionTransforms['torso'].lChild      = definitionTransforms['l_clavicle']
definitionTransforms['r_clavicle'].rChild = definitionTransforms['r_uarm']
definitionTransforms['l_clavicle'].lChild = definitionTransforms['l_uarm']
definitionTransforms['r_uarm'].rChild     = definitionTransforms['r_larm']
definitionTransforms['l_uarm'].lChild     = definitionTransforms['l_larm']
definitionTransforms['r_larm'].rChild     = definitionTransforms['r_hand']
definitionTransforms['l_larm'].lChild     = definitionTransforms['l_hand']
definitionTransforms['pelvis'].rChild     = definitionTransforms['r_thigh']
definitionTransforms['pelvis'].lChild     = definitionTransforms['l_thigh']
definitionTransforms['r_thigh'].rChild    = definitionTransforms['r_shank']
definitionTransforms['l_thigh'].lChild    = definitionTransforms['l_shank']
definitionTransforms['r_shank'].rChild    = definitionTransforms['r_foot']
definitionTransforms['l_shank'].lChild    = definitionTransforms['l_foot']
definitionTransforms['r_foot'].rChild     = definitionTransforms['r_toes']
definitionTransforms['l_foot'].lChild     = definitionTransforms['l_toes']

# Multiply transformation matrices of each body such that they are all in terms
# of their proximal connecting body frames (their parent frame).
finalizeTree(definitionTransforms['torso'])
finalizeTree(definitionTransforms['pelvis'])

# Convert each rotation to a Vec3 of body-fixed Euler angles in XYZ order and
# each translation array to a Vec3 of translations.
for bodyName in bodyNameList:
    eulerAngleDict[bodyName] = convertArrayToBodyFixedXYZ(\
                           definitionTransforms[bodyName].getRotationInParent())
    translationDict[bodyName] = osim.Vec3(\
                       *definitionTransforms[bodyName].getTranslationInParent())

# Initialize the root joints (head, torso, and pelvis).
for root in ['head','torso','pelvis']:
    jointDict[root] = osim.FreeJoint(root+'-ground',
                                     MODEL_GROUND,
                                     translationDict[root],
                                     eulerAngleDict[root],
                                     bodyDict[root],
                                     VEC3_ZERO,
                                     VEC3_ZERO)

# Define the joint type for each body according to the Theia documentation.
jointTypeDict['r_clavicle'] = osim.BallJoint # this should be a UniversalJoint
jointTypeDict['l_clavicle'] = osim.BallJoint # this should be a UniversalJoint
jointTypeDict['r_uarm']     = osim.FreeJoint # this could be a BallJoint
jointTypeDict['l_uarm']     = osim.FreeJoint # this could be a BallJoint
jointTypeDict['r_larm']     = osim.BallJoint # this should be a UniversalJoint
jointTypeDict['l_larm']     = osim.BallJoint # this should be a UniversalJoint
jointTypeDict['r_hand']     = osim.BallJoint # this should be a UniversalJoint
jointTypeDict['l_hand']     = osim.BallJoint # this should be a UniversalJoint
jointTypeDict['r_thigh']    = osim.BallJoint
jointTypeDict['l_thigh']    = osim.BallJoint
jointTypeDict['r_shank']    = osim.BallJoint # this could be a UniversalJoint
jointTypeDict['l_shank']    = osim.BallJoint # this could be a UniversalJoint
jointTypeDict['r_foot']     = osim.FreeJoint # this could be a BallJoint
jointTypeDict['l_foot']     = osim.FreeJoint # this could be a BallJoint
jointTypeDict['r_toes']     = osim.BallJoint # this should be a PinJoint
jointTypeDict['l_toes']     = osim.BallJoint # this should be a PinJoint

# Construct the rest of the joints.
for parent in bodyNameList:
    rChild = definitionTransforms[parent].rChild
    lChild = definitionTransforms[parent].lChild
    if (type(rChild) == BinaryTreeTransform):
        name = rChild.name
        jointDict[name] = jointTypeDict[name](name+'-'+parent,
                                              bodyDict[parent],
                                              translationDict[name],
                                              eulerAngleDict[name],
                                              bodyDict[name],
                                              VEC3_ZERO,
                                              VEC3_ZERO)
    if (type(lChild) == BinaryTreeTransform):
        name = lChild.name
        jointDict[name] = jointTypeDict[name](name+'-'+parent,
                                              bodyDict[parent],
                                              translationDict[name],
                                              eulerAngleDict[name],
                                              bodyDict[name],
                                              VEC3_ZERO,
                                              VEC3_ZERO)

# Add bodies and joints to the model.
for bodyName in bodyNameList:
    if bodyName == 'pelvis_shifted':
        continue
    model.addBody(bodyDict[bodyName])
    model.addJoint(jointDict[bodyName])

# Write the default coordinate names to a dict.
coordSet = model.getCoordinateSet()
nCoords = model.getNumCoordinates()
coordNameDict = {}
for c in range(nCoords):
    coordNameDict[coordSet.get(c).getName()] = coordSet.get(c).getName()

# Modify coordinate names to fit Rajagopal as much as possible; discrepancies
# are addressed in mapping.
coordNameDict['l_foot-l_shank_coord_0']    = 'ankle_angle_l'
coordNameDict['l_foot-l_shank_coord_2']    = 'subtalar_angle_l'
coordNameDict['l_hand-l_larm_coord_0']     = 'wrist_flex_l'
coordNameDict['l_hand-l_larm_coord_1']     = 'wrist_dev_l'
coordNameDict['l_larm-l_uarm_coord_0']     = 'elbow_flex_l'
coordNameDict['l_larm-l_uarm_coord_2']     = 'pro_sup_l'
coordNameDict['l_shank-l_thigh_coord_0']   = 'knee_angle_l'
coordNameDict['l_shank-l_thigh_coord_1']   = 'knee_abduction_l'
coordNameDict['l_shank-l_thigh_coord_2']   = 'knee_extRotation_l'
coordNameDict['l_thigh-pelvis_coord_0']    = 'hip_flexion_l'
coordNameDict['l_thigh-pelvis_coord_1']    = 'hip_adduction_l'
coordNameDict['l_thigh-pelvis_coord_2']    = 'hip_rotation_l'
coordNameDict['l_toes-l_foot_coord_0']     = 'mtp_angle_l'
coordNameDict['l_uarm-l_clavicle_coord_0'] = 'arm_flex_l'
coordNameDict['l_uarm-l_clavicle_coord_1'] = 'arm_add_l'
coordNameDict['l_uarm-l_clavicle_coord_2'] = 'arm_rot_l'
coordNameDict['r_foot-r_shank_coord_0']    = 'ankle_angle_r'
coordNameDict['r_foot-r_shank_coord_2']    = 'subtalar_angle_r'
coordNameDict['r_hand-r_larm_coord_0']     = 'wrist_flex_r'
coordNameDict['r_hand-r_larm_coord_1']     = 'wrist_dev_r'
coordNameDict['r_larm-r_uarm_coord_0']     = 'elbow_flex_r'
coordNameDict['r_larm-r_uarm_coord_2']     = 'pro_sup_r'
coordNameDict['r_shank-r_thigh_coord_0']   = 'knee_angle_r'
coordNameDict['r_shank-r_thigh_coord_1']   = 'knee_abduction_r'
coordNameDict['r_shank-r_thigh_coord_2']   = 'knee_intRotation_r'
coordNameDict['r_thigh-pelvis_coord_0']    = 'hip_flexion_r'
coordNameDict['r_thigh-pelvis_coord_1']    = 'hip_adduction_r'
coordNameDict['r_thigh-pelvis_coord_2']    = 'hip_rotation_r'
coordNameDict['r_toes-r_foot_coord_0']     = 'mtp_angle_r'
coordNameDict['r_uarm-r_clavicle_coord_0'] = 'arm_flex_r'
coordNameDict['r_uarm-r_clavicle_coord_1'] = 'arm_add_r'
coordNameDict['r_uarm-r_clavicle_coord_2'] = 'arm_rot_r'
coordNameDict['pelvis-ground_coord_0']     = 'pelvis_tilt'
coordNameDict['pelvis-ground_coord_1']     = 'pelvis_list'
coordNameDict['pelvis-ground_coord_2']     = 'pelvis_rotation'
coordNameDict['pelvis-ground_coord_3']     = 'pelvis_tz'
coordNameDict['pelvis-ground_coord_4']     = 'pelvis_tx'
coordNameDict['pelvis-ground_coord_5']     = 'pelvis_ty'
coordNameDict['torso-ground_coord_0']      = 'lumbar_extension'
coordNameDict['torso-ground_coord_1']      = 'lumbar_bending'
coordNameDict['torso-ground_coord_2']      = 'lumbar_rotation'
if customAnkleCoordNames:
    coordNameDict['r_foot-r_shank_coord_0'] = ankleCoordNames['ankle_angle_r']
    coordNameDict['r_foot-r_shank_coord_2'] = ankleCoordNames['subtalar_angle_r']
    coordNameDict['l_foot-l_shank_coord_0'] = ankleCoordNames['ankle_angle_l']
    coordNameDict['l_foot-l_shank_coord_2'] = ankleCoordNames['subtalar_angle_l']

for c in range(nCoords):
    coord = coordSet.get(c)
    # print(f"coordNameDict['{coord.getName()}'] = '{coordNameDict[coord.getName()]}'")
    coord.setName(coordNameDict[coord.getName()])

# Add markers to the model. Loop over the bodies and add a marker to each (at
# the body frame origin location, as per Theia).
bodySet = model.getBodySet()
for i in range(bodySet.getSize()):
    body = bodySet.get(i)
    bodyName = body.getName()
    # Add a suffix to the marker name such that it is different to the body name
    # (this is also done in script2_createTRCFile).
    markerName = bodyName + '_m'
    marker = osim.Marker(markerName, body, VEC3_ZERO)
    model.addMarker(marker)

# Check that the bodies of the model are in the correct positions and
# orientations according to the input data.
state = model.initSystem()
nb = model.getNumBodies()
iRot = osim.Rotation()
for b in range(nb):
    body = bodySet.get(b)
    bodyName = body.getName()
    
    rot = body.getRotationInGround(state)
    pos = body.getPositionInGround(state)
    
    if not np.allclose(originDefinition[bodyName], pos.to_numpy(),
                       rtol=0,atol=2e-5):
        print(f'WARNING: {bodyName} not in correct position.\n'
              f'Translation from PKL file:\n{originDefinition[bodyName]}\n'
              f'Translation from TLO model:\n{pos.to_numpy()}')
    if not np.allclose(orientDefinition[bodyName], rot.to_numpy(),
                       rtol=0,atol=2e-5):
        print(f'WARNING: {bodyName} not in correct orientation.\n'
              f'Rotation from PKL file:\n{orientDefinition[bodyName]}\n'
              f'Rotation from TLO model:\n{rot.to_numpy()}')

# Finalize.
modelBaseName = os.path.basename(outputModelFileName)
modelName = os.path.splitext(modelBaseName)[0]
model.setName(modelName)
model.initSystem()
model.printToXML(outputModelFileName)
print(f'Done!\nTLO model created: {outputModelFileName}')
