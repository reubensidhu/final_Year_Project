#CLASS TAKES FRAME FROM GUI THEN GIVES FRAME TO YOLO AND GETCORNERS CLASS
#CLASS THEN USES CORNER AND BALL CENTERS TO COMPUTE PROJECTION USING TABLE CREATION CLASS
#CAN just use theight and twidth in this class then they can be added to borders in tablecreation

#declare model when gui app launches then feed that model here for the first frame.
#then each time button is pressed use that same model to compute results and draw ball
import numpy as np
import cv2 as cv
from getCorners import getCorners
#from yoloModel import PoolBallDetection

#class ProjectionCalculator:
#  def __init__(self, points3d, points2d):
#    if not (points3d and points2d):
#       raise Exception('Two arrays with points must be provided. ')
#    if (len(points3d) != len(points2d)):
#      raise Exception('Lengths of point arrays must be equal. ')
#    self.points3d = points3d
#    self.points2d = points2d

class ProjectionCalculator3d():
  def __init__(self, frame, centers):
        self.frame = frame
        self.centers = centers
        self.getPoints()

  def getPoints(self):
    c = getCorners(self.frame)
    corners = c()
    print('corners', corners)
    #labels, coords = self.model.score_frame(self.frame)
    #coords = coords.numpy()
    centers = self.centers
    #for i in range(0, 2):
    #  centers.append([int(self.frame.shape[1] * ((self.coords[i][0]+self.coords[i][2])/2)), 
    #  int(self.frame.shape[0] * ((self.coords[i][1]+self.coords[i][3])/2))])
    if c.isShortSide and (centers[0][1] > centers[1][1]):
      centers[0], centers[1] = centers[1], centers[0]
    elif (not c.isShortSide) and (centers[0][0] > centers[1][0]):
      centers[0], centers[1] = centers[1], centers[0]
    else:
      pass
    print('corners', corners, 'center', centers)
    self.points2d = np.concatenate((corners, centers))
    print('2d points', self.points2d)
    self.points3d = [[0, 0, 0], [865, 0, 0], [865, 445, 0], [0, 445, 0], [222.5, 222.5, -6], [642.5, 222.5, -6]] # change 5 to -h + r for last 2
    self.calculateMatrix()
  
  def calculateMatrix(self):
    generalMatrix = np.zeros((len(self.points2d) * 2, 12))
    k = 0
    for i in range(0, len(self.points3d)):
      generalMatrix[k,0], generalMatrix[k,1], generalMatrix[k,2] = self.points3d[i][0], self.points3d[i][1], self.points3d[i][2]
      generalMatrix[k,3] = 1
      generalMatrix[k,4] = 0
      generalMatrix[k,5] = 0
      generalMatrix[k,6] = 0
      generalMatrix[k,7] = 0
      generalMatrix[k,8] = -self.points3d[i][0] * self.points2d[i][0]
      generalMatrix[k,9] = -self.points3d[i][1] * self.points2d[i][0]
      generalMatrix[k,10] = -self.points3d[i][2] * self.points2d[i][0]
      generalMatrix[k,11] = -self.points2d[i][0]
      generalMatrix[k + 1,0] = 0
      generalMatrix[k + 1,1] = 0
      generalMatrix[k + 1,2] = 0
      generalMatrix[k + 1,3] = 0
      generalMatrix[k + 1,4], generalMatrix[k + 1,5], generalMatrix[k + 1,6] = self.points3d[i][0], self.points3d[i][1], self.points3d[i][2]
      generalMatrix[k + 1,7] = 1
      generalMatrix[k + 1,8] = -self.points3d[i][0] * self.points2d[i][1]
      generalMatrix[k + 1,9] = -self.points3d[i][1] * self.points2d[i][1]
      generalMatrix[k + 1,10] = -self.points3d[i][2] * self.points2d[i][1]
      generalMatrix[k + 1,11] = -self.points2d[i][1]
      k += 2

    u, s, vh = np.linalg.svd(generalMatrix) #unsure about the vh here
    matrix = vh #matrix = vh.transpose()
    print('shape', matrix.shape)
    subMatrix = matrix[11]
    print('shape', subMatrix)
    #subMatrix = matrix.subMatrix(11, 11, 0, 11)[0]
    self.resultMatrix = np.matrix([
      [subMatrix[0], subMatrix[1], subMatrix[2], subMatrix[3]],
      [subMatrix[4], subMatrix[5], subMatrix[6], subMatrix[7]],
      [subMatrix[8], subMatrix[9], subMatrix[10], subMatrix[11]],
      [0, 0, 0, 1]
    ])
    self.resultMatrixInversed = np.linalg.inv(self.resultMatrix)

  def getUnprojectedPoint(self, point2d, height):
    if height == None:
      raise Exception('Point height must be defined for 3d unprojection')
    point1 = np.matrix([[point2d[0]], [point2d[1]], [1], [1]])
    point2 = np.matrix([[100 * point2d[0]], [100 * point2d[1]], [100], [1]])
    rayPoint1 = np.matmul(self.resultMatrixInversed, point1)
    rayPoint2 = np.matmul(self.resultMatrixInversed, point2)
    result = self.getIntersectionLineAndPlane(
      rayPoint1,
      rayPoint2,
      [0, 0, height],
      [0, 10, height],
      [10, 0, height],
    )
    return result

  def getIntersectionLineAndPlane(self, linePoint1, linePoint2, planePoint1, planePoint2, planePoint3):
    A = (planePoint2[1] - planePoint1[1]) * (planePoint3[2] - planePoint1[2]) - (planePoint2[2] - planePoint1[2]) * (planePoint3[1] - planePoint1[1])
    B = (planePoint2[2] - planePoint1[2]) * (planePoint3[0] - planePoint1[0]) - (planePoint2[0] - planePoint1[0]) * (planePoint3[2] - planePoint1[2])
    C = (planePoint2[0] - planePoint1[0]) * (planePoint3[1] - planePoint1[1]) - (planePoint2[1] - planePoint1[1]) * (planePoint3[0] - planePoint1[0])
    D = A * planePoint1[0] + B * planePoint1[1] + C * planePoint1[2]
    a = linePoint2[0] - linePoint1[0]
    b = linePoint2[1] - linePoint1[1]
    c = linePoint2[2] - linePoint1[2]
    d = a * linePoint1[1] - b * linePoint1[0]
    e = a * linePoint1[2] - c * linePoint1[0]

    intersectionX = (a * D - d * B - e * C) / (a * A + b * B + c * C)
    intersectionY = (b * intersectionX + d) / a
    intersectionZ = (c * intersectionX + e) / a

    return [intersectionX, intersectionY, intersectionZ]

#im = cv.imread(r'C:\Users\reuby\OneDrive\Pictures\2ball.jpeg')
#model = PoolBallDetection()
#c = ProjectionCalculator3d(im, model)
#print(c.getUnprojectedPoint([885, 598], -3.65625))