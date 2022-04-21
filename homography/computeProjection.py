from os import TMP_MAX
import numpy as np
import cv2 as cv
from homography.getCorners import getCorners


class ProjectionCalculator3d():
    def __init__(self, frame, centers):
        self.frame = frame
        self.centers = centers
        self.getPoints()

    def getPoints(self):
        c = getCorners(self.frame)
        corners = c()
        #print('corners', corners)    
        centers = self.centers
        if c.isShortSide and (centers[0][1] > centers[1][1]):
            tmp1 = centers[0]
            tmp2 = centers[1]
            centers = [tmp2, tmp1]
        elif (not c.isShortSide) and (centers[0][0] > centers[1][0]):
            tmp1 = centers[0]
            tmp2 = centers[1]
            centers = [tmp2, tmp1]
        else:
            pass
        self.points2d = np.concatenate((corners, centers))
        self.points3d = [[0, 0, 0], [1730, 0, 0], [1730, 890, 0], [0, 890, 0], [
            445, 445, -12], [1285, 445, -12]] 
        self.calculateMatrix()

    def calculateMatrix(self):
        generalMatrix = np.zeros((len(self.points2d) * 2, 12))
        k = 0
        for i in range(0, len(self.points3d)):
            generalMatrix[k, 0], generalMatrix[k, 1], generalMatrix[k,
                                                                    2] = self.points3d[i][0], self.points3d[i][1], self.points3d[i][2]
            generalMatrix[k, 3] = 1
            generalMatrix[k, 4] = 0
            generalMatrix[k, 5] = 0
            generalMatrix[k, 6] = 0
            generalMatrix[k, 7] = 0
            generalMatrix[k, 8] = -self.points3d[i][0] * self.points2d[i][0]
            generalMatrix[k, 9] = -self.points3d[i][1] * self.points2d[i][0]
            generalMatrix[k, 10] = -self.points3d[i][2] * self.points2d[i][0]
            generalMatrix[k, 11] = -self.points2d[i][0]
            generalMatrix[k + 1, 0] = 0
            generalMatrix[k + 1, 1] = 0
            generalMatrix[k + 1, 2] = 0
            generalMatrix[k + 1, 3] = 0
            generalMatrix[k + 1, 4], generalMatrix[k + 1, 5], generalMatrix[k + 1,
                                                                            6] = self.points3d[i][0], self.points3d[i][1], self.points3d[i][2]
            generalMatrix[k + 1, 7] = 1
            generalMatrix[k + 1, 8] = - \
                self.points3d[i][0] * self.points2d[i][1]
            generalMatrix[k + 1, 9] = - \
                self.points3d[i][1] * self.points2d[i][1]
            generalMatrix[k + 1, 10] = - \
                self.points3d[i][2] * self.points2d[i][1]
            generalMatrix[k + 1, 11] = -self.points2d[i][1]
            k += 2

        u, s, vh = np.linalg.svd(generalMatrix) 
        matrix = vh  
        subMatrix = matrix[11]
        self.resultMatrix = np.matrix([
            [subMatrix[0], subMatrix[1], subMatrix[2], subMatrix[3]],
            [subMatrix[4], subMatrix[5], subMatrix[6], subMatrix[7]],
            [subMatrix[8], subMatrix[9], subMatrix[10], subMatrix[11]],
            [0, 0, 0, 1]
        ])
        self.resultMatrixInversed = np.linalg.inv(self.resultMatrix)

    def getUnprojectedPoint(self, point2d, height=-12):
        if height == None:
            raise Exception('Point height must be defined for 3d unprojection')
        point1 = np.matrix([[point2d[0]], [point2d[1]], [1], [1]])
        point2 = np.matrix(
            [[100 * point2d[0]], [100 * point2d[1]], [100], [1]])
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
        A = (planePoint2[1] - planePoint1[1]) * (planePoint3[2] - planePoint1[2]) - \
            (planePoint2[2] - planePoint1[2]) * \
            (planePoint3[1] - planePoint1[1])
        B = (planePoint2[2] - planePoint1[2]) * (planePoint3[0] - planePoint1[0]) - \
            (planePoint2[0] - planePoint1[0]) * \
            (planePoint3[2] - planePoint1[2])
        C = (planePoint2[0] - planePoint1[0]) * (planePoint3[1] - planePoint1[1]) - \
            (planePoint2[1] - planePoint1[1]) * \
            (planePoint3[0] - planePoint1[0])
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

    def getUnprojectedSet(self, points):
        return [self.getUnprojectedPoint(x) for x in points]


