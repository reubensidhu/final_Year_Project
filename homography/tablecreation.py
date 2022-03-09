import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from computeProjection import ProjectionCalculator3d
from yoloModel import PoolBallDetection
from getCorners import getCorners


class tablecreation:
    def __init__(self, projectionCalculator3d):#, frameWidth=2016, frameHeight=1512):
        self.projCalculator = projectionCalculator3d
        self.frameWidth = self.projCalculator.frame.shape[1]
        self.frameWidth = self.projCalculator.frame.shape[1]
        self.theight, self.twidth = 445, 865 #252, 504
        self.border_size = 29 #17
        #self.height, self.width = 300, 531
        #self.theight, self.twidth = self.height - self.border_size, self.width - self.border_size
        self.rail_size = int(self.theight*0.02808988764)
        #self.frameWidth = frameWidth
        #self.frameHeight = frameHeight
        self.background = None

    def create_table(self, hsv = [96.91171098, 204.05003044, 209.48951356]):
        # new generated img 
        img = np.zeros((self.theight+(2*self.border_size),self.twidth+(2*self.border_size),3), dtype=np.uint8) # create 2D table image 
        img[:, :] = [12, 99, 227]
        img[self.border_size: self.theight+self.border_size, self.border_size: self.twidth+self.border_size] = [hsv[0], hsv[1], hsv[2]] # setting HSV colors to pool table color
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB) 
        
        #create rail
        cv2.line(img,(self.border_size+self.rail_size, self.border_size+self.rail_size),(self.twidth+self.border_size - self.rail_size, self.border_size + self.rail_size),(0,0,0))
        cv2.line(img,(self.border_size+self.rail_size, self.border_size+self.rail_size),(self.border_size + self.rail_size, self.theight + self.border_size - self.rail_size),(0,0,0))
        cv2.line(img,(self.twidth+self.border_size-self.rail_size, self.border_size+self.rail_size),(self.twidth+self.border_size-self.rail_size, self.theight+self.border_size-self.rail_size),(0,0,0))
        cv2.line(img,(self.border_size+self.rail_size, self.theight+self.border_size-self.rail_size),(self.twidth+self.border_size-self.rail_size, self.theight+self.border_size-self.rail_size),(0,0,0))

        #create dots
        cv2.circle(img, (self.border_size + self.rail_size + (self.twidth-self.rail_size)//4, self.border_size + self.theight//2), 2, (0, 0, 0), -1) 
        cv2.circle(img, (self.border_size + + self.rail_size + (int(3*(self.twidth-self.rail_size)/4)), self.border_size + (self.theight//2)), 2, (0, 0, 0), -1)

        #self.background = img

        #return img

    #def draw_holes(self, color3 = (0,0,0)):
            
        color = (144, 154, 171) # gray color
        color2 = (0, 0, 0) #  gray color, for circles (holes) on generated img

        #img = self.background.copy() # make a copy of input image
        
        # adding circles to represent holes on table
        pocket_radius = int(self.twidth*0.029)
        cv2.circle(img, (self.border_size+5, self.border_size+5), pocket_radius+4,color, -1) # top right
        cv2.circle(img, (self.twidth+self.border_size-5,self.border_size+5), pocket_radius+4, color, -1) # top left
        cv2.circle(img, (self.border_size+5,self.theight+self.border_size-5), pocket_radius+4, color, -1) # bot left
        cv2.circle(img, (self.twidth+self.border_size-5,self.theight+self.border_size-5), pocket_radius+4, color, -1) # bot right
        cv2.circle(img, (self.twidth//2 + self.border_size, self.theight+self.border_size+3), pocket_radius+2, color, -1) # mid right
        cv2.circle(img, (self.twidth//2 + self.border_size,self.border_size-3), pocket_radius+2, color, -1) # mid left
        
        # adding another, smaller circles to the previous ones
        cv2.circle(img, (self.border_size+5, self.border_size+5), pocket_radius, color2, -1) # top right
        cv2.circle(img, (self.twidth+self.border_size-5,self.border_size+5), pocket_radius, color2, -1) # top left
        cv2.circle(img, (self.border_size+5,self.theight+self.border_size-5), pocket_radius, color2, -1) # bot left
        cv2.circle(img, (self.twidth+self.border_size-5,self.theight+self.border_size-5), pocket_radius, color2, -1) # bot right
        cv2.circle(img, (self.twidth//2 + self.border_size,self.theight+self.border_size+3), pocket_radius-2, color2, -1) # mid right
        cv2.circle(img, (self.twidth//2 + self.border_size,self.border_size-3), pocket_radius-2, color2, -1) # mid left
        
        #return self.background
        self.background = img
        return self.background

    def draw_balls(self, results, size = -1, img = 0): #radius of ball is roughly 4.75cm 
        final = self.background.copy()
        radius=int(self.theight*0.02584269662)
        labels, cords = results
        print(results)
        ball_points = []
        for i in range(len(labels)):
            notAdd = False
            if len(ball_points)==16:
                break
            row = cords[i]
            if row[4] >= 0.4:
                x, y = (((row[0]+row[2])*self.frameWidth)//2, ((row[1]+row[3])*self.frameHeight)//2)
                X, Y, Z = self.projCalculator.getUnprojectedPoint((x, y), self.theight*-0.01348314606)#change z component
                X, Y = int(X), int(Y)
                print(X, Y, row[4], labels[i])
                for x in ball_points:
                    if ((math.dist(x[0], (X, Y))) <= (radius*2)) or (labels[i] == x[1]):
                        notAdd = True
                        break
                        
                if notAdd:
                    continue
                else:
                    ball_points.append(((X, Y), labels[i]))

                if labels[i] == 0:
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (255, 255, 255), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1) 
                elif labels[i] == 1: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (245, 212, 27), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(final, (X + self.border_size-3,Y+self.border_size-4), radius//5, (255,255,255), -1)
                elif labels[i] == 2: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (3, 15, 252), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(final, (X + self.border_size-3,Y+self.border_size-4), radius//5, (255,255,255), -1)
                elif labels[i] == 3: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (252, 3, 19), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(final, (X + self.border_size-3,Y+self.border_size-4), radius//5, (255,255,255), -1)
                elif labels[i] == 4: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (30, 1, 115), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(final, (X + self.border_size-3,Y+self.border_size-4), radius//5, (255,255,255), -1)
                elif labels[i] == 5: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (255, 123, 8), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(final, (X + self.border_size-3,Y+self.border_size-4), radius//5, (255,255,255), -1)
                elif labels[i] == 6: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (10, 69, 0), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(final, (X + self.border_size-3,Y+self.border_size-4), radius//5, (255,255,255), -1)
                elif labels[i] == 7: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (61, 11, 0), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(final, (X + self.border_size-3,Y+self.border_size-4), radius//5, (255,255,255), -1)
                elif labels[i] == 8: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (0, 0, 0), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(final, (X + self.border_size-3,Y+self.border_size-4), radius//5, (255,255,255), -1)
                elif labels[i] == 9: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (255, 255, 255), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    #final = cv2.ellipse(final, (X + self.border_size,Y+self.border_size), (radius//3, radius), 90, 0, 360, (255,255,255), -1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+2,Y+self.border_size-5), (X + self.border_size+radius-2,Y+self.border_size+5), (245, 212, 27), -1 )
                elif labels[i] == 10: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (255, 255, 255), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+2,Y+self.border_size-5), (X + self.border_size+radius-2,Y+self.border_size+5), (3, 15, 252), -1 )
                elif labels[i] == 11: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (255, 255, 255), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+2,Y+self.border_size-5), (X + self.border_size+radius-2,Y+self.border_size+5), (252, 3, 19), -1 )
                elif labels[i] == 12: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (255, 255, 255), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+2,Y+self.border_size-5), (X + self.border_size+radius-2,Y+self.border_size+5), (30, 1, 115), -1 )
                elif labels[i] == 13: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (255, 255, 255), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+2,Y+self.border_size-5), (X + self.border_size+radius-2,Y+self.border_size+5), (255, 123, 8), -1 )
                elif labels[i] == 14: 
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (255, 255, 255), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+2,Y+self.border_size-5), (X + self.border_size+radius-2,Y+self.border_size+5), (10, 69, 0), -1 )
                else:
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, (255, 255, 255), size) # -1 to fill ball with color
                    final = cv2.circle(final, (X + self.border_size,Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+2,Y+self.border_size-5), (X + self.border_size+radius-2,Y+self.border_size+5), (61, 11, 0), -1 )
            
            # balls design:
            
            # circle to represent snooker ball

            
            # add black color around the drawn ball (for cosmetics)            
            # small circle for light reflection
            #final = cv2.circle(final, (X-2,Y-2), 2, (255,255,255), -1)
        return final

    def show_img_compar_i(self, imgs):
        f, ax = plt.subplots(1, len(imgs), figsize=(10,10))
        for i in range(len(imgs)):
            img = imgs[i]
            ax[i].imshow(img)
            ax[i].axis('off')
        f.tight_layout()
        plt.savefig('demo.png', bbox_inches='tight')
        #plt.show()

model = PoolBallDetection()
projector = ProjectionCalculator3d(cv2.imread(r'C:\Users\reuby\OneDrive\Pictures\longside1.jpeg') , model)
c = tablecreation(projector)
c.create_table()
frame = cv2.imread(r'C:\Users\reuby\OneDrive\Pictures\longside1.jpeg')
table = c.draw_balls(results=model.score_frame(frame))
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
c.show_img_compar_i([frame, table])