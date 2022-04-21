import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

class tablecreation:
    
    def __init__(self, projectionCalculator3d):
        self.projCalculator = projectionCalculator3d
        self.frameHeight = self.projCalculator.frame.shape[0]
        self.frameWidth = self.projCalculator.frame.shape[1]
        self.theight, self.twidth = 890, 1730  # 445, 865
        self.border_size = 58  # 29

        self.rail_size = 25  
        self.background = None

    def create_table(self, hsv=[96.91171098, 204.05003044, 209.48951356]):
        # new generated img
        img = np.zeros((self.theight+(2*self.border_size), self.twidth +
                       (2*self.border_size), 3), dtype=np.uint8)  # create 2D table image
        img[:, :] = [12, 99, 227]
        img[self.border_size: self.theight+self.border_size, self.border_size: self.twidth +
            self.border_size] = [hsv[0], hsv[1], hsv[2]]  # setting HSV colors to pool table color
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        # create rail
        cv2.line(img, (self.border_size+self.rail_size, self.border_size+self.rail_size),
                 (self.twidth+self.border_size - self.rail_size, self.border_size + self.rail_size), (0, 0, 0))
        cv2.line(img, (self.border_size+self.rail_size, self.border_size+self.rail_size),
                 (self.border_size + self.rail_size, self.theight + self.border_size - self.rail_size), (0, 0, 0))
        cv2.line(img, (self.twidth+self.border_size-self.rail_size, self.border_size+self.rail_size),
                 (self.twidth+self.border_size-self.rail_size, self.theight+self.border_size-self.rail_size), (0, 0, 0))
        cv2.line(img, (self.border_size+self.rail_size, self.theight+self.border_size-self.rail_size),
                 (self.twidth+self.border_size-self.rail_size, self.theight+self.border_size-self.rail_size), (0, 0, 0))

        # create dots
        cv2.circle(img, (self.border_size + self.rail_size + round((self.twidth -
                   self.rail_size)/4), round(self.border_size + self.theight/2)), 2, (0, 0, 0), -1)
        cv2.circle(img, (self.border_size + + self.rail_size + (round(3*(self.twidth -
                   self.rail_size)/4)), round(self.border_size + self.theight/2)), 2, (0, 0, 0), -1)

      

        color = (144, 154, 171)  # gray color
        color2 = (0, 0, 0)  # gray color, for circles (holes) on generated img


        # adding circles to represent holes on table
        pocket_radius = round(self.twidth*0.029)
        cv2.circle(img, (self.border_size+5, self.border_size+5),
                   pocket_radius+4, color, -1)  # top right
        cv2.circle(img, (self.twidth+self.border_size-5,
                   self.border_size+5), pocket_radius+4, color, -1)  # top left
        cv2.circle(img, (self.border_size+5, self.theight +
                   self.border_size-5), pocket_radius+4, color, -1)  # bot left
        cv2.circle(img, (self.twidth+self.border_size-5, self.theight +
                   self.border_size-5), pocket_radius+4, color, -1)  # bot right
        cv2.circle(img, (self.twidth//2 + self.border_size, self.theight +
                   self.border_size+3), pocket_radius+2, color, -1)  # mid right
        cv2.circle(img, (self.twidth//2 + self.border_size,
                   self.border_size-3), pocket_radius+2, color, -1)  # mid left

        # adding another, smaller circles to the previous ones
        cv2.circle(img, (self.border_size+5, self.border_size+5),
                   pocket_radius, color2, -1)  # top right
        cv2.circle(img, (self.twidth+self.border_size-5,
                   self.border_size+5), pocket_radius, color2, -1)  # top left
        cv2.circle(img, (self.border_size+5, self.theight +
                   self.border_size-5), pocket_radius, color2, -1)  # bot left
        cv2.circle(img, (self.twidth+self.border_size-5, self.theight +
                   self.border_size-5), pocket_radius, color2, -1)  # bot right
        cv2.circle(img, (self.twidth//2 + self.border_size, self.theight +
                   self.border_size+3), pocket_radius-2, color2, -1)  # mid right
        cv2.circle(img, (self.twidth//2 + self.border_size,
                   self.border_size-3), pocket_radius-2, color2, -1)  # mid left

        # return self.background
        self.background = img
        return self.background

    def draw_balls(self, outputs, confs, size=-1, img=0):  # radius of ball is roughly 4.75cm
        final = self.background.copy()
        radius = 23  # int(self.theight*0.02584269662)
        ball_points = []
        for i, (output, conf) in enumerate(zip(outputs, confs)):
            notAdd = False
            if len(ball_points) == 16:
                break
            #row = cords[i]
            if conf >= 0.4:
                X, Y = int(output[0]), int(output[1])
                print(X, Y, confs[i], output[3])
                for x in ball_points:
                    if ((math.sqrt((x[0][0] - X)**2 + (x[0][1] - Y)**2)) <= ((radius*2) or (output[3] == x[1]))):
                        notAdd = True
                        break

                if notAdd:
                    continue
                else:
                    ball_points.append(((X, Y), output[3]))

                if output[3] == 0:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (255, 255, 255), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                elif output[3] == 1:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (245, 212, 27), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(
                        final, (X + self.border_size-4, Y+self.border_size-6), radius//5, (255, 255, 255), -1)
                elif output[3] == 2:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (3, 15, 252), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(
                        final, (X + self.border_size-4, Y+self.border_size-6), radius//5, (255, 255, 255), -1)
                elif output[3] == 3:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (252, 3, 19), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(
                        final, (X + self.border_size-4, Y+self.border_size-6), radius//5, (255, 255, 255), -1)
                elif output[3] == 4:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (30, 1, 115), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(
                        final, (X + self.border_size-4, Y+self.border_size-6), radius//5, (255, 255, 255), -1)
                elif output[3] == 5:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (255, 123, 8), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size-4, Y+self.border_size-6), radius//5, (255, 255, 255), -1)
                elif output[3] == 6:
                    # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, (10, 69, 0), size)
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(
                        final, (X + self.border_size-4, Y+self.border_size-6), radius//5, (255, 255, 255), -1)
                elif output[3] == 7:
                    # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, (61, 11, 0), size)
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(
                        final, (X + self.border_size-4, Y+self.border_size-6), radius//5, (255, 255, 255), -1)
                elif output[3] == 8:
                    # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, (0, 0, 0), size)
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.circle(
                        final, (X + self.border_size-4, Y+self.border_size-6), radius//5, (255, 255, 255), -1)
                elif output[3] == 9:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (255, 255, 255), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+3, Y+self.border_size-10),
                                          (X + self.border_size+radius-3, Y+self.border_size+10), (245, 212, 27), -1)
                elif output[3] == 10:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (255, 255, 255), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+3, Y+self.border_size-10),
                                          (X + self.border_size+radius-3, Y+self.border_size+10), (3, 15, 252), -1)
                elif output[3] == 11:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (255, 255, 255), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+3, Y+self.border_size-10),
                                          (X + self.border_size+radius-3, Y+self.border_size+10), (252, 3, 19), -1)
                elif output[3] == 12:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (255, 255, 255), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+3, Y+self.border_size-10),
                                          (X + self.border_size+radius-3, Y+self.border_size+10), (30, 1, 115), -1)
                elif output[3] == 13:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (255, 255, 255), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+3, Y+self.border_size-10),
                                          (X + self.border_size+radius-3, Y+self.border_size+10), (255, 123, 8), -1)
                elif output[3] == 14:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (255, 255, 255), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+3, Y+self.border_size-10),
                                          (X + self.border_size+radius-3, Y+self.border_size+10), (10, 69, 0), -1)
                else:
                    final = cv2.circle(final, (X + self.border_size, Y + self.border_size),
                                       radius, (255, 255, 255), size)  # -1 to fill ball with color
                    final = cv2.circle(
                        final, (X + self.border_size, Y + self.border_size), radius, 0, 1)
                    final = cv2.rectangle(final, (X + self.border_size-radius+3, Y+self.border_size-10),
                                          (X + self.border_size+radius-3, Y+self.border_size+10), (61, 11, 0), -1)

        return final

    def show_img_compar_i(self, imgs):
        f, ax = plt.subplots(1, len(imgs), figsize=(10, 10))
        for i in range(len(imgs)):
            img = imgs[i]
            ax[i].imshow(img)
            ax[i].axis('off')
        f.tight_layout()
        plt.savefig('demo.png', bbox_inches='tight')
        # plt.show()

