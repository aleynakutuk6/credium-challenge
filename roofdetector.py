import cv2
import copy
import math
import numpy as np

from utils.shape_utils import *

class RoofDetector:
    def __init__(self):
        pass
    
    
    def detect(self, contours, no_dach_num_floors, img_path=None):
        """
            Args:
                contours: contours of the segmentation mask. These are extracted using cv2.
                no_dach_num_floors: the number of floors but roof is not included.
            Returns:
                res_flag: returns True or False depends on roof existence.
        """
        strokes = self.contour_to_stroke3(contours)
        simplified_strokes = apply_RDP(strokes, is_absolute=True, epsilon=5.0)
        corners, middles, _ = self.get_main_pts(simplified_strokes)
        res_flag = self.check_roof_existence(corners, middles, no_dach_num_floors)
        
        if img_path is not None:
            xmin, ymin, xmax, ymax = get_absolute_bounds(simplified_strokes)
            out_img = draw_strokes(simplified_strokes, max_dim=int(max(xmax, ymax))+50, keep_size=True, is_absolute=True)
        
            for x, y, _ in simplified_strokes:
                out_img = cv2.circle(out_img, (int(x), int(y)), 1, (0, 0, 255), 2)
                
            for m in middles:
                if m is not None:
                    out_img = cv2.circle(out_img, m, 2, (0, 255, 0), 2)
                
            for c in corners:
                if c is not None:
                    out_img = cv2.circle(out_img, c, 2, (255, 0, 0), 2)
        
            cv2.imwrite(img_path, out_img)
        
        return res_flag
    
    
    def contour_to_stroke3(self, contours: np.ndarray):
        """
            This function takes contours of page segmentation masks and 
                converts them into stroke-3 format (frequently used in sketch representation). 
        """
        strokes = []
        for c in contours:
            c = c[:,0,:]
            c = np.concatenate([c, np.zeros((c.shape[0], 1))], axis=-1)
            c[-1, -1] = 1.0
            strokes.append(c)        
        
        if len(strokes) == 1:
            strokes = strokes[0].astype(float)
        else:
            strokes = np.concatenate(strokes, axis=0).astype(float)

        return strokes
        

    def get_main_pts(self, strokes):
        """
            This function takes contours in stroke-3 format and finds the four corners of the shape along with the roof top points
            in any 90 degree rotation (e.g. a house can flipped or rotated in the pdf file).
        """
        
        xmin, ymin, xmax, ymax = get_absolute_bounds(strokes)
        cx_l, cy_l = xmin + (xmax - xmin) / 5, ymin + (ymax - ymin) / 5
        cx_r, cy_r = xmin + 4 * (xmax - xmin) / 5, ymin + 4 * (ymax - ymin) / 5
        
        y_order = np.argsort(strokes[:, 1])
        x_order = np.argsort(strokes[:, 0])
    
        topleft, topright, bottomleft, bottomright = None, None, None, None
        midtop, midbottom, midleft, midright = None, None, None, None
        
        for o in y_order:
            x, y = int(strokes[o, 0]), int(strokes[o, 1])
            if x < cx_l and topleft is None: topleft = [x, y]
            elif x > cx_r and topright is None: topright = [x, y]
            elif x > cx_l and x < cx_r and midtop is None: midtop = [x, y]
            if topleft is not None and topright is not None and midtop is not None:
                break
                
        for o in y_order[::-1]:
            x, y= int(strokes[o, 0]), int(strokes[o, 1])
            if x < cx_l and bottomleft is None: bottomleft = [x, y]
            elif x > cx_r and bottomright is None: bottomright = [x, y]
            elif x > cx_l and x < cx_r and midbottom is None: midbottom = [x, y]
            if bottomleft is not None and bottomright is not None and midbottom is not None:
                break      
        
        for o in x_order:
            x, y = int(strokes[o, 0]), int(strokes[o, 1])
            if y > cy_l and y < cy_r and midleft is None: 
                midleft = [x, y]
                break
                
        for o in x_order[::-1]:
            x, y = int(strokes[o, 0]), int(strokes[o, 1])
            if y > cy_l and y < cy_r and midright is None:
                midright = [x, y]
                break
        
        return [topleft, topright, bottomleft, bottomright], [midtop, midbottom, midleft, midright], [cx_l, cy_l, cx_r, cy_r]
    

    def check_roof_existence(self, corners, middles, no_dach_num_floors):
        """
            Checks if the middle points (possible roof tops) are between the corners and has a significant
            height compared to the closest side between corners.
        """
        tl, tr, bl, br = corners
        mt, mb, ml, mr = middles
        # print("corners:", corners)
        # print("middles:", middles)
        
        # If the middle point is on the top
        if mt is not None and mt[1] < min(tl[1], tr[1]):
            h = max(bl[1], br[1]) - min(tl[1], tr[1])
            roof_len = min(tl[1], tr[1]) - mt[1]
            min_roof_len = 0.75 * (h / no_dach_num_floors)
            # print("case 1")
            # print(min_roof_len, roof_len)
            if roof_len > min_roof_len:
                return True
        
        # If the middle point is on the bottom
        if mb is not None and mb[1] > max(bl[1], br[1]):
            h = max(bl[1], br[1]) - min(tl[1], tr[1])
            roof_len = mb[1] - max(bl[1], br[1])
            min_roof_len = 0.75 * (h / no_dach_num_floors)
            # print("case 2")
            # print(min_roof_len, roof_len)
            if roof_len > min_roof_len: 
                return True
        
        # If the middle point is on the left
        if ml is not None and ml[0] < min(tl[0], bl[0]):
            h = max(tr[0], br[0]) - min(tl[0], bl[0])
            roof_len = min(tl[0], bl[0]) - ml[0]
            min_roof_len = 0.75 * (h / no_dach_num_floors)
            # print("case 3")
            # print(min_roof_len, roof_len)
            if roof_len > roof_len: 
                return True
        
        # If the middle point is on the right
        if mr is not None and mr[0] > max(tr[0], br[0]):
            h = max(tr[0], br[0]) - min(tl[0], bl[0])
            roof_len = mr[0] - max(tr[0], br[0])
            min_roof_len = 0.75 * (h / no_dach_num_floors)
            # print("case 4")
            # print(min_roof_len, roof_len)
            if roof_len > min_roof_len:
                return True
        
        return False