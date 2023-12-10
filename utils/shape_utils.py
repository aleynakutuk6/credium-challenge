import copy
import math
import numpy as np
import cv2

from rdp import rdp


def get_absolute_bounds(data: np.ndarray) -> list:
    """
        Args:
            data: gets strokes as ndarray.
        Returns:
            min_x, min_y, max_x, max_y: returns bounds of data.
    
    """
    min_x, max_x, min_y, max_y = 10000000, 0, 10000000, 0
    
    for i in range(data.shape[0]):
        x = float(data[i, 0])
        y = float(data[i, 1])
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x)
        max_y = max(max_y, y)

    return min_x, min_y, max_x, max_y


def absolute_to_relative(strokes: np.ndarray, start_from_zero: bool=True) -> np.ndarray:
    """ 
    This function returns a relative strokes with the start point as [0, 0] 
    
    """
    relative_strokes = np.zeros_like(strokes)
    for i in range(1, strokes.shape[0]):
        relative_strokes[i, :2] = strokes[i, :2] - strokes[i-1, :2]
        relative_strokes[i, 2:] = strokes[i, 2:]
    if not start_from_zero:
        relative_strokes[0, :2] = strokes[0, :2]
    return relative_strokes   
    

def apply_RDP(strokes: np.ndarray, is_absolute: bool=False, epsilon: float=2.0):
    """
        The Ramer-Douglas-Peucker algorithm is an algorithm for reducing the number of points in a curve that is approximated by a series of points.
        
        Args:
            strokes: stroke-3 format in relative coordinates.
            is_absolute: If strokes are in absolute coordinates, set True. default=False
            epsilon: a float denotes the simplification threshold, indicating the minimum distance required for 
                the center point to the line formed by connecting two adjacent points.
                
        Returns simplified_strokes applied rdp algorithm.
    
    """
    
    if is_absolute:
        rel_strokes = absolute_to_relative(strokes, start_from_zero=False)
    else:
        rel_strokes = strokes
    
    los = stroke3_to_strokelist(rel_strokes)
    new_lines = []
    for stroke in los:
        simplified_stroke = rdp(stroke, epsilon=epsilon)
        if len(simplified_stroke) > 1:
            new_lines.append(simplified_stroke)
    
    simplified_strokes = strokelist_to_stroke3(new_lines, return_is_absolute=is_absolute)
    return simplified_strokes


def draw_strokes(
    strokes: np.ndarray, save_path: str=None, margin: int=10, keep_size: bool=False,
    is_absolute: bool=False, max_dim: int=512, white_bg: bool=False):
    
    """
        This function takes strokes and saves the image to the given path.
        
        Args:
            strokes: stroke-3 format in relative coordinates.
            save_path: image path to save.
            keep_size: whether to keep the size of image same or not.
            is_absolute: If strokes are in absolute coordinates, set True. default=False
            white_bg: image will be drawn on white or black background.
    """
    if white_bg:
        canvas = np.full((max_dim, max_dim, 3), 255, dtype=np.uint8)
        fill_color = (0, 0, 0)
    else:
        canvas = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
        fill_color = (255, 255, 255)
    
    
    if not is_absolute:
        abs_strokes = relative_to_absolute(copy.deepcopy(strokes))
    else:
        abs_strokes = copy.deepcopy(strokes)
    
    if not keep_size:
        xmin, ymin, xmax, ymax = get_absolute_bounds(abs_strokes)
        abs_strokes[:,0] -= xmin
        abs_strokes[:,1] -= ymin
        abs_strokes = normalize_to_scale(
            abs_strokes, is_absolute=True, scale_factor=max_dim-2*margin)
        if abs_strokes is None:
            return None
        abs_strokes[:,:2] += margin # pads margin px to top and left sides
    
    for i in range(1, abs_strokes.shape[0]):
        if abs_strokes[i-1, -1] > 0.5: continue # stroke end
        px, py = int(abs_strokes[i-1, 0]), int(abs_strokes[i-1, 1])
        x, y   = int(abs_strokes[i, 0]), int(abs_strokes[i, 1])
        canvas = cv2.line(canvas, (px, py), (x, y), color=fill_color, thickness=2)
    
    if save_path is not None:
        cv2.imwrite(save_path, canvas)
    
    return canvas
    

def stroke3_to_strokelist(strokes: np.ndarray, is_absolute: bool=False) -> list:
    """ 
    This function makes conversion:
        - From: stroke-3 format 
        - To: [[[x1, y1], [x2, y2], ...], [[x3, y3], [x4, y4], ...], ...]
    """
    if not is_absolute:
        abs_strokes = relative_to_absolute(copy.deepcopy(strokes))
    else:
        abs_strokes = strokes
    
    new_strokes, points = [], []
    for i in range(strokes.shape[0]):
        points.append([abs_strokes[i, 0], abs_strokes[i, 1]])
        if abs_strokes[i, 2] == 1:
            new_strokes.append(points)
            points = []
    
    return new_strokes


def relative_to_absolute(strokes: np.ndarray) -> np.ndarray:
    absolute_strokes = np.zeros_like(strokes)
    absolute_strokes[0, :] = strokes[0, :]
    for i in range(1, strokes.shape[0]):
        absolute_strokes[i, 0] = absolute_strokes[i-1, 0] + strokes[i, 0]
        absolute_strokes[i, 1] = absolute_strokes[i-1, 1] + strokes[i, 1]
        absolute_strokes[i, 2:] = strokes[i, 2:]
    return absolute_strokes


def strokelist_to_stroke3(strokelist: list, return_is_absolute: bool=False) -> np.ndarray:
    """ Makes conversion:
    - From: [[[x1, y1], [x2, y2], ...], [[x3, y3], [x4, y4], ...], ...]
    - To: stroke-3 format 
    """
    stroke3 = []
    for stroke in strokelist:
        for x, y in stroke:
            stroke3.append([x, y, 0.0])
        stroke3[-1][-1] = 1.0
    
    stroke3 = np.asarray(stroke3)
    
    if not return_is_absolute:
        stroke3 = absolute_to_relative(stroke3)
        
    return stroke3