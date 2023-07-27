import numpy as np


def get_lane_center(lanes: np.ndarray):
    """
    Takes a list of lanes and returns the center of the closest lane and its slope.

        Parameters:
            lanes (np.ndarray): The list of lanes to process.

        Returns:
            center_intercept (float): The horizontal intercept of the center of the closest lane.
            center_slope (float): The slope of the closest lane.
    """
    height, width, depth = img.shape()
    center = width/2
    min = 10000000000
    closest_lane = np.ndarray
    for lane in lanes:
        # [[[x1,y1,x2,y2],slope,intercept],[[x1,y1,x2,y2],slope,intercept]]
        intercepts = lane[:,2]
        for i in range(0,len(intercepts)-1):
            if intercepts[i]<min:
                min = intercepts[i]
                closest_lane = lane
        midpoint1 = [(lane[0,2]-lane[1,2])/2,0]
        midpoint2 = [((lane[0,2]+lane[0,1])-(lane[1,2]+lane[1,1])),1]
        center_slope = (midpoint1[1]-midpoint2[1])/(midpoint1[0]-midpoint2[0])
        center_intercept = midpoint1[0]
    # [[[[x1,y1,x2,y2],slope,intercept],[[x1,y1,x2,y2],slope,intercept]]]

    return(center_intercept, center_slope)


def recommend_direction(center: float, slope: float):
    """
    Takes the center of the closest lane and its slope as inputs and returns a direction.

        Parameters:
            center (float): The center of the closest lane.
            slope (float): The slope of the closest lane.

        Returns:
            direction (str): left, right, forward
    """

    height, width = img.shape() 
    camera_pov = width/2

    if camera_pov < center and slope < 0:
        return "Right"
    elif camera_pov > center and slope > 0:
        return "Left"
    else:
        return "Straight"

    