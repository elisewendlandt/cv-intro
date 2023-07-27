import cv2
import numpy as np
import matplotlib.pyplot as plt


def detect_lines(
    img: np.ndarray,
    threshold1: int = 50,
    threshold2: int = 150,
    apertureSize: int = 3,
    minLineLength: int = 100,
    maxLineGap: int = 10,
) -> np.ndarray:
    """
    Takes an image and returns a list of detected lines.

        Parameters:
            img (np.ndarray): The image to process.
            threshold1 (int): The first threshold for the Canny edge detector (default: 50).
            threshold2 (int): The second threshold for the Canny edge detector (default: 150)
            apertureSize (int): The aperture size for the Sobel operator (default: 3)
            minLineLength (int): The minimum length of a line (default: 100)
            maxLineGap (int): The maximum gap between two points to be considered in the same line (default: 10)

        Returns:
            lines (np.ndarray)
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2, apertureSize=apertureSize)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, 100, minLineLength=minLineLength, maxLineGap=maxLineGap
    )
    print(edges, lines)
    return lines


def draw_lines(img: np.ndarray, lines: np.ndarray, color: tuple = (0, 255, 0)):
    """
    Takes an image and a list of lines and returns an image with the lines drawn on it.

        Parameters:
            img (str): The image to process.
            lines (np.ndarray): The list of lines to draw
            color (tuple): The color of the lines. Default: (0, 255, 0)
    """

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, 2)


def get_slopes_intercepts(lines: np.ndarray):
    """
    Takes a list of lines and returns a list of slopes and a list of intercepts.

        Parameters:
            lines (np.ndarray)

        Returns:
            slopes (np.ndarray): The list of slopes
            intercepts (np.ndarray): The list of intercepts
    """

    slopes = (lines[:, :, 3] - lines[:, :, 1]) / (lines[:, :, 1] - lines[:, :, 0])
    # b = y - mx
    b = lines[:, :, 1] - slopes * lines[:, :, 0]
    intercepts = (np.zeros_like(slopes) - b) / slopes
    return slopes, intercepts


def detect_lanes(img: np.ndarray, lines: np.ndarray):
    """
    Takes a list of lines as an input and returns a list of lanes.

        Parameters:
            img (np.ndarray): The image, for comparing pixels.
            lines (np.ndarray): The list of lines to process.

        Returns:
            lanes (list): The list of lanes.
    """

    def chunk(l, n):
        return [l[i : i + n] for i in range(0, len(l), n)]

    def avg_color(*colors):
        """
        Find the average of color one and two.

            Parameters:
                *colors (tuple)
        """

        return np.average(colors)

    slopes, intercepts = get_slopes_intercepts(lines)
    # => ([x1, y1, x2, y2], slope, x-intercept) for every "line"
    sort = sorted(zip(lines, slopes, intercepts), key=lambda pair: pair[1])
    #

    return chunk(sort, 2)


def draw_lanes(img: np.ndarray, lanes: list):
    """
    Takes an image and a list of lanes as inputs and returns an image with the lanes drawn on it.

        Parameters:
            img (np.ndarray): The image to process
            lanes (list)
    """

    random_color = lambda: list(np.random.random(size=3) * 256)
    for pair in lanes:
        print(pair)
        color = random_color()
        for lane in pair:
            x1, y1, x2, y2 = lane[0][0]
            cv2.line(img, (x1, y1), (x2, y2), color, 2)