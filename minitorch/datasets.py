import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """
    Generates a dataset that is linearly separable by a single vertical line.

    Points with x_1 < 0.5 are labeled 1, all others are labeled 0.
    This is the easiest dataset to classify: a single straight decision
    boundary (x_1 = 0.5) perfectly separates the two classes.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """
    Generates a dataset that is linearly separable by a diagonal line.

    Points with x_1 + x_2 < 0.5 are labeled 1, all others are labeled 0.
    The decision boundary is the diagonal line x_1 + x_2 = 0.5, so a
    single linear classifier with both weights nonzero can separate it.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """
    Generates a dataset split into two vertical bands on the outer edges.

    Points with x_1 < 0.2 or x_1 > 0.8 are labeled 1, all points in the
    middle band (0.2 <= x_1 <= 0.8) are labeled 0. This is NOT linearly
    separable by a single line, since class 1 occupies two disjoint
    regions on opposite sides of class 0.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """
    Generates the classic XOR dataset.

    Points are labeled 1 when x_1 and x_2 fall on opposite sides of 0.5
    (i.e. exactly one of them is > 0.5), and 0 otherwise. This is the
    canonical example of a dataset that is NOT linearly separable and
    requires a nonlinear (or multi-layer) decision boundary.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """
    Generates a dataset where one class forms a ring around the center.

    Points outside a circle of radius sqrt(0.1) centered at (0.5, 0.5)
    are labeled 1, points inside are labeled 0. This requires a circular
    (nonlinear) decision boundary and cannot be separated by a straight line.
    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """
    Generates two interleaved spiral arms, one per class.

    Class 0 and class 1 each form a spiral curve winding around the
    center, offset from each other. This is a highly nonlinear pattern
    that cannot be separated by simple curves like a line or circle,
    and is typically used to test more expressive (multi-layer) models.
    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
