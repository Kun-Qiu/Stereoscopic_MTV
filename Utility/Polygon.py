"""
Contains functions to check if a points lies within a
polygon given its vertices

Assumption: Polygon is a convex polygon with no holes

Code partially taken from Ray Casting Algorithm by Computer Geekery
"""
import math


def sort_coordinates_clockwise(points):
    # Calculate the centroid
    centroid_x = sum(point.x for point in points) / len(points)
    centroid_y = sum(point.y for point in points) / len(points)

    # Function to calculate the angle of each point
    def angle_from_centroid(point):
        return math.atan2(point.y - centroid_y, point.x - centroid_x)

    # Sort points by the calculated angles
    sorted_points = sorted(points, key=angle_from_centroid)

    return sorted_points


def is_point_on_edge(point, A, B):
    """
    Check if the point is exactly on the edge defined by points A and B
    :param point: Point to check
    :param A: Endpoint of the edge
    :param B: Endpoint of the edge
    :return: True if point is on the edge, False otherwise
    """
    cross_product = (point.y - A.y) * (B.x - A.x) - (point.x - A.x) * (B.y - A.y)
    if abs(cross_product) > 1e-10:
        return False

    dot_product = (point.x - A.x) * (B.x - A.x) + (point.y - A.y) * (B.y - A.y)
    if dot_product < 0:
        return False

    squared_length = (B.x - A.x) ** 2 + (B.y - A.y) ** 2
    if dot_product > squared_length:
        return False

    return True


# ________________________________________________________________________________________

class Point:
    def __init__(self, x, y):
        """
        A point specified by (x,y) coordinates in the cartesian plane
        :param x: x coordinate
        :param y: y coordinate
        """
        self.x = x
        self.y = y


class Polygon:
    def __init__(self, points):
        """
        :param points:  a list of points
        """
        self.points = sort_coordinates_clockwise(points)

    def edges(self):
        """
        Obtain the edges contained by the vertices
        :return: tuples that contains 2 points of an edge element wise
        """
        edge_list = []
        for i, p in enumerate(self.points):
            p1 = p
            p2 = self.points[(i + 1) % len(self.points)]
            edge_list.append((p1, p2))

        return edge_list

    def contains(self, point):
        """
        Determine if the point is on or inside the polygon
        :param point: Test point
        :return: Boolean on whether the point satisfy the requirement
        """
        import sys
        # _huge is used to act as infinity if we divide by 0
        # _eps is used to make sure points are not on the same line as vertices
        _huge = sys.float_info.max
        _eps = 0.00001

        inside = False
        for edge in self.edges():
            A, B = edge[0], edge[1]
            if A.y > B.y:
                A, B = B, A

            # Check if point is on the edge
            if is_point_on_edge(point, A, B):
                return True

            # Make sure point is not at the same height as vertex
            if point.y == A.y or point.y == B.y:
                point.y += _eps

            if point.y > B.y or point.y < A.y or point.x > max(A.x, B.x):
                # The horizontal ray does not intersect with the edge
                continue

            if point.x < min(A.x, B.x):  # The ray intersects with the edge
                inside = not inside
                continue

            try:
                m_edge = (B.y - A.y) / (B.x - A.x)
            except ZeroDivisionError:
                m_edge = _huge

            try:
                m_point = (point.y - A.y) / (point.x - A.x)
            except ZeroDivisionError:
                m_point = _huge

            if m_point >= m_edge:
                # The ray intersects with the edge
                inside = not inside
                continue

        return inside
