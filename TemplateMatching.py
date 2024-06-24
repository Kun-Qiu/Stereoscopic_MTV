import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import copy

import Utility.Polygon as pl
from torchvision import models, transforms
from robustTemplateMatching.FeatureExtractor import FeatureExtractor


def nms(dets, scores, thresh):
    """
    Non-maximum suppression of multiple detection of the template within
    an image.

    :param dets:
    :param scores:
    :param thresh:
    :return:
    """
    x1 = dets[:, 0, 0]
    y1 = dets[:, 0, 1]
    x2 = dets[:, 1, 0]
    y2 = dets[:, 1, 1]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def moving_average_validation(vectors, radius, threshold=0.6):
    """
    Moving average filter which compares displacement vectors with its neighbors.
    If the deviation is above a threshold, the vector is replaced with the average
    of the neighborhood of radius r.

    :param vectors:     The displacement vectors with their spatial positions (list of tuples).
    :param radius:      The radius of the neighborhood.
    :param threshold:   Threshold value for deviation from average of neighborhood.
    :return:            Array of validated displacement vectors based on moving average.
    """
    validated_vectors = []

    for index, vector in enumerate(vectors):
        neighborhood = []
        for i in range(len(vectors)):
            if i != index:
                distance = np.sqrt((vector[0][0] - vectors[i][0][0]) ** 2 +
                                   (vector[0][1] - vectors[i][0][1]) ** 2)
                if distance < radius:
                    neighborhood.append([vectors[i][1], vectors[i][2]])

        if len(neighborhood) == 0:
            avg_vector = vector[1:]  # If no neighbors, keep the original vector (excluding the position)
        else:
            avg_vector = np.mean(np.array(neighborhood), axis=0)

        # Cosine similarity
        dis_vector = np.array([vector[1], vector[2]])
        avg_vector = np.array(avg_vector)

        norm_dis_vector = np.linalg.norm(dis_vector)
        norm_avg_vector = np.linalg.norm(avg_vector)

        if norm_dis_vector == 0 or norm_avg_vector == 0:
            cosine_similarity = 0  # Handle the zero norm case
        else:
            inner_product_displacement = np.dot(dis_vector, avg_vector)
            product_displacement = norm_dis_vector * norm_avg_vector
            cosine_similarity = inner_product_displacement / product_displacement

        if cosine_similarity < threshold:
            validated_vectors.append((vector[0], avg_vector[0], avg_vector[1]))
        else:
            validated_vectors.append(vector)

    return validated_vectors


def average_filter(vectors, radius):
    """
    The average filter method can be used to filter out the vector maps by arithmetic
    averaging over vector neighbors to reduce the noise

    :param vectors:     Array of displacement vectors
    :param radius:      Radius of the neighborhood
    :return:            Smoothed displacement field based on the neighborhood average
    """
    smoothed_vectors = []

    for index, vector in enumerate(vectors):
        neighborhood = []
        for i in range(len(vectors)):
            if i != index:
                distance = np.sqrt((vector[0][0] - vectors[i][0][0]) ** 2 +
                                   (vector[0][1] - vectors[i][0][1]) ** 2)
                if distance < radius:
                    neighborhood.append([vectors[i][1], vectors[i][2]])

        avg_vector = np.mean(np.array(neighborhood), axis=0)
        smoothed_vectors.append((vectors[index][0], avg_vector[0], avg_vector[1]))
    return smoothed_vectors


def match_template(raw_img, img, template, polygon, use_CUDA=False, use_cython=False,
                   threshold=None, nms_thresh=0.5):
    """
    Using OpenCV template matching module, the template is being matched to the
    image to find location where the similarity is the strongest

    :param raw_img:         The image before preprocessing
    :param img:             The image of which template matching is applied to
    :param template:        The template of which used to match for similarity
    :param polygon:         The spatial constraint for filtering out false positive detections
    :param use_CUDA:        Use CUDA for computation (Speed up computation)
    :param use_cython:      Use Cython to compile C
    :param threshold:       Threshold for the normalized cross correlation
    :param nms_thresh:      Threshold for the Non-Maximum Suppression
    :return:                List of (x, y) coordinates where similarity is above threshold value
    """

    vgg_feature = models.vgg13(pretrained=True).features
    FE = FeatureExtractor(vgg_feature, use_cuda=use_CUDA, padding=True)
    boxes, centers, scores = FE(
        template, img, threshold=threshold, use_cython=use_cython)
    d_img = raw_img.astype(np.uint8).copy()

    # Avoid index error if no box is detected
    real_centers = []
    img_temp = d_img.copy()
    if len(boxes) > 0:
        nms_res = nms(np.array(boxes), np.array(scores), thresh=nms_thresh)
        print("detected objects: {}".format(len(nms_res)))
        for i in nms_res:
            # if polygon.contains(pl.Point(centers[i][0], centers[i][1])):
            cv2.circle(img_temp, centers[i], 0, (0, 0, 255), 2)
            real_centers.append(centers[i])

    # cv2.namedWindow("Match", cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("Match", 800, 800)
    # cv2.imshow("Match", img_temp)
    # cv2.waitKey()
    return real_centers


class TemplateMatcher:
    def __init__(self, source_image_path, target_image_path, template_path,
                 thresh_source=0.8, thresh_target=0.8):
        """
        Default Constructor for Template Matcher

        :param source_image_path:       Path to the source image
        :param target_image_path:       Path to the target image
        :param template_path:           Path to the template image
        :param thresh_source:           Correlation threshold for source image (Default=0.8)
        :param thresh_target:           Correlation threshold for target image (Default=0.8)
        :return:                        Returns template matching object
        """

        image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

        self._raw_source = cv2.imread(source_image_path)[..., ::-1]
        self._raw_target = cv2.imread(target_image_path)[..., ::-1]
        self._raw_template = cv2.imread(template_path)[..., ::-1]

        self._template = image_transform(self._raw_template.copy()).unsqueeze(0)
        self._source = image_transform(self._raw_source.copy()).unsqueeze(0)
        self._target = image_transform(self._raw_target.copy()).unsqueeze(0)

        self._source_points = None
        self._target_points = None
        self._polygon = None
        self._displacement = None
        self._length = np.loadtxt('../length.txt')
        self._thresh_source = thresh_source
        self._thresh_target = thresh_target

    def set_boundary(self):
        """
        For easier detection of the relevant points, the user is asked to plot the vertices of the polygon
        that encompasses the region of interest. The boundary and the edges of the polygon is obtained to
        assist the filtering of the detected intersection.

        :return: A boundary object
        """
        points = []
        img_clone = self._raw_source.copy()

        def pick(event, x, y, flags, param):
            nonlocal points
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append(pl.Point(x, y))
                cv2.circle(img_clone, (x, y), 3, (0, 255, 0), -1)
                cv2.imshow("Boundary", img_clone)

        cv2.namedWindow("Boundary", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Boundary", 600, 600)
        cv2.imshow("Boundary", img_clone)
        cv2.setMouseCallback("Boundary", pick)

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):  # press 'q' to quit picking points
                break
        cv2.destroyAllWindows()

        return pl.Polygon(points)

    def visualize_match(self, source_points, target_points, source_correspondence):
        """
        Visualize the displacement of the source points to the target points

        :param source_points:           List of tuples [(x, y), (x1, y1), ...] representing source points
        :param target_points:           List of tuples [(x, y), (x1, y1), ...] representing target points
        :param source_correspondence:   List of tuples that represents the displacement
        :return:                        None
        """

        # Create a plot
        plt.figure(figsize=(10, 8))
        plt.imshow(self._raw_source)

        # Plot the source points
        source_x = [point[0] for point in source_points]
        source_y = [point[1] for point in source_points]
        plt.scatter(source_x, source_y, c='red', label='Source Points', s=40, edgecolor='black')

        # Plot the target points
        target_x = [point[0] for point in target_points]
        target_y = [point[1] for point in target_points]
        plt.scatter(target_x, target_y, c='blue', label='Target Points', s=40, edgecolor='black')

        # Plot vectors with validation
        for correspondence in source_correspondence:
            start_point = correspondence[0]
            plt.arrow(start_point[0], start_point[1], correspondence[1], correspondence[2],
                      head_width=2, head_length=4, fc='red', ec='red')

        plt.legend()
        plt.title("Approximation of the intersection position")
        plt.axis('off')
        plt.show()

    def correspondence_position(self, source, target):
        """
        Find the correspondence between two set of intersection points. Any pair of (source, target) point
        is considered corresponding if the distance between the pair is less than half the length
        of perpendicular distance from one line to the other line.

        Assumption: The Delta time is kept small such that the grids will move less than half the
                    distance from one line to the other line

        :param source:      Intersections within the source image
        :param target:      Intersections within the target image
        :return:            A correspondence between the intersection points
        """

        distance_thresh = 0.5 * self._length * np.sin(np.pi / 4)
        target_tmp = copy.deepcopy(target)

        correspondence = []
        for source_point in source:
            dist_source = []
            for target_point in target_tmp:
                dist_source.append(np.sqrt((target_point[0] - source_point[0]) ** 2 +
                                           (target_point[1] - source_point[1]) ** 2))

            # Find the minimum distance for the current source point
            min_index = 0
            if len(dist_source) > 0:
                min_index = np.argmin(dist_source)
                if dist_source[min_index] <= distance_thresh:
                    x_displace = target_tmp[min_index][0] - source_point[0]
                    y_displace = target_tmp[min_index][1] - source_point[1]
                    correspondence.append((source_point, x_displace, y_displace))
                    target_tmp.pop(min_index)
            else:
                correspondence.append((source_point, 0, 0))

        return correspondence

    def match_template_driver(self):
        """
        Credit must be given to Zhirui Gao, et al in their paper "Learning Accurate Template Matching
        with Differentiable Coarse-to-Fine Correspondence Refinement

        The template matching driver function
        """
        # self._polygon = self.set_boundary()
        print("Starting Source Matching")
        self._source_points = match_template(self._raw_source, self._source, self._template,
                                             self._polygon, threshold=self._thresh_source)
        print("Starting Target Matching")
        self._target_points = match_template(self._raw_target, self._target, self._template,
                                             self._polygon, threshold=self._thresh_target)
        displacement_field = self.correspondence_position(self._source_points, self._target_points)

        # Apply Filtering to reduce noise and outliers
        radius = 2 * self._length
        moving_average_validation_arr = moving_average_validation(displacement_field,
                                                                  radius)
        self._displacement = average_filter(moving_average_validation_arr,
                                            radius)
        self.visualize_match(self._source_points, self._target_points, self._displacement)

    def get_boundary(self):
        return self._polygon

    def get_displacement(self):
        return self._displacement


def main():
    parser = argparse.ArgumentParser(description="Template Matching")
    parser.add_argument("source_image", help="Path to the source image")
    parser.add_argument("target_image", help="Path to the target image")
    parser.add_argument("template", help="Path to the template image")
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--use_cython', action='store_true')
    parser.add_argument('--threshold', type=float, default=None)
    parser.add_argument('--nms', type=float, default=0.5)
    args = parser.parse_args()

    """
    Sample visualization of the result of template matching on the image
    """
    matcher = TemplateMatcher(args.source_image, args.target_image, args.template)
    matcher.match_template_driver()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
