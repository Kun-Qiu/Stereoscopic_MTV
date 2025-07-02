import cv2
import numpy as np
from PIL import Image
from matplotlib.pyplot import imsave


class fileReader:
    # Public Interface

    def __init__(self, img, save_path):
        """
        Default Constructor
        :param img:         The tif image
        :param save_path:   Path in which to save the image if chosen to do so
        """
        self.__img = img
        self.__save_path = save_path
        self.__read_images(save_path)
        self.__sort_array_signal()

    def get_uint8(self, img):
        """
        Get the uint8 image from the object
        :param img: The input image
        :return: uint8 image
        """
        self.__convert2uint8(img)
        return self.__img_uint8

    def average_tif(self, index1=None, index2=None, total_average=False, save_path=None):
        """
        From multiple images obtain an average image of the input images
        :param save_path:       Path to which the image will be saved at
        :param index1:          First image of the series to be averaged
        :param index2:          Second image of the series to be averaged
        :param total_average:   Whether all image should be averaged
        :return: The averaged image of the tif
        """

        if self.__array_length <= 0:
            raise Exception("No images found.")

        images = [img_tuple[0] for img_tuple in self.__tif_images]

        if total_average:
            averaged_img = np.mean(images, axis=0)

        elif index1 is None and index2 is None:
            averaged_img = np.mean(images[-5:], axis=0)

        else:
            if index1 is None or index2 is None:
                raise Exception("Both index1 and index2 should be provided if 'total_average' is not True.")

            if index1 < 0 or index2 < 0 or index1 >= self.__array_length or index2 >= self.__array_length:
                raise Exception("Indices out of range.")

            initial = min(index1, index2)
            end = max(index1, index2) + 1

            averaged_img = np.mean(images[initial:end], axis=0)

        if save_path:
            imsave(self.__save_path, averaged_img.astype(np.float32))

        return averaged_img

    def get_image(self, index):
        """
        Get the image at the specified index
        :param index: The index of the image
        :return: The image at that index
        """

        if index >= self.__array_length:
            raise Exception("Out of Bound Error, {}".format(index))
        return self.__tif_images[index][0]

    def max_intensity_tif(self):
        """
        Return the image with the highest intensity aka the highest mean across all pixels
        :return: Highest intensity image
        """

        return self.get_image(self.__array_length - 1)

    def min_intensity_tif(self):
        """
        Return the image with the lowest intensity aka the lowest mean across all pixels
        :return: Image with the lowest intensity
        """
        return self.get_image(0)

    def filter(self, img, filter_type="sharpen"):
        """
        Proceed with extreme caution, as over filtering can lead to inaccurate results
        :param img: the image in which the filter will be applied on
        :param filter_type: type of filter applied to an image
        :return: the filtered image
        """

        if filter_type not in self.__sobel_operators:
            return cv2.filter2D(img, -1, self.__sobel_operators["sharpen"])

        return cv2.filter2D(img, -1, self.__sobel_operators[filter_type])

    # Private Interface

    # Number of img in the tif file
    __array_length = 0
    # Arrays of arrays that contains [image, mean intensity]
    __tif_images = []
    __tif_sorted = False

    # Sobel Operators for filtering image
    __sobel_operators = {
        "horizontal": np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]]),
        "vertical": np.array([[1, 0, -1],
                              [2, 0, -2],
                              [1, 0, -1]]),
        "left_diagonal": np.array([[1, -1, -1],
                                   [-1, 1, -1],
                                   [-1, -1, 1]]),
        "right_diagonal": np.array([[-1, -1, 1],
                                    [-1, 1, -1],
                                    [1, -1, -1]]),
        "edge_detection": np.array([[-1, -1, -1],
                                    [-1, 8, -1],
                                    [-1, -1, -1]]),
        "sharpen": np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]]),
        "box_blur": (1 / 9.0) * np.array([[1., 1., 1.],
                                          [1., 1., 1.],
                                          [1., 1., 1.]]),
        "gaussian_blur": (1 / 16.0) * np.array([[1., 2., 1.],
                                                [2., 4., 2.],
                                                [1., 2., 1.]])
    }

    def __read_images(self, path):
        """
        Access the image in tif and store in an array of images

        :param path : Path to where all the files are saved
        :return     : None
        """

        try:
            img = Image.open(self.__img)

            # Ensure it's a tif file
            if img.format != 'TIFF':
                raise Exception("File is not a TIFF image.")

            self.__tif_images.clear()  # Clear previous data if any
            self.__array_length = 0  # Reset array length

            for i in range(img.n_frames):
                img.seek(i)

                # Convert to NumPy array and possibly change dtype
                frame_array = np.array(img)
                self.__tif_images.append(frame_array)
                self.__array_length += 1

                if path is not None:
                    imsave(f'frame_{i}.png', frame_array)

        except IOError:
            raise Exception("Could not open or read the image file.")
        except Exception as e:
            raise Exception(f"An error occurred: {str(e)}")

    def __sort_array_signal(self):
        """
        Sorts the internal class array of images in tif based on the intensity of the image
        :return: None
        """

        new_array = []
        for img in self.__tif_images:
            new_array.append([img, np.mean(img)])

        # Sort images based on the mean intensity
        new_array = sorted(new_array, key=lambda x: x[1])
        self.__tif_sorted = True
        self.__tif_images = new_array

    def __convert2uint8(self, img):
        """
        Convert a np.array consisting of float32 or 64 to uint8

        :param img: input image to be converted
        :return: None
        """
        try:
            imin = img.min()
            imax = img.max()

            copy_img = np.array(img)
            # Conversion from image data type to uint8
            a = (255 - 0) / (imax - imin)
            b = 255 - a * imax
            new_img = (a * copy_img + b).astype(np.uint8)
            self.__img_uint8 = new_img

        except Exception as e:
            raise Exception("Failed to convert image to uint8. Original error: {}".format(e))
