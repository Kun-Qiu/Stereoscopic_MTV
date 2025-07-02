import os
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QFileDialog, QLabel, \
    QComboBox, QTabWidget, QScrollArea, QSizePolicy, QLineEdit, QMessageBox

import src.calibration_transform_coefficient as cd


class FileSelectorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Main layout
        main_layout = QVBoxLayout()

        # Create tab widget
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        # Create calibration tab
        self.calibration_tab = QWidget()
        self.initCalibrationTab()
        self.tabs.addTab(self.calibration_tab, "Calibration")

        # Create 3D velocity measurement tab
        self.velocity_tab = QWidget()
        self.initVelocityTab()
        self.tabs.addTab(self.velocity_tab, "3D Velocity Measurement")

        # Set main layout
        self.setLayout(main_layout)

        # Set window title and size
        self.setWindowTitle("File Selector with Tabs")
        self.resize(1920, 1080)

    def initCalibrationTab(self):
        # Layout for calibration tab
        calibration_layout = QVBoxLayout()

        # Left image folder selection section
        left_folder_layout = QHBoxLayout()
        left_folder_label = QLabel("Select Left Image Folder:")
        left_folder_layout.addWidget(left_folder_label)

        self.left_image_folder_combo = QComboBox()
        self.left_image_folder_combo.setMinimumWidth(300)
        self.left_image_folder_combo.setMaximumWidth(500)
        left_folder_layout.addWidget(self.left_image_folder_combo)

        left_folder_button = QPushButton("Browse")
        left_folder_button.clicked.connect(self.select_left_image_folder)
        left_folder_layout.addWidget(left_folder_button)

        calibration_layout.addLayout(left_folder_layout)

        # Right image folder selection section
        right_folder_layout = QHBoxLayout()
        right_folder_label = QLabel("Select Right Image Folder:")
        right_folder_layout.addWidget(right_folder_label)

        self.right_image_folder_combo = QComboBox()
        self.right_image_folder_combo.setMinimumWidth(300)
        self.right_image_folder_combo.setMaximumWidth(500)
        right_folder_layout.addWidget(self.right_image_folder_combo)

        right_folder_button = QPushButton("Browse")
        right_folder_button.clicked.connect(self.select_right_image_folder)
        right_folder_layout.addWidget(right_folder_button)

        calibration_layout.addLayout(right_folder_layout)

        # Output location section
        output_layout = QHBoxLayout()
        output_label = QLabel("Select Output Location:")
        output_layout.addWidget(output_label)

        self.calibration_output_combo = QComboBox()
        self.calibration_output_combo.setMinimumWidth(300)
        self.calibration_output_combo.setMaximumWidth(500)
        output_layout.addWidget(self.calibration_output_combo)

        output_button = QPushButton("Browse")
        output_button.clicked.connect(self.select_calibration_output_folder)
        output_layout.addWidget(output_button)

        output_layout.addStretch()
        calibration_layout.addLayout(output_layout)

        # Grid parameters section
        grid_parameters_layout = QVBoxLayout()

        # Number of Grid
        number_of_grid_layout = QHBoxLayout()
        number_of_grid_label = QLabel("Number of Grid:")
        number_of_grid_layout.addWidget(number_of_grid_label)

        self.number_of_grid_input = QLineEdit()
        self.number_of_grid_input.setPlaceholderText("Enter number of grid")
        number_of_grid_layout.addWidget(self.number_of_grid_input)

        grid_parameters_layout.addLayout(number_of_grid_layout)

        # Dimension of Grid
        dimension_of_grid_layout = QHBoxLayout()
        dimension_of_grid_label = QLabel("Dimension of Grid (W, H):")
        dimension_of_grid_layout.addWidget(dimension_of_grid_label)

        self.dimension_of_grid_input = QLineEdit()
        self.dimension_of_grid_input.setPlaceholderText("Enter dimension of grid")
        dimension_of_grid_layout.addWidget(self.dimension_of_grid_input)

        grid_parameters_layout.addLayout(dimension_of_grid_layout)

        calibration_layout.addLayout(grid_parameters_layout)

        # Section for Calculate Parameters Button
        calculate_parameters_layout = QHBoxLayout()
        calculate_parameters_button = QPushButton("Calculate Parameters")
        calculate_parameters_button.clicked.connect(self.calculate_parameters)
        calculate_parameters_button.setMinimumSize(200, 60)  # Set minimum width and height
        calculate_parameters_button.setStyleSheet("font-size: 20px;")  # Apply CSS styling
        calculate_parameters_layout.addWidget(calculate_parameters_button, alignment=Qt.AlignCenter)

        calibration_layout.addLayout(calculate_parameters_layout)

        # Image preview section
        preview_layout = QVBoxLayout()
        preview_title = QLabel("Preview:")
        preview_title.setStyleSheet("font-weight: bold; font-size: 26px;")
        preview_layout.addWidget(preview_title)

        # Create layout for image previews
        image_preview_layout = QHBoxLayout()

        # Left image preview
        left_layout = QVBoxLayout()
        left_label = QLabel("Left Image:")
        left_layout.addWidget(left_label)

        self.left_image_label = QLabel()
        self.left_image_label.setAlignment(Qt.AlignCenter)
        self.left_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_area_left = QScrollArea()
        scroll_area_left.setWidgetResizable(True)
        scroll_area_left.setWidget(self.left_image_label)
        left_layout.addWidget(scroll_area_left)

        image_preview_layout.addLayout(left_layout)

        # Right image preview
        right_layout = QVBoxLayout()
        right_label = QLabel("Right Image:")
        right_layout.addWidget(right_label)

        self.right_image_label = QLabel()
        self.right_image_label.setAlignment(Qt.AlignCenter)
        self.right_image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_area_right = QScrollArea()
        scroll_area_right.setWidgetResizable(True)
        scroll_area_right.setWidget(self.right_image_label)
        right_layout.addWidget(scroll_area_right)

        image_preview_layout.addLayout(right_layout)

        preview_layout.addLayout(image_preview_layout)
        calibration_layout.addLayout(preview_layout)

        self.calibration_tab.setLayout(calibration_layout)

    def initVelocityTab(self):
        # Layout for velocity measurement tab
        velocity_layout = QVBoxLayout()

        # Folder selection section
        folder_layout = QHBoxLayout()
        folder_label = QLabel('Select Velocity Measurement Folder:')
        folder_layout.addWidget(folder_label)

        self.velocity_folder_combo = QComboBox()
        self.velocity_folder_combo.setMinimumWidth(300)
        self.velocity_folder_combo.setMaximumWidth(500)
        folder_layout.addWidget(self.velocity_folder_combo)

        folder_button = QPushButton('Browse Velocity Measurement Folder')
        folder_button.clicked.connect(self.select_velocity_folder)
        folder_layout.addWidget(folder_button)

        folder_layout.addStretch()
        velocity_layout.addLayout(folder_layout)

        # Add velocity measurement-specific widgets here

        self.velocity_tab.setLayout(velocity_layout)

    def select_left_image_folder(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Left Image Folder", options=options)
        if directory:
            self.left_image_folder_combo.addItem(directory)
            self.left_image_folder_combo.setCurrentText(directory)
            self.update_image_previews(directory, side="left")

    def select_right_image_folder(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Right Image Folder", options=options)
        if directory:
            self.right_image_folder_combo.addItem(directory)
            self.right_image_folder_combo.setCurrentText(directory)
            self.update_image_previews(directory, side="right")

    def select_calibration_output_folder(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Output Location", options=options)
        if directory:
            self.calibration_output_combo.addItem(directory)
            self.calibration_output_combo.setCurrentText(directory)

    def select_velocity_folder(self):
        options = QFileDialog.Options()
        directory = QFileDialog.getExistingDirectory(self, "Select Velocity Measurement Folder", options=options)
        if directory:
            self.velocity_folder_combo.addItem(directory)
            self.velocity_folder_combo.setCurrentText(directory)

    def update_image_previews(self, directory, side):
        # Clear previous images
        if side == "left":
            self.left_image_label.clear()
            # Find and display the first image file in the selected directory
            image_files = [f for f in os.listdir(directory) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if image_files:
                left_image_path = os.path.join(directory, image_files[0])
                # print(f"Left Image Path: {left_image_path}")  # Debug print
                if os.path.isfile(left_image_path):
                    pixmap_left = QPixmap(left_image_path)
                    self.left_image_label.setPixmap(
                        pixmap_left.scaled(self.left_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
                else:
                    print(f"Error: File does not exist: {left_image_path}")  # Error print
            else:
                print("No image files found in left folder.")  # Debug print

        elif side == "right":
            self.right_image_label.clear()
            # Find and display the first image file in the selected directory
            image_files = [f for f in os.listdir(directory) if
                           f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
            if image_files:
                right_image_path = os.path.join(directory, image_files[0])
                # print(f"Right Image Path: {right_image_path}")  # Debug print
                if os.path.isfile(right_image_path):
                    pixmap_right = QPixmap(right_image_path)
                    self.right_image_label.setPixmap(
                        pixmap_right.scaled(self.right_image_label.size(), Qt.AspectRatioMode.KeepAspectRatio))
                else:
                    print(f"Error: File does not exist: {right_image_path}")  # Error print
            else:
                print("No image files found in right folder.")  # Debug print

    def calculate_parameters(self):
        try:
            # Retrieve the output location
            output_location = self.calibration_output_combo.currentText()
            left_calibrate = self.left_image_folder_combo.currentText()
            right_calibrate = self.right_image_folder_combo.currentText()
            num_grid = int(self.number_of_grid_input.text())

            # Parse dimension of grid
            dim_grid = self.dimension_of_grid_input.text()
            dim = dim_grid.strip("()").split(", ")
            W, H = map(float, dim)

            print(f"Left Calibration Folder: {left_calibrate}")
            print(f"Right Calibration Folder: {right_calibrate}")
            print(f"Output Location: {output_location}")
            print(f"Number of Grid: {num_grid}")
            print(f"Dimension of Grid: {dim_grid}")

            # Attempt to create the corner detection object
            try:
                corner_detection_object = cd.CalibrationPointDetector(
                    left_calibrate, right_calibrate, output_location, num_grid, (W, H)
                )
            except Exception as e:
                print(f"Error initializing CalibrationPointDetector: {e}")
                self.show_error_message(f"Failed to initialize calibration: {str(e)}")
                return

            # Run the calibration process
            try:
                print("####     Begin Calculating Parameters for Soloff Polynomial...     ####")
                corner_detection_object.run_calibration()
                print("####     Calibration completed successfully.                       ####")
            except Exception as e:
                print(f"Error during calibration: {e}")
                self.show_error_message(f"Failed to run calibration: {str(e)}")

        except Exception as e:
            print(f"Unexpected error in calculate_parameters: {e}")
            self.show_error_message(f"An unexpected error occurred: {str(e)}")

    def show_error_message(self, message):
        error_dialog = QMessageBox(self)
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText(message)
        error_dialog.setWindowTitle("Error")
        error_dialog.exec_()


def main():
    app = QApplication(sys.argv)
    mainWindow = FileSelectorApp()
    mainWindow.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
