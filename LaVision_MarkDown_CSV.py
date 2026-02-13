import xml.etree.ElementTree as ET
import csv


class StereoCalibrationXML:
    def __init__(self, xml_path):
        self.xml_path = xml_path
        self.cam1_data = None
        self.cam2_data = None

    def _parse_stereo_xml(self):
        """
        Parse stereo XML and store camera data as:
        {(x, y, z): (X, Y)}
        """

        tree = ET.parse(self.xml_path)
        root = tree.getroot()

        cameras = root.findall(".//Camera")

        if len(cameras) != 2:
            raise ValueError("Expected exactly 2 Camera blocks in XML.")

        cam_dicts = []

        for cam in cameras:
            data = {}

            for mark in cam.findall(".//Mark"):
                raw = mark.find("RawPos")
                world = mark.find("WorldPos")

                if raw is None or world is None:
                    continue

                X = float(raw.attrib["x"])
                Y = float(raw.attrib["y"])

                x = float(world.attrib["x"])
                y = float(world.attrib["y"])
                z = float(world.attrib["z"])

                data[(x, y, z)] = (X, Y)

            cam_dicts.append(data)

        self.cam1_data, self.cam2_data = cam_dicts

    def write_matched_csv(self, output_cam1, output_cam2):
        """
        Write two CSVs containing only common (x, y, z),
        ensuring strict stereo correspondence.
        """

        # Parse if not already parsed
        if self.cam1_data is None or self.cam2_data is None:
            self._parse_stereo_xml()

        # Intersection of world coordinates
        common_world = sorted(
            set(self.cam1_data.keys()) & set(self.cam2_data.keys())
        )

        print(f"Camera 1 total points: {len(self.cam1_data)}")
        print(f"Camera 2 total points: {len(self.cam2_data)}")
        print(f"Common matched points: {len(common_world)}")

        if len(common_world) == 0:
            raise ValueError("No common world points found between cameras.")

        fieldnames = ["X", "Y", "x", "y", "z"]

        with open(output_cam1, "w", newline="") as f1, \
             open(output_cam2, "w", newline="") as f2:

            writer1 = csv.DictWriter(f1, fieldnames=fieldnames)
            writer2 = csv.DictWriter(f2, fieldnames=fieldnames)

            writer1.writeheader()
            writer2.writeheader()

            for (x, y, z) in common_world:
                X1, Y1 = self.cam1_data[(x, y, z)]
                X2, Y2 = self.cam2_data[(x, y, z)]

                writer1.writerow({"X": X1, "Y": Y1, "x": x, "y": y, "z": z})
                writer2.writerow({"X": X2, "Y": Y2, "x": x, "y": y, "z": z})

        print("Matched stereo CSV files written successfully.")

if __name__ == "__main__":

    # ======== USER INPUT ========
    xml_file = r"C:\Users\Kun Qiu\Desktop\3D_Stereo_Exp\Calibration_Data\MarkPositionTable.xml"
    output_cam1 = r"C:\Users\Kun Qiu\Desktop\3D_Stereo_Exp\Calibration_Data\cam1_pts.csv"
    output_cam2 = r"C:\Users\Kun Qiu\Desktop\3D_Stereo_Exp\Calibration_Data\cam2_pts.csv"
    # ============================

    try:
        converter = StereoCalibrationXML(xml_file)
        converter.write_matched_csv(output_cam1, output_cam2)

        print("Stereo calibration export completed successfully.")

    except Exception as e:
        print("Error during stereo calibration export:")
        print(e)
