import spe2py as spe

# Load the SPE file(s) using a graphical file dialog
spe_tools = spe.load()

# Display the image of the first frame and region of interest
# Check if multiple files are loaded and handle accordingly
if isinstance(spe_tools, list):
    # If multiple files, display the first frame of the first file
    spe_tools[0].image()
    # Access and print metadata from the first file
    sensor_height = spe_tools[0].file.footer.SpeFormat.Calibrations.SensorInformation['height']
    print(f"Sensor height: {sensor_height}")
else:
    # If only one file, display the first frame
    spe_tools.image()
    # Access and print metadata
    sensor_height = spe_tools.file.footer.SpeFormat.Calibrations.SensorInformation['height']
    print(f"Sensor height: {sensor_height}")