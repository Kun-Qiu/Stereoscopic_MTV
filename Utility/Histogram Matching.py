from skimage import exposure, io
import numpy as np

source_path = '../Data/Source/source_25.png'
target_path = '../Data/Target/target_25.png'
template_path = '../Data/Template/template25.png'

# Load images
raw_source = io.imread(source_path)
raw_target = io.imread(target_path)
raw_template = io.imread(template_path)

# Perform histogram matching
matched_image = exposure.match_histograms(raw_target, raw_template)
matched_image = exposure.rescale_intensity(matched_image, out_range=np.uint8).astype(np.uint8)
# Display or save the matched image
io.imshow(matched_image)
io.show()

# Optionally, save the matched image
io.imsave('../Data/Target/target_25_new.png', matched_image)
