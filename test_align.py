from cropper import Cropper

# self, image_path, image_size, output_dir, margin, gpu_memory_fraction, detect_multiple_faces
test = Cropper('./detections/2019_08_08 17_01_01_045106.jpg', '112', './aligned', 44, 1.0, True)

test.cropper()