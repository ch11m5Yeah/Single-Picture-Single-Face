from valid import *
from dataset import *

input_shape = (90, 120, 3)
json_dir = r"D:\Dataset\faceV2\Labels"
batch_size = 64

_, valid_dataset, test_dataset = generate_face_datasets(
        json_dir, input_shape=input_shape, batch_size=batch_size,
        train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
    )
run_tflite_inference(
    tflite_model_path="face_detector_best_int8.tflite",
    test_dataset=test_dataset,
    reg_max=7,
    num_samples=20
)
# run_h5_inference(h5_model_path="face_detector_final.h5",
#                  test_dataset=test_dataset,
#                  reg_max=7,
#                  num_samples=5)