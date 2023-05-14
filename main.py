import cv2
import os
import json
import numpy as np
from data_utils import load_test_images, load_training_images
from eigenfaces import Eigenfaces

# Load parameters from JSON file
with open('parameters.json') as json_file:
    parameters = json.load(json_file)

# Extract parameters from the JSON object
n_subjects = parameters['n_subjects']
n_training_images = parameters['n_training_images']
n_test_images = parameters['n_test_images']
img_ext = parameters['img_ext']
training_data_path = parameters['training_data_path']
test_data_path = parameters['test_data_path']
n_eigenfaces = parameters['n_eigenfaces']
threshold = parameters['threshold']

# Load training and test data
# Load training and test data
training_images, training_labels = load_training_images(training_data_path, n_subjects, n_training_images, img_ext)
test_images, test_labels = load_test_images(test_data_path, n_subjects, img_ext)

# Train Eigenfaces model
training_data_flat = training_images.reshape(training_images.shape[0], -1)
eigenfaces = Eigenfaces(n_components=n_eigenfaces)
eigenfaces.train(training_data_flat)

# Test Eigenfaces model
recognition_rate = 0
correct_count = 0
total_count = test_images.shape[0]

for i, test_img in enumerate(test_images):
    test_img_flat = test_img.flatten()
    projected_test_face = eigenfaces.project(test_img_flat)
    recognized_subject_id, _ = eigenfaces.recognize(projected_test_face, threshold=threshold)

    if recognized_subject_id is not None:
        recognized_subject_id = training_labels[recognized_subject_id]
        recognition_rate += 1

        if recognized_subject_id == test_labels[i]:
            correct_count += 1

    print(f"Test Image {i + 1}: Recognized as s{recognized_subject_id}")

recognition_rate = (recognition_rate / total_count) * 100
accuracy = (correct_count / total_count) * 100
print(f"Recognition Rate: {recognition_rate:.2f}%") #any recognition
print(f"Accuracy: {accuracy:.2f}%") #correct recognition


def find_similar_image(input_image_location):
    input_image = cv2.imread(input_image_location, cv2.IMREAD_GRAYSCALE)
    if input_image.shape[0] > 100 or input_image.shape[1] > 100:
        input_image = cv2.resize(input_image, (100, 100))
    
    input_image_flat = input_image.flatten()
    projected_input_face = eigenfaces.project(input_image_flat)
    
    distances = []
    for i, training_face in enumerate(training_data_flat):
        projected_training_face = eigenfaces.project(training_face)
        distance = np.linalg.norm(projected_input_face - projected_training_face)
        distances.append(distance)
    
    min_distance = np.min(distances)
    if min_distance <= threshold:
        closest_image_idx = np.argmin(distances)
        closest_image_label = training_labels[closest_image_idx]
        closest_image_path = os.path.join(training_data_path, f's{closest_image_label}', f'{closest_image_label}{img_ext}')
        print(f"Image found in training data. Closest match: s{closest_image_label}, opening...")
        cv2.imshow("Closest Image", cv2.imread(closest_image_path))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Image not found in training data")

# Specify the location of the input image
input_image_location = 'test_image.pgm'

# Find the most similar image
choice = input("\nDo you want to run facial recognition for an input image? [y/n]\nif yes, before writing the reply put a 'test_image.pgm' in the root directory.\n")
if choice == "y":
    try:
        find_similar_image(input_image_location)
    except:
        print("Image not found, please check the file format and name.")
else:
    exit()
