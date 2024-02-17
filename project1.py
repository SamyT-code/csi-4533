import cv2
import numpy as np
import os

def histogram(x, image):
    roi = image[x[1]:x[1]+x[3], x[0]:x[0]+x[2]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    return hist
    #return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
#function to calculate upper part of cam images
def histogramUpper(x, image):
    roi = image[x[1]:x[1]+x[3]//2, x[0]:x[0]+x[2]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    #return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist
    #return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

def load_images_from_folder(folder):
    images = {}
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images[os.path.splitext(filename)[0]] = img
    return images

folder_path = 'C:/Users/user/Desktop/csi-4533/subsequence_cam1'

# Load images from the folder
image_sequence = load_images_from_folder(folder_path)

image1FirstFull = [315, 267, 125, 189]
image1FirstUpper = [315, 267, 125, 189//2]

image1SecondFull = [456, 247, 124, 209]
image1SecondUpper = [456, 247, 124, 209//2]

image2FirstFull = [235, 128, 70, 277]
image2FirstUpper = [235, 128, 70, 277//2]

image2SecondFull = [330, 130, 55, 259]
image2SecondUpper = [330, 130, 55, 259//2]

image2ThirdFull = [381, 269, 149, 186]
image2ThirdUpper = [381, 269, 149, 186//2]

original_histograms = []
image1 = cv2.imread('image test 1.png')
original_histograms.append(histogram(image1FirstFull, image1))
original_histograms.append(histogram(image1FirstUpper, image1))
original_histograms.append(histogram(image1SecondFull, image1))
original_histograms.append(histogram(image1SecondUpper, image1))
image2 = cv2.imread('image test 2.png')
original_histograms.append(histogram(image2FirstFull, image2))
original_histograms.append(histogram(image2FirstUpper, image2))
original_histograms.append(histogram(image2SecondFull, image2))
original_histograms.append(histogram(image2SecondUpper, image2))
original_histograms.append(histogram(image2ThirdFull, image2))
original_histograms.append(histogram(image2ThirdUpper, image2))


histogram_coordinates = []

histogram_names_first = [
    "image1FirstFull", "image1FirstUpper",
    "image1SecondFull", "image1SecondUpper",
    "image2FirstFull", "image2FirstUpper",
    "image2SecondFull", "image2SecondUpper",
    "image2ThirdFull", "image2ThirdUpper"
]
#extract file name and coordinates from labels.txt
with open('labels.txt', 'r') as file:
    for line in file:
        values = line.strip().split(',')
        filename = values[0]
        coordinates = list(map(int, values[-4:]))
        histogram_coordinates.append((filename, coordinates))

person_max_value = []

# Compare histograms and find the maximum intersection for each person
for i, (filename, coordinates) in enumerate(histogram_coordinates):
    if filename not in image_sequence:
        # skip if 
        continue

    image_cam= cv2.imread('C:/Users/user/Desktop/csi-4533/subsequence_cam1/'+filename+'.png')

    intersections = [
        #compare test images with cam image full
        cv2.compareHist(np.float32(histogram(image1FirstFull, image1)), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image1FirstUpper, image1)), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image1SecondFull, image1)), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image1SecondUpper, image1)), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2FirstFull, image2)), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2FirstUpper, image2)), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2SecondFull, image2)), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2SecondUpper, image2)), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2ThirdFull, image2)), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2ThirdUpper, image2)), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        #cmpare test images with cam image upper
        cv2.compareHist(np.float32(histogram(image1FirstFull, image1)), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image1FirstUpper, image1)), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image1SecondFull, image1)), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image1SecondUpper, image1)), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2FirstFull, image2)), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2FirstUpper, image2)), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2SecondFull, image2)), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2SecondUpper, image2)), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2ThirdFull, image2)), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
        cv2.compareHist(np.float32(histogram(image2ThirdUpper, image2)), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT),
    ]
    max_intersection = max(intersections)

    # get max value after comparaisons
    if intersections:
        max_histogram_name = histogram_names_first[intersections.index(max_intersection) % len(histogram_names_first)]
    else:
        max_histogram_name = "No valid intersection"
    #print(f"Processed image: {filename}, Max intersection: {max_intersection}, Max histogram name: {max_histogram_name}")



    # Store person index, filename, histogram name, and coordinates of the max intersection value
    person_max_value.append((i, filename, max_histogram_name, max_intersection, coordinates))

# Sort the person_max_value list based on max intersection values in descending order
sorted_person_max_value = sorted(person_max_value, key=lambda x: (x[3], x[4]), reverse=True)

# Get the top 100 
top_100_people = sorted_person_max_value[:100]


print("Top 100 most similar people:")
for person in top_100_people:
    print("Filename:", person[1], "Histogram Name:", person[2], "Coordinates:", person[4])
