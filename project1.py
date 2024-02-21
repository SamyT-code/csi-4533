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

folder_path = './subsequence_cam1'

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


image1 = cv2.imread('image test 1.png')

image2 = cv2.imread('image test 2.png')


histogram_coordinates = []
histogram_labels=["image1FirstFull", "image1FirstUpper",
    "image1SecondFull", "image1SecondUpper",
    "image2FirstFull", "image2FirstUpper",
    "image2SecondFull", "image2SecondUpper",
    "image2ThirdFull", "image2ThirdUpper"]

histogram_names_first = [
    [(image1FirstFull,image1),(image1FirstUpper,image1)],
    [(image2FirstFull,image2), (image2FirstUpper,image2)]
]
histogram_names_second = [
    [(image1SecondFull,image1), (image1SecondUpper,image1)],
   [ (image2SecondFull,image2), (image2SecondUpper,image2)]
]
histogram_names_third = [
    [(image2ThirdFull,image2), (image2ThirdUpper,image2)]]

#extract file name and coordinates from labels.txt
with open('labels.txt', 'r') as file:
    for line in file:
        values = line.strip().split(',')
        filename = values[0]
        coordinates = list(map(int, values[-4:]))
        histogram_coordinates.append((filename, coordinates))


def compare(histogram_names):
    person_max_value = []
    for j in range (len(histogram_names)):
        
        
        
        

        # Compare histograms and find the maximum intersection for each person
        for i, (filename, coordinates) in enumerate(histogram_coordinates):
            if filename not in image_sequence:
                # skip if 
                continue

            image_cam= cv2.imread('./subsequence_cam1/'+filename+'.png')

            intersections = [
                #compare test images with cam images and add to intersections array
                (cv2.compareHist(np.float32(histogram(histogram_names[j][0][0], histogram_names[j][0][1])), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT)),
                (cv2.compareHist(np.float32(histogram(histogram_names[j][1][0], histogram_names[j][1][1])), np.float32(histogram(coordinates, image_cam)), cv2.HISTCMP_INTERSECT)),
                (cv2.compareHist(np.float32(histogram(histogram_names[j][0][0], histogram_names[j][0][1])), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT)),
                (cv2.compareHist(np.float32(histogram(histogram_names[j][1][0], histogram_names[j][1][1])), np.float32(histogramUpper(coordinates, image_cam)), cv2.HISTCMP_INTERSECT))

            ]
            #find max
            max_intersection = max(intersections)

           
            # Store person index, filename, and coordinates of the max intersection value
            person_max_value.append((i, filename, max_intersection, coordinates))

        # Sort the person_max_value list based on max intersection values in descending order
    sorted_person_max_value = sorted(person_max_value, key=lambda x: (x[2], x[3]), reverse=True)

        # Get the top 100 
    top_100_people = sorted_person_max_value[:100]
    print("Top 100 most similar people:")
    for person in top_100_people:
        print("Filename:", person[1], "Coordinates:", person[3])
    return top_100_people


   
    
def show_images_one_by_one(top_people, folder_path):
    window_name = "Image Viewer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    for person in top_people:
        filename, coordinates = person[1],  person[3]
        image_path = os.path.join(folder_path, filename + '.png')
        image = cv2.imread(image_path)

        if image is not None:
            x, y, w, h = coordinates
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            cv2.imshow(window_name, image)
            cv2.waitKey(0)
        else:
            print(f"Failed to load image: {filename}")

    cv2.destroyAllWindows()
#first person
# top_100_people=compare(histogram_names_first)
# show_images_one_by_one(top_100_people, folder_path)
    
#second person
top_100_people=compare(histogram_names_second)
show_images_one_by_one(top_100_people, folder_path)

#third person
# top_100_people=compare(histogram_names_third)
# show_images_one_by_one(top_100_people, folder_path)
