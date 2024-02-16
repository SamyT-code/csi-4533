import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('image test 1.png')

if image is None:
    print("Error: Image not loaded.")
    exit()

def histogram(x):
    roi = image[x[1]:x[1]+x[3], x[0]:x[0]+x[2]]
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([roi_gray], [0], None, [256], [0, 256])
    return cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    #plt.plot(hist)
    #plt.title('Histogram')
    #plt.xlabel('Pixel Value')
    #plt.ylabel('Frequency')
    #plt.show()


image1FirstFull=[315,267,125,189]
image1FirstUpper=[315,267,125,189//2]

image1SecondFull=[456,247,124,209]
image1SecondUpper=[456,247,124,209//2]


image2FirstFull=[235,128,70,277]
image2FirstUpper=[235,128,70,277//2]

image2SecondFull=[330,130,55,259]
image2SecondUpper=[330,130,55,259//2]

image2ThirdFull=[381,269,149,186]
image2ThirdUpper=[381,269,149,186//2]


original_histograms = []

original_histograms.append(histogram(image1FirstFull))
original_histograms.append(histogram(image1FirstUpper))
original_histograms.append(histogram(image1SecondFull))
original_histograms.append(histogram(image1SecondUpper))
original_histograms.append(histogram(image2FirstFull))
original_histograms.append(histogram(image2FirstUpper))
original_histograms.append(histogram(image2SecondFull))
original_histograms.append(histogram(image2SecondUpper))
original_histograms.append(histogram(image2ThirdFull))
original_histograms.append(histogram(image2ThirdUpper))


histogram_coordinates = []

histogram_names = [
    "image1FirstFull", "image1FirstUpper",
    "image1SecondFull", "image1SecondUpper",
    "image2FirstFull", "image2FirstUpper",
    "image2SecondFull", "image2SecondUpper",
    "image2ThirdFull", "image2ThirdUpper"
]

with open('labels.txt', 'r') as file:
    
    for line in file:
        values = line.strip().split(',')
        
        
        coordinates = list(map(int, values[-4:]))
        histogram_coordinates.append((values[0], coordinates))

    
        
person_max_value = []



# Compare histograms and find the maximum intersection for each person
for i, (filename, coordinates) in enumerate(histogram_coordinates):
           
           intersections = [
            cv2.compareHist(np.float32(histogram(image1FirstFull)),  np.float32(histogram(coordinates)), cv2.HISTCMP_INTERSECT),
            cv2.compareHist(np.float32(histogram(image1FirstUpper)),  np.float32(histogram(coordinates)), cv2.HISTCMP_INTERSECT),
            cv2.compareHist(np.float32(histogram(image1SecondFull)),  np.float32(histogram(coordinates)), cv2.HISTCMP_INTERSECT),
            cv2.compareHist(np.float32(histogram(image1SecondUpper)),  np.float32(histogram(coordinates)), cv2.HISTCMP_INTERSECT),
            cv2.compareHist(np.float32(histogram(image2FirstFull)),  np.float32(histogram(coordinates)), cv2.HISTCMP_INTERSECT),
            cv2.compareHist(np.float32(histogram(image2FirstUpper)),  np.float32(histogram(coordinates)), cv2.HISTCMP_INTERSECT),
            cv2.compareHist(np.float32(histogram(image2SecondFull)),  np.float32(histogram(coordinates)), cv2.HISTCMP_INTERSECT),
            cv2.compareHist(np.float32(histogram(image2SecondUpper)),  np.float32(histogram(coordinates)), cv2.HISTCMP_INTERSECT),
            cv2.compareHist(np.float32(histogram(image2ThirdFull)),  np.float32(histogram(coordinates)), cv2.HISTCMP_INTERSECT),
            cv2.compareHist(np.float32(histogram(image2ThirdUpper)),  np.float32(histogram(coordinates)), cv2.HISTCMP_INTERSECT),

        ]
           #get max value
           max_intersection = max(intersections)
           max_histogram_name = histogram_names[intersections.index(max_intersection)]
           person_max_value.append((i, filename, max_histogram_name, max_intersection,coordinates))

# Store person index, filename, histogram name, and coordinates
                    
sorted_person_max_value = sorted(person_max_value, key=lambda x: (x[3],x[4]), reverse=True)
 # Sort  in descending order
top_100_people = sorted_person_max_value[:100]

 #  top 100 people 
print("Top 100 most similar people:")
for person in top_100_people:
    print("Filename:", person[1],"Histogram Name:", person[2],"Coordinates:",person[4])

