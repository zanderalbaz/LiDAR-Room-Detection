import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
points = pd.read_csv('labeled_points.csv', sep='\t', header=None, names=["point_batch1", "point_batch2", "point_batch3", "point_batch4", "point_batch5", "Room+Direction"])

print(points.head())


def processSet(pb_set):
    pb_set = pb_set.replace('[(', '')
    pb_set = pb_set.replace(')]', '')
    return pb_set.split('), (')

def processPoints(p_list): #Process all points
    point_sets = []
    for ps in p_list:
        points = []
        for p in processPoints_(ps): #call helper funcion
            points.append(p)
        point_sets.append(points) #re-seperate into 5 batches
    return np.asarray(point_sets)

def processPoints_(ps): #process points to make point tensors smaller
    points = []
    for point in ps:
        p = point.split(', ')
        x = int(float(p[0])) #convert to float then int to convert string to int
        x = x - (x % 10) #bin points into intervals of 10
        x = x + 800 #make everything positive
        x = x / 10 #reduce maximum tensor size from 800x800 to 80x80
        #do same to y
        y = int(float(p[1]))
        y = y - (y % 10)
        y = y + 800
        y = y / 10
        
        points.append((x,y))
    
    points = list(set(points))
    return np.asarray(points)




tensor_labels = []
point_tensors = []
labels = points["Room+Direction"]

def pointsToTensors():
    pb1 = points['point_batch1']
    pb2 = points['point_batch2']
    pb3 = points['point_batch3']
    pb4 = points['point_batch4']
    pb5 = points['point_batch5']

    for i in range(len(points)): # We collected 5 batches of 500 points every collection interval
        pb_set1 = pb1[i]
        pb_set2 = pb2[i]
        pb_set3 = pb3[i]
        pb_set4 = pb4[i]
        pb_set5 = pb5[i]

        points1 = processSet(pb_set1)
        points2 = processSet(pb_set2)
        points3 = processSet(pb_set3)
        points4 = processSet(pb_set4)
        points5 = processSet(pb_set5)

        points_list = [points1]
        pt_sets = processPoints(points_list)
        label = labels[i]
        for pts in pt_sets:
            point_tensor = np.zeros((160, 160))
            for pt in pts:
                x, y = pt
                x = int(x)
                y = int(y)
                point_tensor[x, y] = 1
            point_tensors.append(point_tensor)
            tensor_labels.append(label)
    return point_tensors, tensor_labels
        
        
def outputPointTensorsAsImages():
    #Create label map for outputing the write index for each image in each folder
    label_counts = {}
    for label in labels.unique():
        label_counts[label] = 1
    point_tensors, tensor_labels = pointsToTensors()
    for i in range(len(tensor_labels)):
        if i % 100 == 0:
            print(i)
        label = tensor_labels[i]
        plt.imshow(point_tensors[i], cmap="Greys")
        plt.axis('off')
        plt.savefig(f"point_images/{label}/{label_counts[label]:04}.jpg",  bbox_inches='tight', pad_inches=0)
        label_counts[label] += 1

if __name__ == '__main__':
    outputPointTensorsAsImages()


