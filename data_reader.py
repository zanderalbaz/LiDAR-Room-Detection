import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
points = pd.read_csv('labeled_points.csv', sep='\t', header=None, names=["point_batch1", "point_batch2", "point_batch3", "point_batch4", "point_batch5", "Room+Direction"])

print(points.head())
print(len(points))

pb1 = points['point_batch1']
pb2 = points['point_batch2']
pb3 = points['point_batch3']
pb4 = points['point_batch4']
pb5 = points['point_batch5']

def processSet(pb_set):
    pb_set = pb_set.replace('[(', '')
    pb_set = pb_set.replace(')]', '')
    return pb_set.split('), (')

def processPoints(p_list):
    points = []
    for ps in p_list:
        for p in processPoints_(ps):
            points.append(p)
    return np.asarray(points)

def processPoints_(ps):

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





point_tensors = []
for i in range(len(points)): 
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

    points_list = [points1, points2, points3, points4, points5]
    pts = processPoints(points_list)

    point_tensor = np.zeros((160, 160))
    for pt in pts:
        x, y = pt
        x = int(x)
        y = int(y)
        point_tensor[x, y] = 1
    point_tensors.append(point_tensor)
    
    
labels = points["Room+Direction"]

print(point_tensors)
print(labels)
print(len(point_tensors))
print(len(labels))

# point_batches, label = labeled_data[-1]
# print(points['Room+Direction'].unique())
# print(point_batches.shape)
# print(label)
