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
        points.append(processPoints_(ps))
    return np.asarray(points)

def processPoints_(ps):
    xs = []
    ys = []
    for point in ps:
        p = point.split(', ')
        x = p[0]
        xs.append(int(float(x))) #convert to float then int to catch points that were not quantized
        y = p[1]
        ys.append(int(float(y))) 
        
    points = []
    points.append(np.array(xs))
    points.append(np.array(ys))
    return np.asarray(points)

labeled_data = []
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
    point_batches = processPoints(points_list)
    labels = points["Room+Direction"]

    labeled_data.append((point_batches, labels[i]))

point_batches, label = labeled_data[-1]
print(points['Room+Direction'].unique())
print(point_batches.shape)
print(label)
