import pandas as pd
import matplotlib.pyplot as plt

points = pd.read_csv('labeled_points.csv', sep='\t', header=None, names=["point_batch1", "point_batch2", "point_batch3", "point_batch4", "point_batch5", "Room+Direction"])

print(points.head())

pb1 = points['point_batch1']

point_set = pb1[0]

point_set = point_set.replace('[(', '')
point_set = point_set.replace(')]', '')

points = point_set.split('), (')
print(points)

xs = []
ys = []
for point in points:
    p = point.split(', ')
    x = p[0]
    xs.append(int(x))
    y = p[1]
    ys.append(int(y))
    
print(xs, ys)

plt.plot(xs, ys)
plt.show()
