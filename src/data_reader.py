import pandas as pd
import numpy as np

df = pd.read_csv('processed_labeled_points.csv')

print(df)

for index, row in df.iterrows():
    label = row['label']
    print(row['point_tensor'])
    point_tensor = np.fromstring(row['point_tensor'][1:-1], sep=' ')
    point_tensor = point_tensor.reshape(160, 160) #reshape tensor to original size
    print("Label:", label)
    print("Point Tensor:")
    print(point_tensor)
