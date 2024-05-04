# Problem Statement
We wanted to create a neural network that classifies objects based off of processed LiDAR data. Our main goal is to add thoughts and research, revolutionizing how we understand and interact with the physical world using LiDAR technology and AI, specifically in geospatial analysis. 


The initial scope of this project was to use a 3D point cloud from our LiDAR sensor to classify different types of terrain. After realizing this goal was far-fetched for our timeframe, we reduced the problem size to use a 2D point cloud trying to classify which room we are in and what direction we are facing. We want to be able to feed our model new data, and have it classify the room and direction it is in somewhat decently.

# FHL-LD20 LiDAR Sensor
Link: https://www.amazon.com/WayPonDEV-360-Degree-Lidar-Sensor/dp/B0C1C4VW47/ref=asc_df_B0C1C4VW47/?tag=hyprod-20&linkCode=df0&hvadid=663203410548&hvpos=&hvnetw=g&hvrand=5461963860498555154&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=9029132&hvtargid=pla-2187194875134&psc=1&mcid=d87d21ec20b33975b300b3cd7cde4621


![Lidar Sensor](https://m.media-amazon.com/images/I/41MjeVHQy9L._SX425_.jpg)


SPECS:
+ Distance Range: 8m
+ Ranging Accuracy: 1 deg, 2300Hz
+ 360 deg scans
+ Anti-ambient light: 30000Lux


Sensor Documentation: https://wiki.youyeetoo.com/en/Lidar/LD20


# LiDAR Pre-Processing
For this project, we decided to create our own dataset for this project. This part took quite a long time since we had issues getting our sensor to read data correctly. We needed to parse our sensor data directly from the serial port, then do some math to figure out the angle of each distance in the packet. After only selecting points with high intensity (sensor is confident it is correct). We then took the angle and distance, and converted it into an x,y point.


After this, we collected 500 of these points, quantized them into bins of 10, removed duplicates, then created a 160x160 tensor that represents the 8 meter radius the sensor rated for. Before inputting the points into the tensor we applied a gaussian filter to the points to reduce noise. We also applied DBSCAN, and removed points it declared as noise. Each point in the tensor is either a zero if no point was collected at that location, or a 1 if there was a point at that location. These tensors were then stored as images to make training easier.



# Dataset (cleaned)
1. located: `src/point_images`
2. About 100-200 images (160x160) varying across 3 locations (2 rooms and a hallway) for 4 directions each
3. LABELS:
   + HALLWAY1-NORTH
   + HALLWAY1-EAST
   + HALLWAY1-SOUTH
   + HALLWAY1-WEST
   + RADY129-NORTH
   + RADY129-EAST
   + RADY129-SOUTH
   + RADY129-WEST
   + RADY131-NORTH
   + RADY131-EAST
   + RADY131-SOUTH
   + RADY131-WEST
  
# Models
We created quite a few models. Some of our earlier iterations and the current model we used are located in: `src/models.py`


We are running our models using `src/main.py`

# Future Directions
Now that we have a decent model working, our next steps are to refine our current model. We may also play with our image size, as it seems smaller images are easier to classify. After we are satisfied with our results, we will move on to quantizing and pruning. Once we have a fine-tuned model, we plan on moving our neural network onto a Raspberry Pi, which we want to strap onto a small robot or drone. We can then see if our model works real time autonomously.


Ideally once we have a good proof of concept, we will implement some mechanism to capture 3D LiDAR data on the drone/robot. All we should need is a precise servo or stepper motor to rotate our sensor. With that data, we can move on to classifying real objects, making 3D scans of rooms, and much more. 

 ## TODO: (in somewhat order of relative importance)
 1. ~~Attempt to remove noise from data~~
 ~~2. Collect data and label it
    a. add functionality to the python script to:
     i. collect N data points
     ii. label data points (user input)
     iii. output labeled data points to csv~~
~~3. Create CNN to classify data (tune until satisfied with results)~~
    ~~4. Create graphics comparing ground truth to predictions (for presentation)~~
 6. Quantize and Prune CNN
 7. Tune pruned and quantized models (until satisfied)
 8. Create as much documentation as we can at this stage. (as if we were done with the project here)
 9. Create a scrpit to run tello drone (more realistically, go find one we already have)
 10. Copy sensor processing code onto Raspberry Pi (debug any version issues and get it running)
 11. Get sensor data from Raspberry Pi to Host Machine (we can likely set up a web socket or something)
 13. Create dashboard to control drone (can use flask app from earlier project (already has movement done)
 14. Gather sensor data while flying the drone around (after we strap the sensor to it somehow...?)
 15. Re-train/update model to work with drone data (if our first data collection effort goes well we should not need to do this)
 16. Display live location on the dashboard (from classification)
 17. Display live lidar data (plotted) on dashboard (from pi)
 18. Update documentation to include new elements of the project
