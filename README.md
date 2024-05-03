# Problem Statement

# FHL-LD20 LiDAR Sensor

# LiDAR Pre-Processing


# Dataset
1. located: `src/point_images`
2. About 100-200 images varying across 3 locations (2 rooms and a hallway, for 4 directions each)
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

# LiDAR-Room-Detection
 Take LiDAR data, process it, and run it through a deep CNN to classify different rooms. If we have time, we will port our code onto a raspberry pi, and strap it to a drone. We only plan on getting to the data collection phase of this part of the project, but ideally, we will teach the drone how to avoid objects, and detect much more than just rooms.

 ## TODO: (in somewhat order of relative importance)
 1. ~~Attempt to remove noise from data~~
 ~~2. Collect data and label it
    a. add functionality to the python script to:
     i. collect N data points
     ii. label data points (user input)
     iii. output labeled data points to csv~~
 3. Create CNN to classify data (tune until satisfied with results)
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
