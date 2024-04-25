import sys 
import threading
from serial import Serial
import pygame_widgets
import pygame
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.cluster import DBSCAN
from scipy.ndimage import gaussian_filter
import numpy as np

pygame.init()

font = pygame.font.SysFont("calibri", 24)

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 400

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("LiDAR Data")
slider = Slider(screen, 200, 200, 800, 40, min=0, max=1, step=0.01)
output = TextBox(screen, 475, 200, 50, 50, fontSize=30)

lidar_usb = Serial('COM5', 230400, timeout=1)

# Global variables for data
radar_speed = 0
starting_angle = 0 
distance = 0
signal_strength = 0
end_angle = 0
timestamp = 0



MEASURE_FREQ = 4000 #samples / second

gdist = 0
gangle = 0
gint = 0

NUM_POINTS = 500


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x_points = [0] * NUM_POINTS
y_points = [0] * NUM_POINTS



scatter = ax.scatter([], [], s=10)
AX_DIM = 800
ax.set_ylim([-AX_DIM, AX_DIM])
ax.set_xlim([-AX_DIM, AX_DIM])

def polarToCart(R, deg):
    return ((R*math.cos((deg*math.pi)/180)), R*math.sin((deg*math.pi)/180))

def animate(i):
    global x_points, y_points
    scatter.set_offsets(list(zip(x_points, y_points)))

    return scatter,
def pygameDisplay():
    global gint, gdist
    try:
        while True:
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.display.quit()
                    pygame.quit()
                    sys.exit()
            screen.fill(WHITE)
            text_surface = font.render(f"Radar speed: {radar_speed} deg/s", True, BLACK)
            screen.blit(text_surface, (20, 20))
            text_surface = font.render(f"Starting angle: {starting_angle} degrees", True, BLACK)
            screen.blit(text_surface, (20, 50))
            text_surface = font.render(f"Distance: {gdist} mm", True, BLACK)
            screen.blit(text_surface, (20, 80))
            text_surface = font.render(f"Signal Strength: {gint}", True, BLACK)
            screen.blit(text_surface, (20, 110))
            text_surface = font.render(f"End angle: {end_angle} degrees", True, BLACK)
            screen.blit(text_surface, (20, 140))
            text_surface = font.render(f"Timestamp: {timestamp} ms", True, BLACK)
            screen.blit(text_surface, (20, 170))
         
            
            pygame.display.update()
    except KeyboardInterrupt:
        pygame.display.quit()
        pygame.quit()
        sys.exit()


def dbscanNoiseReduction(x_points, y_points, eps=500, min_samples=10, gaussian_sigma=1):
    points = list(zip(x_points, y_points))
    points = gaussian_filter(points, gaussian_sigma)
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = db.labels_

    clean_x = []
    clean_y = []
    for i, label in enumerate(labels):
        if label != -1:
            # Clean point
            clean_x.append(x_points[i])
            clean_y.append(y_points[i])

    return clean_x, clean_y

            

## ALL PREPROCESSING DONE IN REFERENCE TO:
##
##   https://wiki.youyeetoo.com/en/Lidar/LD20
def collectData():
    global radar_speed, starting_angle, distance, signal_strength, end_angle, timestamp
    global x_points, y_points
    
    try:
        while True:
            distances = []
            intensity = []
            angles = []
            start_byte = lidar_usb.read(1)
            if start_byte == b'\x54': # Predefined starting byte.
                length = int.from_bytes(lidar_usb.read(1), 'little')
                packet = lidar_usb.read(7 + (3*length))

                radar_speed = round(int.from_bytes(packet[0:2], 'little'), 2)
                starting_angle = round(int.from_bytes(packet[2:4], 'little') * 0.01, 2)
                start = 4
                for i in range(length-1):
                    distance = round(int.from_bytes(packet[start:start+2], 'little'), 2)
                    signal_strength = packet[start+2]
                    start += 3
                    distances.append(distance)
                    intensity.append(signal_strength)
                end_angle = round(int.from_bytes(packet[-5:-3], 'little')* 0.01, 2)
                timestamp = int.from_bytes(packet[-3:-1], 'little')
                crc = packet[-1]
                step = (end_angle - starting_angle)/(length-1)
                


                diff = (end_angle - starting_angle + 360) % 360
                if diff <= radar_speed * length / MEASURE_FREQ *1.5: 
                    for i in range(length-1):
                        angle = starting_angle + step*i
                        while(angle >= 360):
                            angle -= 360
                        angles.append(angle)
                        if(intensity[i] > 200 and distances[i] > 10 and distances[i] <= 8000):
                            x, y = polarToCart(distances[i]*0.01, angles[i])
                            x_points = x_points[1:] + [x]
                            y_points = y_points[1:] + [y]
                            if len(x_points) > NUM_POINTS*0.8:
                                x_points, y_points = dbscanNoiseReduction(x_points, y_points)
                            gdist = distances[i]
                            gangle = angles[i]
                            gint = intensity[i]
               
    except KeyboardInterrupt:
        lidar_usb.close()
        pygame.display.quit()
        pygame.quit()
        sys.exit()

data_thread = threading.Thread(target=collectData)
data_thread.start()

pygame_thread = threading.Thread(target=pygameDisplay)
pygame_thread.start()

ani = animation.FuncAnimation(fig, animate, interval=50, blit=True)
plt.show()

data_thread.join()
pygame_thread.join()
