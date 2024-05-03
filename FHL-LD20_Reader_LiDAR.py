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
from scipy.ndimage import gaussian_filter
import numpy as np
from sklearn.decomposition import PCA
import time
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

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

lidar_usb = Serial('COM7', 230400, timeout=1)

# Global variables for data
radar_speed = 0
starting_angle = 0 
distance = 0
signal_strength = 0
end_angle = 0
timestamp = 0

MEASURE_FREQ = 4000  # samples / second

gdist = 0
gangle = 0
gint = 0

NUM_POINTS = 500
OUTPUT_BATCH_SIZE = 5
LOCATION_NAME = "RADY131"

NOISE_REDUC_ITERS = 500
noise_iter = 0
output_iter = 0
output_points = []
predicted_class = ""

# Create figure for animation
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

# Create zero vector for x and y points
x_points = [0] * NUM_POINTS
y_points = [0] * NUM_POINTS

scatter = ax.scatter([], [], s=10)
AX_DIM = 800
ax.set_ylim([-AX_DIM, AX_DIM])
ax.set_xlim([-AX_DIM, AX_DIM])

model = load_model("lidar_detection.h5") 


def normalize_data(data):
    max_val = np.max(data)
    min_val = np.min(data)
    normalized_data = 2 * ((data - min_val) / (max_val - min_val)) - 1
    return normalized_data

def outputPoints(points):
    global output_iter, output_points
    output_points.append(points)
    output_iter += 1
    if(output_iter >= OUTPUT_BATCH_SIZE):
        # OUTPUT POINTS TO FILE
        with open("labeled_points.csv", 'a') as f:
            for i in range(OUTPUT_BATCH_SIZE):
                f.write(str(output_points[i]) + "\t")
            f.write(f'{LOCATION_NAME}\n')
            f.close()
        output_points = []
        output_iter = 0
        time.sleep(0.1)
    return


def polarToCart(R, deg):
    return ((R*math.cos((deg*math.pi)/180)), R*math.sin((deg*math.pi)/180))

def animate(i):
    global x_points, y_points, noise_iter, predicted_class
    encoder_mapping = {0: 'HALLWAY1-EAST', 1: 'HALLWAY1-NORTH', 2: 'HALLWAY1-SOUTH', 3: 'HALLWAY1-WEST',
                       4: 'RADY129-EAST', 5: 'RADY129-NORTH', 6: 'RADY129-SOUTH', 7: 'RADY129-WEST',
                       8: 'RADY131-EAST', 9: 'RADY131-NORTH', 10: 'RADY131-SOUTH', 11: 'RADY131-WEST'}

    if noise_iter >= NOISE_REDUC_ITERS:
        batch_data = []
        for _ in range(OUTPUT_BATCH_SIZE):
            points = list(zip(x_points, y_points))
            points = list(set(points))
            batch_data.append(points)

        batch = np.array(batch_data)
        batch = normalize_data(batch)
        batch = batch.reshape(1, OUTPUT_BATCH_SIZE, NUM_POINTS, 2)

        predicted_probs = model.predict(batch[0])
        predicted_label_index = np.argmax(predicted_probs)
        predicted_class = encoder_mapping[predicted_label_index]        
        scatter.set_offsets(batch_data[-1])

    return scatter,





def pygameDisplay():
    global gint, gdist, predicted_class
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
            text_surface = font.render(f"Prediction: {predicted_class}", True, BLACK)
            screen.blit(text_surface, (20, 200))

            pygame.display.update()
    except KeyboardInterrupt:
        pygame.display.quit()
        pygame.quit()
        sys.exit()


def noiseReduction(points, gaussian_sigma=1):
    # points = gaussian_filter(points, gaussian_sigma)
    pca = PCA(n_components=2)

    for i, (x, y) in enumerate(points):  # quantize points to bins of 10 after processing
        x = int(x)
        x = x - (x % 5)
        x_points[i] = x
        y = int(y)
        y = y - (y % 5)
        y_points[i] = y
    points = list(zip(x_points, y_points))
    return x_points, y_points


def collectData():
    global radar_speed, starting_angle, distance, signal_strength, end_angle, timestamp
    global x_points, y_points, gint, gdist, gangle
    global noise_iter
    output_points = []

    try:
        while True:
            distances = []
            intensity = []
            angles = []
            start_byte = lidar_usb.read(1)
            if start_byte == b'\x54':  # Predefined starting byte.
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
                            x, y = polarToCart(distances[i]*0.1, angles[i])
                            x_points = x_points[1:] + [x]
                            y_points = y_points[1:] + [y]
                            points = list(zip(x_points, y_points))
                            noise_iter += 1
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

ani = animation.FuncAnimation(fig, animate, interval=100, blit=True)
plt.show()

data_thread.join()
pygame_thread.join()
