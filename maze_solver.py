import cv2
import numpy as np
from collections import deque
from matplotlib import pyplot as plt


img = cv2.imread("Maze.png", cv2.IMREAD_COLOR)
binary = cv2.imread("Maze.png", cv2.IMREAD_GRAYSCALE)


_, raw = cv2.threshold(binary, 128, 255, cv2.THRESH_BINARY)
occupancy_grid = np.where(raw == 255, 0, 1)


def find_nodes(modified_img):
    hsv = cv2.cvtColor(modified_img, cv2.COLOR_BGR2HSV)

    
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])
    maskA = cv2.inRange(hsv, lower_green, upper_green)

    
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    maskB = cv2.inRange(hsv, lower_blue, upper_blue)

    
    coordsA = np.where(maskA > 0)
    if coordsA[0].size > 0:
        Ay, Ax = np.mean(coordsA, axis=1)
        start_point = (int(Ay), int(Ax))
    else:
        print("Warning: Green marker (A) not found. Returning (-1, -1).")
        start_point = (-1, -1)

    
    coordsB = np.where(maskB > 0)
    if coordsB[0].size > 0:
        By, Bx = np.mean(coordsB, axis=1)
        end_point = (int(By), int(Bx))
    else:
        print("Warning: Blue marker (B) not found. Returning (-1, -1).")
        end_point = (-1, -1)

    
    mask_combined = cv2.bitwise_or(maskA, maskB)
    modified_img[mask_combined > 0] = [255, 255, 255]

    return modified_img, start_point, end_point



proper, start, end = find_nodes(img)
print("Start:", start, "End:", end)

plt.imshow(raw, cmap='gray')
plt.show()


gray = cv2.cvtColor(proper, cv2.COLOR_BGR2GRAY)
_, raw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
occupancy_grid = np.where(raw == 255, 0, 1)



def bfs(occ, start, end):
    Sy, Sx = start
    Ey, Ex = end
    visited = set()
    directions = [(1,0), (0,1), (0,-1), (-1,0)]
    queue = deque()
    queue.append([start, [start]])

    while queue:
        current_node, current_path = queue.popleft()
        visited.add(current_node)
        y, x = current_node

        if current_node == end:
            return current_path

        for dx, dy in directions:
            new_x = x + dx
            new_y = y + dy

            if 0 <= new_y < occ.shape[0] and 0 <= new_x < occ.shape[1]:
                new_val = occ[new_y][new_x]
                if (new_y, new_x) not in visited and new_val == 0:
                    new_node = (new_y, new_x)
                    new_path = current_path + [new_node]
                    queue.append([new_node, new_path])



path = bfs(occupancy_grid, start, end)


for i in range(len(path)-1):
    y1, x1 = path[i]
    y2, x2 = path[i+1]
    cv2.line(img, (x1, y1), (x2,y2), (0,0,255), 2)

cv2.imshow("Solved Maze", img)
