import json
import math

import cv2
import numpy as np

import MatterSim

WIDTH = 512
HEIGHT = 512
VFOV = math.radians(90)
HFOV = VFOV*WIDTH/HEIGHT

base_dir = 'data/v1/scans'

sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setDepthEnabled(False) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
sim.setObjectsEnabled(True)
sim.setSkyboxOffsetEnabled(True)
sim.setElevationLimits(-math.pi/2 + 1e-15, math.pi/2 - 1e-15)
sim.setBatchSize(6)
sim.initialize()

with open('connectivity/scans.txt') as scans_file:
    scans = scans_file.readlines()

for sidx, scan in enumerate(scans):
    print(f'Processing scan {sidx+1} of {len(scans)}')
    scan = scan.strip()
    with open(f'connectivity/{scan}_connectivity.json') as conn_file:
        connectivity = json.load(conn_file)
    for vidx, viewpoint in enumerate(connectivity):
        print(f'Generating skybox for viewpoint {vidx+1} of {len(connectivity)}')
        viewpoint_id = viewpoint['image_id']
        sim.newEpisode([scan] * 6, [viewpoint_id] * 6,
                       [math.pi, math.pi, 3 * math.pi / 2, 0, math.pi / 2, 0],
                       [0, -math.pi / 2, 0, math.pi/2, 0, 0])

        imgs = []
        for n, rot, state in zip(range(6), [0, 0, 90, 0, 270, 180], sim.getState()):
            img = np.array(state.object_segmentation, copy=False)
            if rot == 180:
                img = cv2.rotate(img, cv2.ROTATE_180)
            elif rot == 90:
                img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            elif rot == 270:
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            imgs.append(img)
        cv2.imwrite(f'{base_dir}/{scan}/matterport_skybox_images/{viewpoint_id}_skybox_segmentation_small.png', np.concatenate(imgs, axis=1))
