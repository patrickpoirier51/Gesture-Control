# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import print_function

import itertools
import svgwrite
import time

import numpy as np
import pose_camera

from dronekit import connect

#connection_string = ('tcp:192.168.2.250:5763')
connection_string = ('tcp:192.168.2.237:5763')

# Connect to the Vehicle
print('Connecting to vehicle on: %s' % connection_string)
vehicle = connect(connection_string, wait_ready=True)



EXTENT= 1

# General ID
PILOT_1 = 1
PILOT_2 = 2
PILOT_3 = 3

CHANNELS = (PILOT_1, PILOT_2, PILOT_3)


class Identity:
    def __init__(self, color, base_command, channel, extent=EXTENT):
        self.color = color
        self.base_command = base_command
        self.channel = CHANNELS.index(channel)
        self.extent = extent


IDENTITIES = (
    Identity('cyan', 24, PILOT_1),
    Identity('magenta', 12, PILOT_2),
    Identity('yellow', 36, PILOT_3),
)


class Pose:
    def __init__(self, pose, threshold):
        self.pose = pose
        self.id = None
        self.keypoints = {label: k for label, k in pose.keypoints.items()
                          if k.score > threshold}
        self.center = (np.mean([k.yx for k in self.keypoints.values()], axis=0)
                       if self.keypoints else None)

    def quadrance(self, other):
        d = self.center - other.center
        return d.dot(d)


class PoseTracker:
    def __init__(self):
        self.prev_poses = []
        self.next_pose_id = 0

    def assign_pose_ids(self, poses):
        """copy nearest pose ids from previous frame to current frame"""
        all_pairs = sorted(itertools.product(poses, self.prev_poses),
                           key=lambda pair: pair[0].quadrance(pair[1]))
        used_ids = set()
        for pose, prev_pose in all_pairs:
            if pose.id is None and prev_pose.id not in used_ids:
                pose.id = prev_pose.id
                used_ids.add(pose.id)

        for pose in poses:
            if pose.id is None:
                pose.id = self.next_pose_id
                self.next_pose_id += 1

        self.prev_poses = poses


def main():
    pose_tracker = PoseTracker()

    def run_inference(engine, input_tensor):
        return engine.run_inference(input_tensor)

    def render_overlay(engine, output, src_size, inference_box):

        svg_canvas = svgwrite.Drawing('', size=src_size)
        outputs, inference_time = engine.ParseOutput(output)

        poses = [pose for pose in (Pose(pose, 0.2) for pose in outputs)
                 if pose.keypoints]
        pose_tracker.assign_pose_ids(poses)

        velocities = {}
        for pose in poses:
            left = pose.keypoints.get('left wrist')
            right = pose.keypoints.get('right wrist')
            if not (left and right): continue

            identity = IDENTITIES[pose.id % len(IDENTITIES)]
            lefty = 1 - left.yx[0] / engine.image_height
            leftx = 1 - left.yx[1] / engine.image_width 
            righty = 1 - right.yx[0] / engine.image_height
            rightx = 1 - right.yx[1] / engine.image_width
            #print (lefty, leftx, righty, rightx)
            #print (int(lefty * 1200)+900 , int(leftx * 2400)+900, int(righty * 1200)+900, int((rightx-0.5) * 2400)+900 )
            #ch3 = int(lefty * 1200)+900
            #ch4 = int(leftx * 2400)+900
            #ch2 = int(righty * 1200)+900
            # ch1 = int((rightx-0.5) * 2400)+900
            
            ch3 = int(lefty * 600)+1200
            ch4 = int(leftx * 1200)+1200
            ch2 = int(righty * 600)+1200
            ch1 = int((rightx-0.5) * 1200)+1200
            
            print (ch1,ch2,ch3,ch4)
            #print("Set Ch1-Ch8 overrides to 110-810 respectively")
            vehicle.channels.overrides = {'1': ch1, '2': ch2,'3': ch3,'4':ch4 ,'5':1500,'6':1500,'7':1500,'8':1500}
            time.sleep(0.1)


        for i, pose in enumerate(poses):
            identity = IDENTITIES[pose.id % len(IDENTITIES)]
            pose_camera.draw_pose(svg_canvas, pose, src_size, inference_box, color=identity.color)

        return (svg_canvas.tostring(), False)

    pose_camera.run(run_inference, render_overlay)


if __name__ == '__main__':
    main()

