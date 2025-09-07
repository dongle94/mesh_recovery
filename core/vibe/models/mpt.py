
import os
import cv2
import time
import numpy as np
from tqdm import tqdm 

from core.vibe.models.sort import Sort


def run_tracker(image_folder, detector):
    '''
    Run tracker on an input video

    :param video (ndarray): input video tensor of shape NxHxWxC. Preferable use skvideo to read videos
    :return: trackers (ndarray): output tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
    '''

    # initialize tracker
    tracker = Sort()

    print('Running Multi-Person-Tracker')
    trackers = []

    files = [f for f in os.listdir(image_folder)
             if os.path.isfile(os.path.join(image_folder, f)) and
                f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files = sorted(files)
    for batch in tqdm(files):
        img = cv2.imread(os.path.join(image_folder, batch))
        predictions = detector.run(img)

        # collect all detections for this frame, then update tracker once
        dets_list = []
        for pred in predictions:
            dets_list.append(pred[:5])

        if len(dets_list) > 0:
            dets = np.vstack(dets_list)
            track_bbs_ids = tracker.update(dets)
        else:
            track_bbs_ids = np.empty((0, 5))
        trackers.append(track_bbs_ids)

    return trackers


def prepare_output_tracks(trackers):
    '''
    Put results into a dictionary consists of detected people
    :param trackers (ndarray): input tracklets of shape Nx5 [x1,y1,x2,y2,track_id]
    :return: dict: of people. each key represent single person with detected bboxes and frame_ids
    '''
    people = dict()

    for frame_idx, tracks in enumerate(trackers):
        for d in tracks:
            person_id = int(d[4])
            # bbox = np.array([d[0], d[1], d[2] - d[0], d[3] - d[1]]) # x1, y1, w, h

            w, h = d[2] - d[0], d[3] - d[1]
            c_x, c_y = d[0] + w/2, d[1] + h/2
            w = h = np.where(w / h > 1, w, h)
            bbox = np.array([c_x, c_y, w, h])

            if person_id in people.keys():
                people[person_id]['bbox'].append(bbox)
                people[person_id]['frames'].append(frame_idx)
            else:
                people[person_id] = {
                    'bbox' : [],
                    'frames' : [],
                }
                people[person_id]['bbox'].append(bbox)
                people[person_id]['frames'].append(frame_idx)
    for k in people.keys():
        people[k]['bbox'] = np.array(people[k]['bbox']).reshape((len(people[k]['bbox']), 4))
        people[k]['frames'] = np.array(people[k]['frames'])

    return people