import os
import sys
import queue
import cv2
import time
from multiprocessing import Lock, set_start_method

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_PATH)
sys.path.append(os.path.join(ROOT_PATH, 'object_detector'))

from core.tasks import VideoInputTask, ObjectDetectionTask, HybrIKTask, PostHybrIKTask
from core.mp.task import Task, MPTaskLauncher, TaskManager
from utils.config import set_config, get_config
from utils.logger import init_logger


def run(source=None):
    set_start_method('spawn')

    lock = Lock()
    set_config('./configs/config.yaml')
    cfg = get_config()
    init_logger(cfg)

    cfg.media_source = os.path.expanduser(cfg.media_source)
    if os.path.isfile(cfg.media_source) is False:
        print(cfg.media_source)
        raise FileNotFoundError("Not exist Video file")

    task_manager = TaskManager()

    input_task = Task(job=VideoInputTask(cfg), empty_input_task=True)
    mp_input_task = MPTaskLauncher(task=input_task)
    task_manager.add_task(mp_input_task)
    last_queues = input_task.output_queues

    detection_task = Task(job=ObjectDetectionTask(cfg))
    mp_detection_task = MPTaskLauncher(task=detection_task,
                                       proc_init_func=detection_task.set_queues,
                                       proc_init_args=input_task.output_queues)
    task_manager.add_task(mp_detection_task)
    last_queues = detection_task.output_queues

    hybrik_task = Task(job=HybrIKTask(cfg=cfg), is_torch=True)
    mp_hybrik_task = MPTaskLauncher(task=hybrik_task,
                                    proc_init_func=hybrik_task.set_queues,
                                    proc_init_args=detection_task.output_queues)
    task_manager.add_task(mp_hybrik_task)
    last_queues = hybrik_task.output_queues

    post_hybrik_task = Task(job=PostHybrIKTask(cfg=cfg))
    mp_post_hybrik_task = MPTaskLauncher(task=post_hybrik_task,
                                    proc_init_func=post_hybrik_task.set_queues,
                                    proc_init_args=hybrik_task.output_queues)
    task_manager.add_task(mp_post_hybrik_task)
    last_queues = post_hybrik_task.output_queues

    task_manager.start()
    while True:
        try:
            try:
                with lock:
                    item = last_queues[0].get()
            except queue.Empty:
                time.sleep(0.01)
                continue
            if item is None:
                print("Item None. Program stop.")
                break

            # img = item['image']
            is_end = item['end']
            if is_end is True:
                print("Get End sign. Program stop")
                break
            # cv2.imshow('video', img)

            frame = item['frame']
            bframe = item['b_frame']
            cv2.imshow('frame', frame)
            cv2.imshow('bframe', bframe)

            if cv2.waitKey(1) == ord('q'):
                break
        except KeyboardInterrupt:
            print("KeyboardInterrupt. Exit loop.")
            break

    cv2.destroyAllWindows()
    task_manager.stop()


if __name__ == "__main__":

    run()
