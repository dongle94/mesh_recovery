import os
import multiprocessing as mp
from multiprocessing import Value, sharedctypes, Pool
import ctypes
import time

from core.mp.mp_queue import MessageQueue
from core.mp.task import Job, Task, MPTaskLauncher, TaskManager


class InputJOB(Job):
    def __init__(self):
        super().__init__()
        self.count = 0

    def process(self, item):
        data = [self.count, time.time()]
        self.count += 1
        print(f"input data: {item} / put data: {data}")
        return data


class JOB1(Job):
    def __init__(self, worker_num=0):
        super().__init__(i=worker_num)

    def process(self, data):
        print(f"{self.name}: {data}!")
        return data


class JOB2(Job):
    def __init__(self, name):
        super().__init__()
        self.name = name

    def process(self, data):
        print(f"{self.name}: {data}!")
        return data



# class Demo(Job):
#     def __init__(self, input_queue=None, output_queue=None, timeout=1, worker=1):
#         super().__init__(input_queue, output_queue, timeout, worker)
#
#         self.stop_flag = Value(ctypes.c_bool, False)
#         self.start_job(self.run, (self.input_queue, self.output_queue, ))
#         # self.results = self.pool.map_async(self.run, (self.input_queue, self.output_queue, ))
#         # print(self.results)
#
#         # self.ps = mp.Process(target=self.run, args=(self.input_queue, self.output_queue, ))
#         # self.ps.start()
#
#     def process(self, iq: MessageQueue = None, oq: MessageQueue = None):
#         print("start process: ", os.getpid(), time.time())
#         input_queue = iq
#         output_queue = oq
#         while True:
#             if self.stop_flag.value is True:
#                 print(f"pid-{os.getpid()}: get stop flag")
#                 break
#
#             data = None
#             # if input_queue is not None:
#             if input_queue.is_emtpy() is False:
#                 data = input_queue.get()
#
#             if data is not None:
#                 print(f"pid-{os.getpid()}: {data}, {time.time()}")
#                 time.sleep(self.timeout)
#
#             # if output_queue is not None and data is not None:
#             #     output_queue.put(data, wait=True)
#             time.sleep(0.001)
#
#     def stop(self):
#         print("스탑")
#         self.stop_flag.value = True
#
#
# class InputTask(Job):
#     def __init__(self, source, output_queue=None):
#         super().__init__(output_queue=output_queue)
#
#         self.start_job(self.run, source)
#
#     def run(self, source):
#         print("Input process: ", os.getpid())
#         from utils.config import set_config, get_config
#         from core.media_loader import MediaLoader
#         set_config('./configs/config.yaml')
#         cfg = get_config()
#         loader = MediaLoader(source, opt=cfg)
#
#         while True:
#             if self.stop_flag.value is True:
#                 print(f"pid-{os.getpid()}: get stop flag")
#                 break
#             im = loader.get_frame()
#
#             if self.output_queue is not None and im is not None:
#                 try:
#                     self.output_queue.put(im, block=False)
#                 except MessageQueue.Full():
#                     pass
#             time.sleep(0.1)
#
#
# class DetectionTask(Job):
#     def __init__(self, input_queue, output_queue=None, worker=1):
#         super().__init__(input_queue, output_queue, worker=worker)
#         self.detector = None
#
#         self.start_job(self.run, [])
#
#     def run(self):
#         print("Detection process: ", os.getpid())
#         from core.obj_detector import ObjectDetector
#         from utils.config import set_config, get_config
#         from utils.logger import init_logger
#         set_config('./configs/config.yaml')
#         cfg = get_config()
#         init_logger(cfg)
#
#         self.detector = ObjectDetector(cfg=cfg)
#         while True:
#             if self.stop_flag.value is True:
#                 print(f"pid-{os.getpid()}: get stop flag")
#                 break
#             data = None
#             det = None
#             if self.input_queue.is_emtpy() is False:
#                 data = self.input_queue.get()
#
#                 det = self.detector.run(img=data)
#
#             if self.output_queue is not None and det is not None:
#                 self.output_queue.put([data, det])
#             time.sleep(0.0001)
#
#
# class VisualizerTask(Job):
#     def __init__(self, input_queue, output_queue):
#         super().__init__(input_queue, output_queue)
#
#         self.start_job(self.run, [])
#
#     def run(self):
#         print("Visualization process: ", os.getpid())
#         data = None
#         while True:
#             if self.stop_flag.value is True:
#                 print(f"pid-{os.getpid()}: get stop flag")
#                 break
#             if self.input_queue.is_emtpy() is False:
#                 data = self.input_queue.get()
#
#             if data is not None:
#                 f, det = data[0], data[1]
#                 for d in det:
#                     x1, y1, x2, y2 = map(int, d[:4])
#                     cv2.rectangle(f, (x1, y1), (x2, y2), (96, 96, 216), thickness=2, lineType=cv2.LINE_AA)
#
#                 self.output_queue.put(f)
#             time.sleep(0.0001)

# def mp_func(data, t=1):
#     print(data, time.time(), t)
#     time.sleep(t)
#     return int(data + 15)


if __name__ == "__main__":
    print("Main process", os.getpid())
    mp.set_start_method('spawn')

    task_manager = TaskManager()

    first_task = Task(job=InputJOB(), empty_input_task=True)
    qi21 = MessageQueue(q_size=1)
    mptask1 = MPTaskLauncher(task=first_task,
                             proc_init_func=first_task.set_queues,
                             proc_init_args=[None, [qi21]])
    task_manager.add_task(mptask1)

    for i in range(2):
        task1 = Task(job=JOB1(worker_num=i))
        mptask2 = MPTaskLauncher(task=task1,
                                 proc_init_func=task1.set_queues,
                                 proc_init_args=[qi21, None])
        task_manager.add_task(mptask2)

    for i in range(2):
        task2 = Task(job=JOB2(name=f'JOB2-{i}'), use_default_queues=False)
        mptask3 = MPTaskLauncher(task=task2,
                                 proc_init_func=task2.set_queues,
                                 proc_init_args=[task1.output_queues, None])
        task_manager.add_task(mptask3)

    task_manager.start()

    t = 0
    while True:
        try:
            print(f"시작 한지: {t}초")
            t += 10
            time.sleep(10)
        except KeyboardInterrupt:
            print("키보드 인터럽트")
            break

    task_manager.stop()
