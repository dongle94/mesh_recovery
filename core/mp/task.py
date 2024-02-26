import os
import signal
import queue
import time
import multiprocessing as mp

from core.mp.mp_queue import MessageQueue


class Job(object):
    def __init__(self, i=0):
        self.name = self.__class__.__name__ + f'-{i}'

    def init(self):
        pass

    def process(self, data):
        print(f"{self.name}: {data}!")
        return data

    def close(self):
        pass


class Task(object):
    def __init__(self, job=None, use_default_queues=True, empty_input_task=False):
        if use_default_queues:
            self.input_queues = [MessageQueue(q_size=1)]
            self.output_queues = [MessageQueue(q_size=1)]
        else:
            self.input_queues = None
            self.output_queues = None

        self.job = job
        self.output_timeout = 1

        self._is_started = False
        self._is_initialized = False
        self._need_finish = False

        self.lock = mp.Lock()
        self.empty_input_task = empty_input_task

    def set_queues(self, input_queues=None, output_queues=None):
        if input_queues is not None:
            self.input_queues = input_queues if isinstance(input_queues, (list, tuple)) else [input_queues]
        if output_queues is not None:
            self.output_queues = output_queues if isinstance(output_queues, (list, tuple)) else [output_queues]

    @property
    def job(self):
        return self._job

    @job.setter
    def job(self, job):
        self._job = job

    def initialize(self):
        if self._is_initialized:
            return
        self._is_initialized = True

        # self.pool = thread_util.ThreadPool(self.pool_size)
        if self.job:
            try:
                self.job.init()
            except Exception as e:
                msg = e.message if hasattr(e, 'message') else e
                print(f"Failed initializing job {self}, {msg}")
                raise e

        signal.signal(signal.SIGTERM, self._stop_signal)

    def start(self):
        if self._is_started:
            return
        self._is_started = True
        self.initialize()
        self.work_loop()

    def stop(self):
        self._need_finish = True
        self.job.close()
        if self.input_queues:
            for q in self.input_queues:
                q.close()
        if self.output_queues:
            for q in self.output_queues:
                q.close()

    def _stop_signal(self, signum, frame):
        self._need_finish = True

    def work_loop(self):
        print(f"Started {self.__str__()}")
        self._is_started = True
        self.initialize()

        try:
            while self._need_finish is False:
                if self.empty_input_task is True:
                    item = None
                    time.sleep(0.001)
                else:
                    try:
                        with self.lock:
                            item = self.input_queues[0].get(block=True, timeout=0.01)
                    except queue.Empty:
                        time.sleep(0.001)
                        continue

                self._process(item)
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
        except Exception as e:
            print(f"Exception in _work_loop: {e}")

        self.stop()
        print(f"Finished {self}")

    def _process(self, item):
        result = self.job.process(item)
        if result is None or self.output_queues is None:
            return
        new_input = result
        self._process_result(new_input, self.output_queues[0])

    def _process_result(self, res, output_queue):
        if output_queue is None or res is None:
            return
        while output_queue and self._need_finish is False:
            try:
                with self.lock:
                    output_queue.put(res, block=True, timeout=self.output_timeout)
                break
            except queue.Full:
                time.sleep(0.001)
                continue

    def __str__(self):
        return f'Worker(job={self.job.name}, {os.getpid()})'


class TaskLauncher(object):
    def __init__(self, task):
        self.task = task

    def start(self):
        self.task.start()

    def stop(self):
        self.task.stop()


class MPTaskLauncher(TaskLauncher):
    def __init__(self, task, proc_init_func=None, proc_init_args=None):
        super().__init__(task)
        ctx = mp.get_context('spawn')

        self.bg_process = ctx.Process(
            name=task.job.name,
            target=self._process_start
        )
        self.bg_process.daemon = True

        self.proc_init_func = proc_init_func
        self.proc_init_args = proc_init_args

    def start(self):
        self.bg_process.start()

    def _process_start(self):
        if self.proc_init_func:
            self.proc_init_func(*self.proc_init_args)
        self.task.start()

    def stop(self):
        self.task.stop()
        if self.bg_process:
            if self.bg_process.is_alive():
                self.bg_process.join(1.0)
                self.bg_process.terminate()


class TaskManager(object):
    def __init__(self):
        self.task_list = []

    def add_task(self, task):
        self.task_list.append(task)

    def remove_task(self, task):
        if task in self.task_list:
            task.stop()

            self.task_list.remove(task)

    def start(self):
        for t in self.task_list:
            t.start()

    def stop(self):
        for t in self.task_list:
            t.stop()

        self.task_list = []

    def __del__(self):
        if self.task_list:
            self.stop()
