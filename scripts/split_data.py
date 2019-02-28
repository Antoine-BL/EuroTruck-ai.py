import os
import threading
from queue import Queue, Empty
import random

import numpy as np

PCT_TEST = 0.2


def main():
    split_data()


def split_data():
    data_path = os.path.realpath('..\\data')
    unsorted_path = os.path.join(data_path, 'shuffled', 'session-{}.npy')

    read_file_num = 1

    data_file = unsorted_path.format(read_file_num)
    files = Queue()
    while os.path.isfile(data_file):
        read_file_num += 1
        files.put(data_file)
        data_file = unsorted_path.format(read_file_num)

    saver = Saver()
    splitters = list(Splitter(PCT_TEST, saver=saver, files_to_split=files) for x in range(0, 5))

    for sp in splitters:
        sp.start()

    for sp in splitters:
        sp.join()

    np.save(os.path.join(data_path, 'test-set.npy'), np.array(saver.test_nums))
    np.save(os.path.join(data_path, 'train-set.npy'), np.array(saver.train_nums))


class Saver:
    def __init__(self):
        self.test_path = os.path.realpath('..\\data\\test\\session-{}.npy')
        self.train_path = os.path.realpath('..\\data\\train\\session-{}.npy')
        self.test_set = list()
        self.train_set = list()
        self.train_nums = list()
        self.test_nums = list()
        self.file_counter = 0
        self.lock = threading.Lock()

    def add_test(self, data):
        self.test_set.append(data)

        with self.lock:

            if len(self.test_set) >= 500:
                self.file_counter += 1
                files_to_save = np.array(self.test_set[:500])
                del self.test_set[:500]

                np.save(self.test_path.format(self.file_counter), files_to_save)
                self.test_nums.append(self.file_counter)

    def add_train(self, data):
        self.train_set.append(data)

        with self.lock:
            if len(self.train_set) >= 500:
                self.file_counter += 1
                files_to_save = np.array(self.train_set[:500])
                del self.train_set[:500]

                np.save(self.train_path.format(self.file_counter), files_to_save)
                self.train_nums.append(self.file_counter)


class Splitter(threading.Thread):
    def __init__(self, prob_test: float, files_to_split: Queue, saver: Saver):
        super().__init__()
        self.prob_test = prob_test
        self.files_to_split = files_to_split
        self.done = False
        self.saver = saver
        self.random: random.Random = random.Random()

    def run(self):
        try:
            while not self.done:
                file = self.files_to_split.get(True, 0.5)
                data = np.load(file)
                print('splitting: {}'.format(file))
                for data_point in data:
                    if self.random.random() <= PCT_TEST:
                        self.saver.add_test(data_point)
                    else:
                        self.saver.add_train(data_point)
        except Empty:
            return


if __name__ == '__main__':
    main()
