from collections import OrderedDict
import pickle
from datetime import datetime
import os
from utils.datetime_utils import get_dt_string

class MetricRecorder:
    def __init__(self, experiment_name=None, save_dir=None):
        self.train_metrics = OrderedDict()
        self.test_metrics = OrderedDict()
        dt_string = get_dt_string()
        self.experiment_name = experiment_name if experiment_name is not None else dt_string
        self.save_dir = save_dir

    def update_metrics(self, train, test):
        self.train_metrics = train
        self.test_metrics = test

    def save_recorder(self):
        return
        save_dir = self.save_dir if self.save_dir is not None else ""
        file_path = save_dir + self.experiment_name + "_metric_recorder.obj"
        # FIX: no such file or directory problem
        file_handler = open(file_path, 'wb')
        pickle.dump(self, file_handler)
        file_handler.close()


def load_metric_recorder(experiment_name, save_dir):
    save_dir = save_dir if save_dir is not None else ""
    filename = save_dir + experiment_name + "_metric_recorder.obj"
    file_handler = open(filename, 'rb')
    recorder = pickle.Unpickler(file_handler).load()
    file_handler.close()
    return recorder
