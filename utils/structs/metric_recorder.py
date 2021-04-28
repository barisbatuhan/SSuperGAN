from collections import OrderedDict
import pickle
from datetime import datetime


class MetricRecorder:
    def __init__(self, experiment_name=None, save_dir="./"):
        self.train_metrics = OrderedDict()
        self.test_metrics = OrderedDict()
        now = datetime.now()
        # dd/mm/YY-H:M:S
        dt_string = now.strftime("%d/%m/%Y-%H:%M:%S")
        self.experiment_name = experiment_name if experiment_name is not None else dt_string
        self.save_dir = save_dir

    def update_metrics(self, train, test):
        self.train_metrics = train
        self.test_metrics = test

    def save_recorder(self):
        save_dir = self.save_dir if self.save_dir is not None else ""
        file_handler = open(save_dir + self.experiment_name + "_metric_recorder.obj", 'wb+')
        pickle.dump(self, file_handler)
        file_handler.close()


def load_metric_recorder(experiment_name, save_dir):
    save_dir = save_dir if save_dir is not None else ""
    filename = save_dir + experiment_name + "_metric_recorder.obj"
    file_handler = open(filename, 'rb')
    recorder = pickle.Unpickler(file_handler).load()
    file_handler.close()
    return recorder
