from synthesize import synthesize
from hyperparams import Hyperparams as hp
import numpy as np

class Evaluator:
    def __init__(self, keys=["index", "init_time","Latency_beginning","Latency_synthesis", "duration_mels",
                             "duration_mags", "duration_total", "file_name", "n_samples", "duration",
                             "roboticness", "gpu_util", "max_memory_required", "repetitions", "skipping"]):
        self.keys = keys
        self.log = {key: [] for key in self.keys}

    # hardware Measures
    def gpu_util(self):
        return 0

    def max_memory_required(self):
        return 0

    # Phenomenon Detection
    def roboticness(self):
        return 0

    def repetitions(self):
        return 0

    def skipping(self):
        return 0

    # Loss Measures
    def autokorrelation(self, pred, target):
        pass

    def mel_cepstral_distortion(self, pred, target):
        K = 10 / np.log(10) * np.sqrt(2)
        return K * np.mean(np.sqrt(np.sum((pred - target) ** 2, axis=1)))

    def calculate_f0(self, x):
        pass

    def rmse_for_f0(self, pred, target):
        pass

    def calculate_metrics(self, loss, target, pred):
        pass

    # Evaluation
    def evaluate_inside_domain(self, epoch):
        info = synthesize("inside")

    def evaluate_outside_domain(self, epoch):
        info = synthesize("outside")
        for file_name in info["samples"].keys():
            utterance = info["samples"][file_name]
            self.log["file_name"] = file_name
            self.log["n_samples"] = len(utterance)
            self.log["duration"] = self.log["n_samples"] / hp.sr
            # self.log["roboticness"] = self.roboticness()  #TODO implement these functions!!!
            # self.log["gpu_util"] = self.gpu_util()
            # self.log["max_memory_required"] = self.max_memory_required()
            # self.log["repetitions"] = self.repetitions()
            # self.log["skipping"] = self.skipping()
            self.log["epoch"] = epoch
            for key in info["time_measurements"].keys():
                self.log[key].append(info["time_measurements"][key])
            self.log["relative_synthesis_time"] = self.log["duration"][-1] / self.log["duration_total"][-1]

    def evaluate(self, epoch):
        # Speed Measures calculated during synthesis
        info = synthesize()

        print("EVALUATION")
        print(info["time_measurements"])
        for file_name in info["samples"].keys():
            utterance = info["samples"][file_name]
            self.log["file_name"] = file_name
            self.log["n_samples"] = len(utterance)
            self.log["duration"] = self.log["n_samples"] / hp.sr
            # self.log["roboticness"] = self.roboticness()  #TODO implement these functions!!!
            # self.log["gpu_util"] = self.gpu_util()
            # self.log["max_memory_required"] = self.max_memory_required()
            # self.log["repetitions"] = self.repetitions()
            # self.log["skipping"] = self.skipping()

            self.log["epoch"] = epoch
            for key in info["time_measurements"].keys():
                self.log[key].append(info["time_measurements"][key])
            self.log["relative_synthesis_time"] = self.log["duration"][-1] / self.log["duration_total"][-1]
