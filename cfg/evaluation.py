from synthesize import synthesize
from hyperparams import Hyperparams as hp

class Evaluator:
    def __init__(self, keys=["index", "init_time",
                             "Latency_beginning",
                             "Latency_synthesis",
                             "duration_mels",
                             "duration_mags",
                             "duration_total", ]):
        self.keys = keys
        self.log = {key: [] for key in self.keys}

    def roboticness(self):
        return 0

    def gpu_util(self):
        return 0

    def max_memory_required(self):
        return 0

    def repetitions(self):
        return 0

    def skipping(self):
        return 0

    def evaluate(self, epoch):
        info = synthesize()
        print("EVALUATION")
        print(info["time_measurements"])
        for file_name in info["samples"].keys():
            utterance = info["samples"][file_name]
            self.log["file_name"] = file_name
            self.log["n_samples"] = len(utterance)
            self.log["duration"] = self.log["n_samples"]/hp.sr
            self.log["roboticness"] = self.roboticness()
            self.log["gpu_util"] = self.gpu_util()
            self.log["max_memory_required"] = self.max_memory_required()

            self.log["epoch"]=epoch
            for key in info["time_measurements"].keys():
                self.log[key].append(info["time_measurements"][key])
            self.log["relative_synthesis_time"] = self.log["duration"][-1] / self.log["duration_total"][-1]

