import torch.nn as nn
import torch
import torchvision.models


class ModelWrapper(nn.Module):
    def __init__(self, config):
        super(ModelWrapper, self).__init__()
        self.config = config
        self.model = self._normal_model()

    def forward(self, x):
        x = self.model(x)
        return x

    def _normal_model(self):
        print(self.config.model)
        # resnet50
        if self.config.model == "resnet50":
            model = torchvision.models.resnet50(pretrained=self.config.pre_trained)
            model.fc = nn.Linear(in_features=2048, out_features=self.config.num_out, bias=True)
        elif self.config.model == "se-resnext101":
            model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_se_resnext101_32x4d')
            model.fc = nn.Linear(in_features=2048, out_features=self.config.num_out, bias=True)
        else:
            raise AssertionError("Currently, we only support resnet50, se-resnext101."
                                 "Please choose one of them as an input.")
        print(f"Model {self.config.model} is chosen!")
        return model

