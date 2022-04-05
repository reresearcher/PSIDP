# encoding: utf-8
import os
import torch
import torch.nn as nn
import torchvision.models as models



def load_model(arch, code_length):

    if arch == 'vgg16':
        model0 = models.vgg16(pretrained=True)
        model = models.vgg16(pretrained=True)
        model0 = ModelWrapper0(model0)

        model.classifier = model.classifier[:-3]
        model = ModelWrapper(model, 4096, code_length)
    else:
        raise ValueError("Invalid model name!")

    return model, model0

# 1000 VS 4096
class ModelWrapper0(nn.Module):

    def __init__(self, model):
        super(ModelWrapper0, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class ModelWrapper(nn.Module):

    def __init__(self, model, last_node, code_length):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.code_length = code_length
        self.hash_layer = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(last_node, code_length),
            nn.Tanh(),
        )
        self.fake_attribute = nn.Sequential(
            nn.Linear(code_length, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1000),
            nn.ReLU(inplace=True),
        )
        self.attribute_code = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, code_length),
            nn.Tanh(),
        )
        # Extract features
        self.extract_features = False

    def forward(self, x):
        if self.extract_features:
            return self.model(x)
        else:
            return self.hash_layer(self.model(x)), self.fake_attribute(self.hash_layer(self.model(x))), self.attribute_code(self.fake_attribute(self.hash_layer(self.model(x))))
            # hash code, fake attribute, reconstruct code
    def set_extract_features(self, flag):

        self.extract_features = flag

    def snapshot(self, pth_name):
        torch.save({
            'model_state_dict': self.state_dict(),
        }, os.path.join('./checkpoints/', '{}.pth'.format(pth_name)))

    def load_snapshot(self, pth_name):

        checkpoint = torch.load(pth_name)
        self.load_state_dict(checkpoint['model_state_dict'])