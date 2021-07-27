#!/Library/Frameworks/Python.framework/Versions/3.5/bin/python3.5
import math
import torch.nn as nn
import torch

class Hyper_synthesis(nn.Module):
    def __init__(self, num_filters=128):
        super(Hyper_synthesis, self).__init__()
        self.conv1 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1, padding=1)
        self.leaky_relu1 = nn.LeakyReLU()
        self.conv2 = nn.ConvTranspose2d(num_filters, num_filters, 3, stride=2, padding=1, output_padding=1)
        self.leaky_relu2 = nn.LeakyReLU()
        self.conv3 = nn.ConvTranspose2d(num_filters, int(num_filters*1.5), 3, stride=1, padding=1)
        self.leaky_relu3 = nn.LeakyReLU()
        self.conv4 = nn.ConvTranspose2d(int(num_filters*1.5), int(num_filters*1.5), 3, stride=2, padding=1, output_padding=1)
        self.leaky_relu4 = nn.LeakyReLU()
        # self.conv5 = nn.ConvTranspose2d(int(num_filters*1.5), num_filters*2, 3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(int(num_filters*1.5), num_filters*2, 3, stride=1, padding=1)
    
    def forward(self, x):
        x = self.leaky_relu1(self.conv1(x))
        x = self.leaky_relu2(self.conv2(x))
        x = self.leaky_relu3(self.conv3(x))
        x = self.leaky_relu4(self.conv4(x))
        x = self.conv5(x)
        return x

if __name__ == "__main__":
    hyper_synthesis = Hyper_synthesis()
    input_image = torch.zeros([1,128, 8, 12])
    result = hyper_synthesis(input_image)
    print("result: ", result.shape)
