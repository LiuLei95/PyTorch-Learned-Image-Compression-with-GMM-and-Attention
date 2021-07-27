import numpy as np
import os
import torch
import torchvision.models as models
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import math
import torch.nn.init as init
import logging
from torch.nn.parameter import Parameter
from models import *


def save_model(model, iter, name):
    torch.save(model.state_dict(), os.path.join(name, "iter_{}.pth.tar".format(iter)))


def load_model(model, f):
    with open(f, 'rb') as f:
        pretrained_dict = torch.load(f)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    f = str(f)
    if f.find('iter_') != -1 and f.find('.pth') != -1:
        st = f.find('iter_') + 5
        ed = f.find('.pth', st)
        return int(f[st:ed])
    else:
        return 0


class ImageCompressor(nn.Module):
    def __init__(self, out_channel_N=128):
    # def __init__(self, out_channel_N=192, out_channel_M=320):
        super(ImageCompressor, self).__init__()
        # self.Encoder = Analysis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        # self.Decoder = Synthesis_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        # self.priorEncoder = Analysis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        # self.priorDecoder = Synthesis_prior_net(out_channel_N=out_channel_N, out_channel_M=out_channel_M)
        self.Encoder = Analysis_transform(out_channel_N)
        self.Decoder = Synthesis_transform(out_channel_N)
        self.priorEncoder = Hyper_analysis(out_channel_N)
        self.priorDecoder = Hyper_synthesis(out_channel_N)
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.entropy = Entropy(out_channel_N)
        self.out_channel_N = out_channel_N
        # self.out_channel_M = out_channel_M

    def forward(self, input_image):
        quant_noise_feature = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 16, input_image.size(3) // 16).cuda()
        quant_noise_z = torch.zeros(input_image.size(0), self.out_channel_N, input_image.size(2) // 64, input_image.size(3) // 64).cuda()
        quant_noise_feature = torch.nn.init.uniform_(torch.zeros_like(quant_noise_feature), -0.5, 0.5)
        quant_noise_z = torch.nn.init.uniform_(torch.zeros_like(quant_noise_z), -0.5, 0.5)
        feature = self.Encoder(input_image)
        batch_size = feature.size()[0]

        z = self.priorEncoder(feature)
        if self.training:
            compressed_z = z + quant_noise_z
        else:
            compressed_z = torch.round(z)
        phi = self.priorDecoder(compressed_z)
        feature_renorm = feature
        if self.training:
            compressed_feature_renorm = feature_renorm + quant_noise_feature
        else:
            compressed_feature_renorm = torch.round(feature_renorm)

        #------------Add mask conv---------------
        # compressed_feature use mask
        means, sigmas, weights = self.entropy(phi, compressed_feature_renorm)
        recon_image = self.Decoder(compressed_feature_renorm)
        # recon_image = prediction + recon_res
        clipped_recon_image = recon_image.clamp(0., 1.)
        # distortion
        mse_loss = torch.mean((recon_image - input_image).pow(2))
        im_shape = input_image.size()

        def feature_probs_based_sigma(feature, mu, sigma):
            # mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs
       
        def feature_probs_based_GMM(feature, means, sigmas, weights):
            mean1 = torch.squeeze(means[:,:,:,:,0])
            mean2 = torch.squeeze(means[:,:,:,:,1])
            mean3 = torch.squeeze(means[:,:,:,:,2])
            sigma1 = torch.squeeze(sigmas[:,:,:,:,0])
            sigma2 = torch.squeeze(sigmas[:,:,:,:,1])
            sigma3 = torch.squeeze(sigmas[:,:,:,:,2])

            weight1, weight2, weight3 = torch.squeeze(weights[:,:,:,:,0]), torch.squeeze(weights[:,:,:,:,1]), torch.squeeze(weights[:,:,:,:,2])
            # sigma1, sigma2, sigma3 = sigma1.clamp(1e-10, 1e10), sigma2.clamp(1e-10, 1e10), sigma3.clamp(1e-10, 1e10)
            gaussian1 = torch.distributions.laplace.Laplace(mean1, sigma1)
            gaussian2 = torch.distributions.laplace.Laplace(mean2, sigma2)
            gaussian3 = torch.distributions.laplace.Laplace(mean3, sigma3)
            prob1 = gaussian1.cdf(feature + 0.5) - gaussian1.cdf(feature - 0.5)
            prob2 = gaussian2.cdf(feature + 0.5) - gaussian2.cdf(feature - 0.5)
            prob3 = gaussian3.cdf(feature + 0.5) - gaussian3.cdf(feature - 0.5)

            probs = weight1 * prob1 + weight2 * prob2 + weight3 * prob3
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs
 
            

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        # 正常sigma是256维度的，而feature是128维度的，为了匹配，先将sigma改成了128维度
        # total_bits_feature, _ = feature_probs_based_sigma(compressed_feature_renorm, recon_sigma)
        total_bits_feature, _ = feature_probs_based_GMM(compressed_feature_renorm, means, sigmas, weights)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        # im_shape = input_image.size()
        bpp_feature = total_bits_feature / (batch_size * im_shape[2] * im_shape[3])
        bpp_z = total_bits_z / (batch_size * im_shape[2] * im_shape[3])
        bpp = bpp_feature + bpp_z
        return clipped_recon_image, mse_loss, bpp_feature, bpp_z, bpp


def feature_probs_based_More_Gauss(feature, means, sigmas, weights):
    # Calculate the pmf/cdf
    mean1 = torch.squeeze(means[:,:,:,:,0])
    print("mean1:", mean1.shape)
    mean2 = torch.squeeze(means[:,:,:,:,1])
    mean3 = torch.squeeze(means[:,:,:,:,2])
    sigma1 = torch.squeeze(sigmas[:,:,:,:,0])
    sigma2 = torch.squeeze(sigmas[:,:,:,:,1])
    sigma3 = torch.squeeze(sigmas[:,:,:,:,2])
    weight1, weight2, weight3 = torch.squeeze(weights[:,:,:,:,0]),torch.squeeze(weights[:,:,:,:,1]),torch.squeeze(weights[:,:,:,:,2])
    sigma1, sigma2, sigma3 = sigma1.clamp(1e-10, 1e10), sigma2.clamp(1e-10, 1e10), sigma3.clamp(1e-10, 1e10)
    gaussian1 = torch.distributions.laplace.Laplace(mean1, sigma1)
    gaussian2 = torch.distributions.laplace.Laplace(mean2, sigma2)
    gaussian3 = torch.distributions.laplace.Laplace(mean3, sigma3)
    prob1 = gaussian1.cdf(feature + 0.5) - gaussian1.cdf(feature - 0.5)
    prob2 = gaussian2.cdf(feature + 0.5) - gaussian2.cdf(feature - 0.5)
    prob3 = gaussian3.cdf(feature + 0.5) - gaussian3.cdf(feature - 0.5)
    # testweight = weight1 + weight2 + weight3
    # print("testweight: ", testweight)
    probs = weight1 * prob1 + weight2 * prob2 + weight3 * prob3
    total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
    return total_bits, probs

def feature_probs_based_sigma(feature, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-10, 1e10)
        gaussian1 = torch.distributions.laplace.Laplace(mu, sigma)
        gaussian2 = torch.distributions.laplace.Laplace(mu, sigma)
        # gaussian = gaussian1 + gaussian2
        print("gaussian: ", gaussian1)
        probs = gaussian1.cdf(feature + 0.5) - gaussian1.cdf(feature - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
        return total_bits, probs

def test_gaussian():
    mu = 0
    sigma = 0.5
    gaussian = torch.distributions.laplace.Laplace(mu, sigma)
    probs = gaussian.cdf(0.5) - gaussian.cdf(-0.5)
    print(probs)

if __name__ == "__main__":
    # test = ImageCompressor()
    # feature = torch.zeros([4, 128,32,48])
    # means = torch.zeros([4,128,32,48,3])
    # variances = torch.zeros([4,128,32,48,3])
    # probs = torch.zeros([4,128,32,48,3])
    # feature_probs_based_More_Gauss(feature, means, variances, probs)
    # feature_probs_based_sigma(feature, feature)
    total_bit1 = -1.0 * math.log(0.866 + 1e-10) / math.log(2.0)
    total_bit2 = -1.0 * math.log(0.8931 + 1e-10) / math.log(2.0)
    total_bit3 = -1.0 * math.log(0.956 + 1e-10) / math.log(2.0)
    total_bits = -1.0 * math.log(0.89 + 1e-10) / math.log(2.0)
    print(total_bit1, total_bit2, total_bit3, total_bits)
    # test_gaussian()
    

