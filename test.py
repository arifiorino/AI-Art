#!/usr/bin/python3
import torch, matplotlib.pyplot as plt

class Generator(torch.nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(g_input , 96 * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(96 * 8),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d(96 * 8, 96 * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(96 * 4),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d( 96 * 4, 96 * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(96 * 2),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d( 96 * 2, 96, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(96),
            torch.nn.ReLU(True),
            torch.nn.ConvTranspose2d( 96, 3, 5, 3, 1, bias=False),
            torch.nn.Sigmoid())
    def forward(self, input):
        return self.main(input)

g_input=150

device = torch.device('cpu')
#Gs=[torch.load('Generator_epoch_%d.pth'%i,map_location=torch.device('cpu')) for i in range(0,60,5)]
G=torch.load('res/Generator_epoch_145.pth',map_location=torch.device('cpu'))

#for i in range(20):
while 1:
  #for G in Gs:
  noise = (torch.rand(1, g_input, 1, 1) - 0.5) / 0.5
  noise = noise.to(device)
  output = G(noise)
  plt.imshow(output[0].permute(1, 2, 0).detach().cpu())
  plt.show()
  #plt.savefig('out/%d.png'%i)
