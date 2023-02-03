#!/usr/bin/python3
import torch, torchvision, matplotlib.pyplot as plt

epochs=150
batch_size=100
g_input=150

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

class Discriminator(torch.nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.main = torch.nn.Sequential(
      torch.nn.Conv2d(3, 96, 5, 3, 1, bias=False),
      torch.nn.LeakyReLU(0.2, inplace=True),
      torch.nn.Conv2d(96, 96 * 2, 4, 2, 1, bias=False),
      torch.nn.BatchNorm2d(96 * 2),
      torch.nn.LeakyReLU(0.2, inplace=True),
      torch.nn.Conv2d(96 * 2, 96 * 4, 4, 2, 1, bias=False),
      torch.nn.BatchNorm2d(96 * 4),
      torch.nn.LeakyReLU(0.2, inplace=True),
      torch.nn.Conv2d(96 * 4, 96 * 8, 4, 2, 1, bias=False),
      torch.nn.BatchNorm2d(96 * 8),
      torch.nn.LeakyReLU(0.2, inplace=True),
      torch.nn.Conv2d(96 * 8, 1, 4, 1, 0, bias=False),
      torch.nn.Sigmoid())
  def forward(self, x):
    return self.main(x)


training_data = torchvision.datasets.STL10('data/', 'unlabeled', transform=torchvision.transforms.transforms.ToTensor(), download=True)
training_dataloader = torch.utils.data.DataLoader(training_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda')
G=Generator().to(device)
D=Discriminator().to(device)

loss = torch.nn.BCELoss()
optimizerD = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(epochs):
  for idx,(imgs,_) in enumerate(training_dataloader):
    real_inputs = imgs.to(device)
    real_outputs = D(real_inputs)
    real_label = torch.ones(batch_size, 1).to(device)

    noise = (torch.rand(batch_size, g_input, 1, 1) - 0.5) / 0.5
    noise = noise.to(device)
    fake_inputs = G(noise)
    fake_outputs = D(fake_inputs)
    fake_label = torch.zeros(batch_size, 1).to(device)

    outputs = torch.cat((real_outputs.view(-1).unsqueeze(1), fake_outputs.view(-1).unsqueeze(1)), 0)
    targets = torch.cat((real_label, fake_label), 0)

    D_loss = loss(outputs, targets)
    optimizerD.zero_grad()
    D_loss.backward()
    optimizerD.step()

    noise = (torch.rand(batch_size, g_input, 1, 1)-0.5)/0.5
    noise = noise.to(device)
    fake_inputs = G(noise)
    fake_outputs = D(fake_inputs)
    fake_targets = torch.ones([fake_inputs.shape[0], 1]).to(device)
    G_loss = loss(fake_outputs.view(-1).unsqueeze(1), fake_targets)
    optimizerG.zero_grad()
    G_loss.backward()
    optimizerG.step()

    if idx % 100 == 0 or idx == len(training_dataloader):
      print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'.format(epoch, idx, D_loss.item(), G_loss.item()))
  if epoch % 5 == 0:
    torch.save(G, 'Generator_epoch_{}.pth'.format(epoch))
    print('Model saved.')


