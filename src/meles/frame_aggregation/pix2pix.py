from typing import List

import torch
from torch import nn
from torchvision import transforms

SIZE = 256

pix2pix_transform = transforms.Compose([
    transforms.Resize((SIZE, SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class Pix2PixDiscriminator(nn.Module):

    def __init__(self, in_channels:int=3):
        super(Pix2PixDiscriminator, self).__init__()

        # helper function to construct layers quickly
        def conv_block(in_c, out_c, stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True)
            )

        # due to concatenated input of segmented+real, in_channels=in_channels*2
        self.model = nn.Sequential(
            nn.Conv2d(in_channels*2, 64, kernel_size=4, stride=2, padding=1),     # C64, no BatchNorm
            nn.LeakyReLU(0.2, inplace=True),

            conv_block(64, 128, stride=2),                                        # C128
            conv_block(128, 256, stride=2),                                       # C256
            conv_block(256, 512, stride=1),                                       # C512 (stride 1 for 70x70 patches)

            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),                # Final layer
            nn.Sigmoid()
        )


    def forward(self, x, y):

        concatenated = torch.cat([x, y], dim=1)
        verdict = self.model(concatenated)

        return verdict

class Pix2PixDownSample(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, apply_batch_norm: bool=True):
        super(Pix2PixDownSample, self).__init__()

        layers: List[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not apply_batch_norm)]
        if apply_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.down = nn.Sequential(*layers)


    def forward(self, x):

        return self.down(x)


class Pix2PixUpSample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int , apply_dropout: bool=False):
        super(Pix2PixUpSample, self).__init__()

        layers: List[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))

        self.up = nn.Sequential(*layers)


    def forward(self, x, skip):

        x = self.up(x)
        x = torch.cat([x, skip], dim=1)                                         # skip connection
        return x


class Pix2PixGenerator(nn.Module):

    def __init__(self, in_channels: int=3, out_channels: int=3):
        super(Pix2PixGenerator, self).__init__()

        # Encoder (DownSampling)
        self.down1 = Pix2PixDownSample(in_channels, 64, apply_batch_norm=False)         # C64
        self.down2 = Pix2PixDownSample(64, 128)                                        # C128
        self.down3 = Pix2PixDownSample(128, 256)                                       # C256
        self.down4 = Pix2PixDownSample(256, 512)                                       # C512
        self.down5 = Pix2PixDownSample(512, 512)                                       # C512
        self.down6 = Pix2PixDownSample(512, 512)                                       # C512
        self.down7 = Pix2PixDownSample(512, 512)                                       # C512
        self.down8 = Pix2PixDownSample(512, 512)                                       # C512

        # Decoder (Upsampling)
        self.up1 = Pix2PixUpSample(512, 512, apply_dropout=True)                       # CD512
        self.up2 = Pix2PixUpSample(1024, 512, apply_dropout=True)                      # CD1024
        self.up3 = Pix2PixUpSample(1024, 512, apply_dropout=True)                      # CD1024
        self.up4 = Pix2PixUpSample(1024, 512)                                          # C1024
        self.up5 = Pix2PixUpSample(1024, 256)                                          # C1024
        self.up6 = Pix2PixUpSample(512, 128)                                           # C512
        self.up7 = Pix2PixUpSample(256, 64)                                            # C256

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )


    def forward(self, x):

        # Encoder forward
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder forward + skip connections (U-Net)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

class Pix2Pix(nn.Module):

    def __init__(self, generator: Pix2PixGenerator, discriminator: Pix2PixDiscriminator):
        super(Pix2Pix, self).__init__()

        self.generator = generator
        self.discriminator = discriminator
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()


    def generator_loss(self, fake_output: torch.Tensor, fake_target: torch.Tensor, real_target: torch.Tensor, lambda_l1: float=100.0) -> float:

        adversarial_loss = self.criterion_gan(fake_output, torch.ones_like(fake_output, device=fake_output.device))
        l1_loss = self.criterion_l1(fake_target, real_target)

        total_loss = adversarial_loss + lambda_l1 * l1_loss
        return total_loss


    def discriminator_loss(self, real_output: torch.Tensor, fake_output: torch.Tensor) -> float:

        real_loss = self.criterion_gan(real_output, torch.ones_like(real_output, device=real_output.device))
        fake_loss = self.criterion_gan(fake_output, torch.zeros_like(fake_output, device=fake_output.device))

        total_loss = (real_loss + fake_loss) * 0.5
        return total_loss
