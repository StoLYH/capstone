import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2):
        super(UNet, self).__init__()

        def conv_block(in_channels, out_channels): 
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        # kernel_size = 3 -> 연산에 3x3   
        # 커널의 크기가 클수록 필터가 한 번에 처리하는 이미지 영역이 넓어지므로, 더 많은 정보를 한 번에 처리하는 대신, 세밀한 정보는 덜 파악한다
        # padding(1)은 모서리 부분 값이 없을 때를 방지해서 -> 제로 패딩 
        
        # 인코더 에서 해상도를 줄여가며 이미지의 큰 윤곽을 추출한다. -> 이후 디코더에서 이미지 해상도 복구하며 세부적인 정보를 추출함
        # 인코더(다운 샘플링) -> out_channel(2)-(용접선, 전처리 안하면) 두가지를 예측하기 위해 512개 까지의 세부적인 패턴,특징 학습 
        self.encoder1 = conv_block(in_channels, 64) # 입력이미지에 대해서 64개의 출력 체널을 가진다. (64개의 다른 관점으로 표현)
        self.encoder2 = conv_block(64, 128)
        self.encoder3 = conv_block(128, 256)
        self.encoder4 = conv_block(256, 512)        # 점점 더 세부적인 특징, 패턴을 학습한다.

        self.pool = nn.MaxPool2d(2)                 # 복잡한 특징을 효율적으로 학습할 수 있도록 돕는 layer

        self.bottleneck = conv_block(512, 1024)


        # 디코더는 인코더 단계에서 특징 맵 추출을 위해 . 업샘플링 하여 원래 해상도로 복구한다
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = conv_block(128, 64)
        # upconv -> 다운샘플링된 텐서의 해상도를 높입니다.
        # decoder -> 

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)  # 최종적으로 출력 체널2개로 설정 (용접선, 전처리 안한면 특징 추출)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return torch.sigmoid(self.final_conv(dec1))
