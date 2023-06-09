import torch
import torch.nn as nn

__all__ = ['dense_u']

class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(ch_in, ch_out, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm3d(ch_out),
        )

    def forward(self, x):
        return self.up(x)


class conv_final(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_final, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=1, stride=1),
            nn.BatchNorm3d(ch_out),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class dense_u(nn.Module):
    def __init__(self):
        super(dense_u, self).__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

        self.conv11 = conv_block(1, 32)
        self.conv12 = conv_block(1 + 32, 32)

        self.conv21 = conv_block(1 + 32, 64)
        self.conv22 = conv_block(1 + 32 + 64, 64)

        self.conv31 = conv_block(1 + 32 + 64, 128)
        self.conv32 = conv_block(1 + 32 + 64 + 128, 128)

        self.conv41 = conv_block(1 + 32 + 64 + 128, 256)
        self.conv42 = conv_block(1 + 32 + 64 + 128 + 256, 256)

        self.conv51 = conv_block(1 + 32 + 64 + 128 + 256, 512)
        self.conv52 = conv_block(1 + 32 + 64 + 128 + 256 + 512, 512)

        self.up1 = up_conv(1 + 32 + 64 + 128 + 256 + 512, 256)
        self.up2 = up_conv(1 + 32 + 64 + 128 + 256 + 256, 128)
        self.up3 = up_conv(1 + 32 + 64 + 128 + 128, 64)
        self.up4 = up_conv(1 + 32 + 64 + 64, 32)

        self.conv61 = conv_block(256, 256)
        self.conv62 = conv_block(512, 256)

        self.conv71 = conv_block(128, 128)
        self.conv72 = conv_block(256, 128)

        self.conv81 = conv_block(64, 64)
        self.conv82 = conv_block(128, 64)

        self.conv91 = conv_block(32, 32)
        self.conv92 = conv_block(64, 32)

        self.final = conv_final(64, 1)

    def forward(self, x):
        # encoder
        en11 = self.conv11(x)
        en_c11 = torch.cat((x, en11), dim=1)
        en12 = self.conv12(en_c11)
        en_c12 = torch.cat((x, en12), dim=1)
        pool1 = self.pool(en_c12)

        en21 = self.conv21(pool1)
        en_c21 = torch.cat((pool1, en21), dim=1)
        en22 = self.conv22(en_c21)
        en_c22 = torch.cat((pool1, en22), dim=1)
        pool2 = self.pool(en_c22)

        en31 = self.conv31(pool2)
        en_c31 = torch.cat((pool2, en31), dim=1)
        en32 = self.conv32(en_c31)
        en_c32 = torch.cat((pool2, en32), dim=1)
        pool3 = self.pool(en_c32)

        en41 = self.conv41(pool3)
        en_c41 = torch.cat((pool3, en41), dim=1)
        en42 = self.conv42(en_c41)
        en_c42 = torch.cat((pool3, en42), dim=1)
        pool4 = self.pool(en_c42)

        en51 = self.conv51(pool4)
        en_c51 = torch.cat((pool4, en51), dim=1)
        en52 = self.conv52(en_c51)
        en_c52 = torch.cat((pool4, en52), dim=1)

        # decoder
        de_up1 = self.up1(en_c52)
        de61 = self.conv61(de_up1)
        de_c61 = torch.cat((de61, de_up1), dim=1)
        de62 = self.conv62(de_c61)
        de_c62 = torch.cat((de62, de_up1), dim=1)
        de_cross1 = torch.cat((de_c62, pool3), dim=1)

        de_up2 = self.up2(de_cross1)
        de71 = self.conv71(de_up2)
        de_c71 = torch.cat((de71, de_up2), dim=1)
        de72 = self.conv72(de_c71)
        de_c72 = torch.cat((de72, de_up2), dim=1)
        de_cross2 = torch.cat((de_c72, pool2), dim=1)

        de_up3 = self.up3(de_cross2)
        de81 = self.conv81(de_up3)
        de_c81 = torch.cat((de81, de_up3), dim=1)
        de82 = self.conv82(de_c81)
        de_c82 = torch.cat((de82, de_up3), dim=1)
        de_cross3 = torch.cat((de_c82, pool1), dim=1)

        de_up4 = self.up4(de_cross3)
        de91 = self.conv91(de_up4)
        de_c91 = torch.cat((de91, de_up4), dim=1)
        de92 = self.conv92(de_c91)
        de_c92 = torch.cat((de92, de_up4), dim=1)

        out = self.final(de_c92)

        return out
