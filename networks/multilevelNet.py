import torch.nn as nn
import torch
import math

class multilevelNet(nn.Module):
    def __init__(self, channel_settings, output_shape, num_class):
        super(multilevelNet, self).__init__()
        self.channel_settings = channel_settings
        laterals, upsamples, predict = [], [], []
        for i in range(len(channel_settings)):
            laterals.append(self._lateral(channel_settings[i]))
            predict.append(self._predict(output_shape, num_class))
            if i != len(channel_settings) - 1:
                upsamples.append(self._upsample())
        self.laterals = nn.ModuleList(laterals)
        self.upsamples = nn.ModuleList(upsamples)
        self.predict = nn.ModuleList(predict)


        pool_ratios = [0.1,0.2,0.3]
        self.adaptive_pool_output_ratio = pool_ratios
        self.high_lateral_conv = nn.ModuleList()
        self.high_lateral_conv.extend(
            [nn.Conv2d(2048, 256, 1) for k in range(len(self.adaptive_pool_output_ratio))])
        self.high_lateral_conv_attention = nn.Sequential(
            nn.Conv2d(256 * (len(self.adaptive_pool_output_ratio)), 256, 1), nn.ReLU(),
            nn.Conv2d(256, len(self.adaptive_pool_output_ratio), 3, padding=1))


        for m in self.high_lateral_conv_attention.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()





        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def _lateral(self, input_size):
        layers = []
        layers.append(nn.Conv2d(input_size, 256,
                                kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        return nn.Sequential(*layers)

    def _upsample(self):
        layers = []
        layers.append(
            nn.ConvTranspose2d(
                in_channels=256,
                out_channels=256,
                kernel_size=4,
                stride=2,
                padding=1,
                output_padding=0,
                bias=False))
        layers.append(nn.BatchNorm2d(256, momentum=0.1))
        return nn.Sequential(*layers)

    def _predict(self, output_shape, num_class):
        layers = []
        layers.append(nn.Conv2d(256, 256,
            kernel_size=1, stride=1, bias=False))
        layers.append(nn.BatchNorm2d(256))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(256, num_class,
            kernel_size=3, stride=1, padding=1, bias=False))
        layers.append(nn.Upsample(size=output_shape, mode='bilinear', align_corners=True))
        layers.append(nn.BatchNorm2d(num_class))

        return nn.Sequential(*layers)

    def forward(self, x):
        h, w = 12, 9
        AdapPool_Features = [torch.nn.functional.upsample(
            self.high_lateral_conv[j](torch.nn.functional.adaptive_avg_pool2d(x[0], output_size=(
                max(1, int(h * self.adaptive_pool_output_ratio[j])),
                max(1, int(w * self.adaptive_pool_output_ratio[j]))))),
            size=(h, w), mode='bilinear', align_corners=True) for j in
                             range(len(self.adaptive_pool_output_ratio))]
        Concat_AdapPool_Features = torch.cat(AdapPool_Features, dim=1)
        fusion_weights = self.high_lateral_conv_attention(Concat_AdapPool_Features)
        fusion_weights = torch.nn.functional.sigmoid(fusion_weights)
        adap_pool_fusion = 0
        for i in range(len(self.adaptive_pool_output_ratio)):
            adap_pool_fusion += torch.unsqueeze(fusion_weights[:, i, :, :], dim=1) * AdapPool_Features[i]

        _fms, _outs = [], []
        for i in range(len(self.channel_settings)):
            if i == 0:
                feature = self.laterals[i](x[i])+adap_pool_fusion
            else:
                feature = up
            _fms.append(feature)
            if i != len(self.channel_settings) - 1:
                up = self.upsamples[i](feature)
            feature = self.predict[i](feature)
            _outs.append(feature)


        return _fms, _outs
