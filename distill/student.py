import math
import torch 
from distill.resNext import resnext18


class ConvClassifier(torch.nn.Module):
    """Convolutional text sequence classifier used as student.

    Model Design:
    * For model trunk use variation on 2016 resNext model (arXiv:1611.05431v2)
    tailored for 1D sequences (instead of 2D images) as this should make
    ~efficient use of parameters.
    * As aim here is efficiency use resNext-18 instead of {34, 50, 101, 152}.
    * Benefit of this architecture is that it is tried and tested + also
    the 2D version is available in PyTorch library at (this link)[https://github.com/pytorch/vision/blob/052edcecef3eb0ae9fe9e4b256fa2a488f9f395b/torchvision/models/resnet.py#L86]
    * Do not use any attention-based residual blocks/models as this will
    singnificantly increase inference time which defeats the point of the
    distiallation.
    """

    def __init__(self, glove_dim=50):
        super().__init__()
        resnext_kwargs = {
            'zero_init_residual': True,
            'num_classes': 3,  # i.e. three class output
            'layers': [2, 2, 2, 2],
            'Cin': glove_dim,
        }
        self.trunk = resnext18(**resnext_kwargs)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, lengths=None):
        # x: B, T, H -> out: B
        x = x.transpose(1, 2)
        x = self.trunk(x) # B, 3
        return self.softmax(x)


def test_conv_classifier():
    model = ConvClassifier()
    x = torch.randn(5, 12, 50)

    res = model(x)
    assert res.shape == (5, 3)
    assert ((0 < res) * (res < 1)).all()
    assert math.isclose(torch.sum(res), 5)
    print('tests run + passed')

if __name__ == '__main__':
    test_conv_classifier()
