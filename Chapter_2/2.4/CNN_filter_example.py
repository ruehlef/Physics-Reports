from __future__ import print_function

# Load Libraries
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms.functional as TF

# seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

import matplotlib as mpl  # for plotting
mpl.use('TkAgg')  # needed for OSX, comment out otherwise

import matplotlib.pyplot as plt  # for plotting
from matplotlib.colors import LinearSegmentedColormap  # colormap to plot 1 channel convolution
plt.rc('text', usetex=True)  # LaTex support for captions


####################
# Do simple CNN
####################

def output_size(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return output


def save_pic(pic_tensor, title, filename):
    """Helper file to save picture with specified title"""
    plt.close()
    plt.title(title, size=24)
    xyrgb = pic_tensor.data[0].numpy().transpose((1, 2, 0))  # convert from RGB x X x Y to X x Y x RGB
    if (xyrgb.shape)[-1] != 3:  # no rgb image => Plot with color map
        xyrgb = xyrgb[:, :, 0]
    plt.imshow(xyrgb, cmap='gray', interpolation='nearest')
    if (xyrgb.shape)[-1] != 3:   # no rgb image => show color bar legend
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=20)
    print(title, xyrgb.shape)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.savefig("./" + filename + ".pdf", dpi=300, bbox_inches='tight')


# inherit from PyTorch's Module class and implement the abstract forward method
class ExampleCNN(torch.nn.Module):

    def __init__(self):
        super(ExampleCNN, self).__init__()

        # Input channels = 3, output channels = 1
        self.conv1 = torch.nn.Conv2d(3, 1, kernel_size=16, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(3, 1, kernel_size=30, stride=10, padding=0)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=10, stride=4, padding=0)

    # This abuses the forward function. It does not really perform a forward pass but merely 4 individual operations
    # Note that each conv/pooling starts again with x abd does not use the y from the previous operation
    def forward(self, x):
        ####################
        # 2d convolutions
        ####################
        y = self.conv1(x)
        # y = F.relu(y)  # do not activate for illustration purpose
        save_pic(y, r"$\mathrm{Convolution~with}~(k,s)=(16,1)$", "CNN_after_conv1")
        y = self.conv2(x)
        # y = F.relu(y)  # do not activate for illustration purpose
        save_pic(y, r"$\mathrm{Convolution~with}~(k,s)=(30,10)$", "CNN_after_conv2")

        ####################
        # 2d max pooling
        ####################
        y = self.pool1(x)
        save_pic(y, r"$\mathrm{Pooling~with}~(k,s)=(2,2)$", "CNN_after_pooling1")
        y = self.pool2(x)
        save_pic(y, r"$\mathrm{Pooling~with}~(k,s)=(6,2)$", "CNN_after_pooling2")

        return x


# read in image, transform channels to match what PyTorch expects
example_pic = Image.open('./example.png')
x = TF.to_tensor(example_pic)
x = x[:3, :, :]  # remove alpha channel
orig = x.data.numpy().transpose((1, 2, 0))
x.unsqueeze_(0)
print(x.shape)
save_pic(x, "Original", "CNN_original")

# initiate CNN
model = ExampleCNN()
# apply CNN (i.e. the 2 convolutions and 2 max pooling operations) to the image
output = model.forward(x)

############################################################
# Show the effect of overlapping filters with small strides
############################################################
res = []
stride = 50
filter_size = 100
filler_pixel = [1., 1., 1.]
row_strided = 0
col_strided = 0
i, j = 0, 0
# this code was written quickly to illustrate the effect of overlapping filters. It is neither very efficient nor is it
# intended to work for a general image. For example, it does not offer padding,...
while i < orig.shape[0]:
    row = list([])
    j = 0
    col_strided = 0
    if i % (filter_size + row_strided) == 0 and i != 0:  # insert some filler_pixeled boundary rows in between two patches scanned by the kernel
        row = [filler_pixel for _ in range(len(res[-1]))]
        res += [row]
        res += [row]
        res += [row]
        res += [row]
        row_strided += stride
        i = row_strided

    else:
        while j < orig.shape[1]:
            if j % (filter_size + col_strided) == 0 and j != 0:  # insert some filler_pixeled boundary columns in between two patches scanned by the kernel
                row += [filler_pixel]
                row += [filler_pixel]
                row += [filler_pixel]
                row += [filler_pixel]
                row += [filler_pixel]
                col_strided += stride
                j = col_strided
            else:
                row += [list(orig[i][j])]
            j += 1

    res += [row]
    i += 1

# Save the picture
plt.close()
plt.title(r"$\mathrm{Convolved~image}$", size=24)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.imshow(res)
plt.savefig("./CNN_example_explodedImage.pdf", dpi=300, bbox_inches='tight')
