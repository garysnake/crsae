import torch
import torch.nn.functional as F
import utils


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


"""
Help visualize the latent space sparsity
Input: model = "PCA" or "TSNE"
       data  = nparray dimension n * d, n is number of data, d is dimension of one data
       label = label class of data
"""

def visualize_latent_space(model, data, label):
    # Use TSNE model to visualize
    if model == "TSNE":
        visualize_model = TSNE(n_components=2, random_state=0)
    elif model == "PCA":
        visualize_model = PCA(n_components=2, random_state=0)

    data = visualize_model.fit_transform(data)

    data = np.vstack((data.T, label)).T
    tsne_df = pd.DataFrame(data=data, columns=('Dim_1','Dim_2', 'label'))

    sns.FacetGrid(tsne_df, hue="label", size=6).map(plt.scatter, 'Dim_1', 'Dim_2').add_legend()
    plt.show()

def visualize_img(img_list, label_list, img_size):
    num_row = int(np.sqrt(len(img_list)))
    num_col = int(np.ceil(len(img_list)/num_row))

    fig = plt.figure(figsize=(8,8))
    
    for i, (img, label) in enumerate(zip(img_list, label_list)):
        img = img.reshape(img_size)
        ax = fig.add_subplot(num_row, num_col, i+1)
        # ax.title.set_text(label)
        plt.imshow(img)
    plt.show()

def visualize_dictionary(img_tensor):
    num_row = int(np.sqrt(img_tensor.size()[0]))
    num_col = int(np.ceil(img_tensor.size()[0]/num_row))
    fig = plt.figure(figsize=(8,8))
    for i in range(img_tensor.size()[0]):
        img = img_tensor[i].numpy()
        ax = fig.add_subplot(num_row, num_col, i+1)
        plt.imshow(img)
    plt.show()


def split_image(x, stride, dictionary_dim, device):
        if stride == 1:
            return x, torch.ones_like(x)
        left_pad, right_pad, top_pad, bot_pad = utils.calc_pad_sizes(
            x, dictionary_dim, stride
        )
        x_batched_padded = torch.zeros(
            x.shape[0],
            stride ** 2,
            x.shape[1],
            top_pad + x.shape[2] + bot_pad,
            left_pad + x.shape[3] + right_pad,
            device=device,
        ).type_as(x)
        valids_batched = torch.zeros_like(x_batched_padded)
        for num, (row_shift, col_shift) in enumerate(
            [(i, j) for i in range(stride) for j in range(stride)]
        ):
            x_padded = F.pad(
                x,
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="reflect",
            )
            valids = F.pad(
                torch.ones_like(x),
                pad=(
                    left_pad - col_shift,
                    right_pad + col_shift,
                    top_pad - row_shift,
                    bot_pad + row_shift,
                ),
                mode="constant",
            )
            x_batched_padded[:, num, :, :, :] = x_padded
            valids_batched[:, num, :, :, :] = valids
        x_batched_padded = x_batched_padded.reshape(-1, *x_batched_padded.shape[2:])
        valids_batched = valids_batched.reshape(-1, *valids_batched.shape[2:])
        return x_batched_padded, valids_batched