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

