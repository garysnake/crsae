import sys, os


import torch
from pytorch_msssim import MS_SSIM
from tqdm import tqdm

import model, generator


def main(argv):
    MODEL_PATH = argv[0]
    # DATA_PATH = argv[1]

    if not os.path.exists(argv[0]):
        print("Not Valid Path")
    elif os.path.isdir(argv[0]):
        print("File is a directory")
    
    result_model = torch.load(MODEL_PATH)

    # data_loader = generator.get_path_loader(1, DATA_PATH, shuffle=False)
    data_loader = generator.get_MNIST_loader(1, trainable=False, shuffle=False)
    device = "cpu"

    dictionary = result_model.get_param("H").data
    print(dictionary.size())

    # for idx, (img, _) in tqdm(enumerate(data_loader)):
    #     img = img.to(device)
    #     img_hat, x_new, _ = result_model(img_test_noisy)




    

if __name__ == "__main__":
    main(sys.argv[1:])
    

# python analysis.py /home/garysnake/Desktop/crsae/pytorch/results/default/2020_04_11_10_51_59/model_init.pt

    
