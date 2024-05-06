from image_processing import split_image, calculate_entropy, calculate_complexity, discard_images
from test_model import test_model
from vgg_model import ModifiedVGG16Model, FusionVGG16Model
import torch

if __name__ == '__main__':
    
    file = "image_test.png"
    images_directory = "./images_cropped"
    output_directory = "./output_images"
    split_width = 256
    overlap_percentage = 0.6
    split_image(file, images_directory, split_width, overlap_percentage)

    discard_images(images_directory, 7.0, 0.40)
    
    classes = ["arma de fuego", "no arma de fuego"]
    print("Testing model")
    path_model = "model_Vgg16"
    model = torch.load(path_model, map_location=torch.device('cpu'))
    test_model(model, images_directory, file, output_directory, split_width, classes)