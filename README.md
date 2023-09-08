# GPU_Week5
CUDA at Scale for the Enterprise course project

Image standardization is used to transforms the pixel values of an image to have a mean of 0 and a standard deviation of 1.
This is accompished by finding the mean and standard deviation of all pixels in the image, subtracting the mean and dividing
by the standard deviation for each pixel.

    standardized_image = (pixel_value - image_mean) / image_std_dev

Standardizing the images is useful for Machine Learning, especially for Neural Networks. It can help speed the convergence
of parameters to a global optimum by making all features have similar ranges. Smaller values also tend to help in speeding
the convergence.

The code in this repository uses a subset of the MNIST dataset. It will go through all images in the directory and save
the standardized image to the ./standard folder using the same name as the original file with an std appended to the
end of the name (before the .)
