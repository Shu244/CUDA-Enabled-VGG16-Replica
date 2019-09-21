import h5py
import numpy as np


def reformat_h5(path_in, path_out):
    file = h5py.File(path_in, 'r')

    # Creating file to save reformated data.
    hf = h5py.File(path_out, 'w')

    layer_count = 0
    for int_layer in (1, 3, 6, 8, 11, 13, 15, 18, 20, 22, 25, 27, 29, 32, 34, 36):
        layer = 'layer_' + str(int_layer)

        weights = file[layer]['param_0']
        biases = file[layer]['param_1']

        # Formats the convolutional weights to be in the expected form.
        if not int_layer in (32, 34, 36):
            weights = np.array(weights)
            weights = weights[:, :, ::-1, ::-1]

        layer_group = hf.create_group(str(layer_count))
        layer_count = layer_count + 1

        layer_group.create_dataset('weights', data=weights)
        layer_group.create_dataset('biases', data=biases)

        # Verify what is being saved.
        print(layer)
        print(weights.shape)
        print(biases.shape)

    hf.close()
    file.close()


def print_h5(path_in):
    file = h5py.File(path_in, 'r')

    for i in range(0, 16):
        weights = file[str(i)]['weights']
        biases = file[str(i)]['biases']

        print(str(i))
        print(weights.shape)
        print(biases.shape)

    file.close()


if __name__ == "__main__":
    old_file_name_h5 = 'C:\\Users\\Shuha\\Desktop\\Diabetic Retinopathy\\vgg16_weights.H5'
    new_file_name_h5 = 'C:\\Users\\Shuha\\Desktop\\Diabetic Retinopathy\\vgg16_weights_reformatted.h5'

    reformat_h5(old_file_name_h5, new_file_name_h5)
    print_h5(new_file_name_h5)

