# CUDA-Enabled-VGG16-Replica
Replicates the VGG16 forward pass with CUDA Numba for parallel computing. 

The weights were obtained from the link https://gist.github.com/baraldilorenzo/07d7802847aaad0a35d3

# Trouble Shooting When Running on Google Colab

If the predictions from the VGG16 model are incorrect, try resetting all the runtimes in Google Colab. Sometimes, the outputs are incorrect due to the restrictions on the free GPUs and RAM in Google Colab. 
