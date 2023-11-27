# mnist-pytorch: convolutional neural net for digit recognition
Bart Massey 2023

This code is almost entirely taken from
<https://nextjournal.com/gkoehler/pytorch-mnist>. This is a
most excellent article on the topic of digit classification
on the MNIST Handwritten Digit dataset using a convolutional
neural net.

My modifications include moving the data to CUDA if
available, reducing the error output, arranging to save the
model, and moving the parameters to be program arguments.
