# MNIST example

This example shows how to use the GoNN package to train a neural net for numerical handwriting
recognition using the [MNIST dataset converted to csv format](https://pjreddie.com/projects/mnist-in-csv/).
Download them and put them in an `/examples/mnist/.data` directory.

You can then run the example from this directory using `go run .`, you should see an output similar
to the following:

```
Epoch 0: 9055/10000 correct (90%)
Epoch 1: 9245/10000 correct (92%)
Epoch 2: 9323/10000 correct (93%)
Epoch 3: 9366/10000 correct (93%)
Epoch 4: 9424/10000 correct (94%)
Epoch 5: 9442/10000 correct (94%)
...
Epoch 25: 9560/10000 correct (95%)
Epoch 26: 9561/10000 correct (95%)
Epoch 27: 9552/10000 correct (95%)
Epoch 28: 9550/10000 correct (95%)
Epoch 29: 9558/10000 correct (95%)
```

If you want to change the paramaters of the neural network, the following options are available:

```
--layers        an array specifying the number of neurons in each hidden layer (default "24 16")
--epochs        the number of training epochs to train the network for (default 30)
--learnRate  the learning rate to use in the backprop algorithm (default 0.7)
--batchsize     the size of mini-batch to use for stochastic gradient decsent (default 20)
--act <sigmoid|relu|linear>  activation function (default "sigmoid")
```
