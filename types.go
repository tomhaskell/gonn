package gonn

type Layer interface {
	// Update takes a slice of inputs from the previous layer and returns the z-vector of the layer
	Update(inputs []float64) []float64
	// Activation returns the activation vector of the layer based on a previous call to Update
	Activation() []float64
	// Forward takes a slice of inputs from the previous layer and returns the activation vector of
	// the layer.
	//
	// This is most often used as a convenience method that calls Update followed by Activation in a
	// single call, intended for use when the NN is "in use".
	//
	// If you're implementing this interface, the code will likely look like:
	// ```
	// func (l Layer) Forward(inputs []float64) []float64 {
	// 	l.Update(inputs)
	// 	return l.Activation()
	// }
	// ```
	Forward(inputs []float64) []float64
}

type Net interface {

	// FeedForward runs the input through the network and returns the output
	FeedForward(inputs []float64) []float64

	// Calculate runs the input through the network and returns the z-vectors and activations of each
	// layer. This can be useful for debugging the network, or for training algorithms that require the
	// intermediate values of the network (such as backpropagation).
	Calculate(inputs []float64) ([][]float64, [][]float64)
}

type Trainer interface {
	Train(net Net, inputs, targets *[][]float64, epochs int)
	TrainEpoch(net Net, inputs, targets *[][]float64)
}
