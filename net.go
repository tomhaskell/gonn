package gonn

import "github.com/tomhaskell/gonn/nn"

// Net is a simple feedforward neural network
type Net struct {
	Layers []*nn.Layer `json:"layers"`
	NumInputs int `json:"numInputs"`
}

// NewNet creates a new neural network with the given layers
func NewNet(Layers []*nn.Layer, numInputs int) *Net {
	return &Net{
		Layers: Layers,
		NumInputs: numInputs,
	}
}

// FeedForward runs the input through the network and returns the output
func (n *Net) FeedForward(inputs []float64) []float64 {
	outputs := inputs
	for _, layer := range n.Layers {
		outputs = layer.Forward(outputs)
	}
	return outputs
}

// Calculate runs the input through the network and returns the z-vectors and activations of each 
// layer. This can be useful for debugging the network, or for training algorithms that require the
// intermediate values of the network (such as backpropagation).
func (n *Net) Calculate(inputs []float64) ([][]float64, [][]float64) {
	outputs := inputs
	zs := make([][]float64, len(n.Layers))
	activations := make([][]float64, len(n.Layers))
	for l, layer := range n.Layers {
		zs[l] = layer.Update(outputs)
		activations[l] = layer.Activation()
		outputs = activations[l]
	}
	return zs, activations
}