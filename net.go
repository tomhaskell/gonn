package gonn

import "github.com/tomhaskell/gonn/nn"

// Net is a simple feedforward neural network
type Net struct {
	Layers []*nn.Layer `json:"layers"`
}

// NewNet creates a new neural network with the given layers
func NewNet(Layers []*nn.Layer) *Net {
	return &Net{
		Layers: Layers,
	}
}

// Process runs the input through the network and returns the output
func (n *Net) FeedForward(inputs []float64) []float64 {
	outputs := inputs
	for _, layer := range n.Layers {
		layer.Forward(outputs)
	}
	return outputs
}
