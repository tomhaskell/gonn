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
func (n *Net) Process(inputs []float32) []float32 {
	outputs := inputs
	for _, layer := range n.Layers {
		outputs = layer.Process(outputs)
	}
	return outputs
}