package gonn

import "github.com/tomhaskell/gonn/nn"

type Net struct {
	Layers []*nn.Layer `json:"layers"`
}

func NewNet(Layers []*nn.Layer) *Net {
	return &Net{
		Layers: Layers,
	}
}

func (n *Net) Process(inputs []float32) []float32 {
	outputs := inputs
	for _, layer := range n.Layers {
		outputs = layer.Process(outputs)
	}
	return outputs
}