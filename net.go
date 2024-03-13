package gonn

import "github.com/tomhaskell/gonn/nn"

type Net struct {
	Layers []*nn.Layer `json:"layers"`
}

type NetBuilder interface {
	AddLayer(numNeurons, numInputs int) NetBuilder
	Build() *Net
}

func NewNetBuilder() NetBuilder {
	return netBuilder{
		layers: make([]*nn.Layer, 0),
	}
}

type netBuilder struct {
	layers []*nn.Layer
}

func (n netBuilder) AddLayer(numNeurons, numInputs int) NetBuilder {
	n.layers = append(n.layers, nn.NewLayer(numNeurons, numInputs))
	return n
}

func (n netBuilder) Build() *Net {
	return &Net{
		Layers: n.layers,
	}
}

func (n *Net) Process(inputs []float32) []float32 {
	outputs := inputs
	for _, layer := range n.Layers {
		outputs = layer.Process(outputs)
	}
	return outputs
}