package gonn

import "github.com/tomhaskell/gonn/nn"

type NetBuilder interface {
	SetInputCount(numInputs int) NetBuilder
	AddLayer(numNeurons int) NetBuilder
	Build() *Net
}

func NewNetBuilder() NetBuilder {
	return netBuilder{
		inputs: 0,
		layers: make([]int, 0),
	}
}

type netBuilder struct {
	inputs int
	layers []int
}

func (n netBuilder) SetInputCount(numInputs int) NetBuilder {
	n.inputs = numInputs
	return n
}
func (n netBuilder) AddLayer(numNeurons int) NetBuilder {
	n.layers = append(n.layers, numNeurons)
	return n
}
func (n netBuilder) Build() *Net {
	l := make([]*nn.Layer, len(n.layers))
	numInputs := n.inputs
	for i, numNeurons := range n.layers {
		l[i] = nn.NewLayer(numNeurons, numInputs)
		numInputs = numNeurons
	}
	return NewNet(l)
}