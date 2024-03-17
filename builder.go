package gonn

import "github.com/tomhaskell/gonn/nn"

// Builder is a Builder pattern for creating a new gonn neural network
type Builder interface {
	SetInputCount(numInputs int) Builder
	AddLayer(numNeurons int) Builder
	Build() *Net
}

// NewBuilder creates a new Builder
func NewBuilder() Builder {
	return netBuilder{
		inputs: 0,
		layers: make([]int, 0),
	}
}

type netBuilder struct {
	inputs int
	layers []int
}

// SetInputCount sets the number of inputs for the neural network
func (n netBuilder) SetInputCount(numInputs int) Builder {
	n.inputs = numInputs
	return n
}
// AddLayer adds a layer to the neural network with the given number of neurons
func (n netBuilder) AddLayer(numNeurons int) Builder {
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