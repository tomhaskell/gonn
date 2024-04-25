package slownn

import "github.com/tomhaskell/gonn/nnmath"

// Builder is a Builder pattern for creating a new gonn neural network
type Builder interface {
	SetDefaultActivation(activationType string) Builder
	SetInputCount(numInputs int) Builder
	AddLayer(numNeurons int) Builder
	AddLayerWithActivation(numNeurons int, activationType string) Builder
	Build() *Net
}

// NewBuilder creates a new Builder
func NewBuilder() Builder {
	return &netBuilder{
		defaultActivation: nnmath.SIGMOID,
		inputs:            0,
		layers:            make([]int, 0),
		activations:       make([]string, 0),
	}
}

type netBuilder struct {
	defaultActivation string
	inputs            int
	layers            []int
	activations       []string
}

// SetType sets the activation function type for the neural network
func (n netBuilder) SetDefaultActivation(activationType string) Builder {
	n.defaultActivation = activationType
	return n
}

// SetInputCount sets the number of inputs for the neural network
func (n netBuilder) SetInputCount(numInputs int) Builder {
	n.inputs = numInputs
	return n
}

// AddLayer adds a layer to the neural network with the given number of neurons, using the default activation function
func (n netBuilder) AddLayer(numNeurons int) Builder {
	return n.AddLayerWithActivation(numNeurons, n.defaultActivation)
}

// AddLayerWithActivation adds a layer to the neural network with the given number of neurons, using the specified activation function
func (n netBuilder) AddLayerWithActivation(numNeurons int, activationType string) Builder {
	n.layers = append(n.layers, numNeurons)
	n.activations = append(n.activations, activationType)
	return n
}

func (n netBuilder) Build() *Net {
	l := make([]*Layer, len(n.layers))
	numInputs := n.inputs
	for i, numNeurons := range n.layers {
		l[i] = NewLayer(n.activations[i], numNeurons, numInputs)
		numInputs = numNeurons
	}
	return NewNet(l, n.inputs)
}
