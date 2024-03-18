package nn

import (
	"math/rand"

	"github.com/tomhaskell/gonn/nnmath"
)

// Activation function types
const (
	// SIGMOID is the sigmoid activation function
	SIGMOID = "sigmoid"
	// RELU is the rectified linear unit activation function
	RELU = "relu"
	// LINEAR is the linear activation function
	LINEAR = "linear"
)

// Neuron represents a single neuron in a neural network
type Neuron struct {
	Type    string    `json:"type"`
	Weights []float64 `json:"weights"`
	Bias    float64   `json:"bias"`
}

// NewNeuron creates a new neuron with the given number of inputs, and using the given activation
// function.
//
// The weights are initialized to random values between -1 and 1, and the bias is initialized to 0.
// The activation function can be one of the following:
// - SIGMOID (default): f(x) = 1 / (1 + e^(-x))
// - RELU: f(x) = max(0, x)
// - LINEAR: f(x) = x
func NewNeuron(activationType string, numInputs int) *Neuron {
	w := make([]float64, numInputs)
	for i := 0; i < numInputs; i++ {
		w[i] = rand.Float64()*2.0 - 1.0 // random number between -1 and 1
	}
	return &Neuron{
		Type:    activationType,
		Weights: w,
		Bias:    0.0,
	}
}

// Process takes a slice of inputs and returns the output of the neuron
func (n *Neuron) Process(inputs []float64) float64 {
	sum := n.Bias
	for i, input := range inputs {
		sum += input * n.Weights[i]
	}
	switch n.Type {
	case SIGMOID:
		return nnmath.Sigmoid(sum)
	case RELU:
		if sum < 0 {
			return 0
		}
		return sum
	case LINEAR:
		return sum
	}
	return 0
}
