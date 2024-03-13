package nn

import "math/rand"

type Neuron struct {
	Weights []float32 `json:"weights"`
	Bias  float32 `json:"bias"`
}

func NewNeuron(numInputs int) *Neuron {
	w := make([]float32, numInputs)
	for i := 0; i < numInputs; i++ {
		w[i] = rand.Float32()*2.0 - 1.0  // random number between -1 and 1
	}
	return &Neuron{
		Weights: w,
		Bias:  0.0,
	}
}

func (n *Neuron) Process(inputs []float32) float32 {
	sum := n.Bias
	for i, input := range inputs {
		sum += input*n.Weights[i]
	}
	return sum
}
