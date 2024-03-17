package nn

// Layer represents a layer of neurons in a neural network
type Layer struct {
	Neurons []*Neuron `json:"neurons"`
}

// NewLayer creates a new layer of `numNeurons` neurons with the given activation type and number of inputs
func NewLayer(activationType string, numNeurons, numInputs int) *Layer {
	n := make([]*Neuron, numNeurons)
	for i := 0; i < numNeurons; i++ {
		n[i] = NewNeuron(activationType, numInputs)
	}
	return &Layer{
		Neurons: n,
	}
}

// Process takes a slice of inputs from the previous layer and returns the output of the layer
func (l *Layer) Process(inputs []float32) []float32 {
	outputs := make([]float32, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Process(inputs)
	}
	return outputs
}
