package slownn

// Layer represents a layer of neurons in a neural network
type Layer struct {
	Neurons []*Neuron `json:"neurons"`
	Type    string
}

// NewLayer creates a new layer of `numNeurons` neurons with the given activation type and number of inputs
func NewLayer(activationType string, numNeurons, numInputs int) *Layer {
	n := make([]*Neuron, numNeurons)
	for i := 0; i < numNeurons; i++ {
		n[i] = NewNeuron(activationType, numInputs)
	}
	return &Layer{
		Neurons: n,
		Type:    activationType,
	}
}

// Update takes a slice of inputs from the previous layer and returns the z-vector of the layer
func (l *Layer) Update(inputs []float64) []float64 {
	zs := make([]float64, len(l.Neurons))
	for i, neuron := range l.Neurons {
		zs[i] = neuron.Update(inputs)
	}
	return zs
}

// Activation returns the activation vector of the layer based on a previous call to Update
func (l *Layer) Activation() []float64 {
	outputs := make([]float64, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Activation()
	}
	return outputs
}

// Forward takes a slice of inputs from the previous layer and returns the activation vector of the layer.
//
// This is a convenience method that calls Update followed by Activation in a single call.
func (l *Layer) Forward(inputs []float64) []float64 {
	l.Update(inputs)
	return l.Activation()
}
