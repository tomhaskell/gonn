package nn

type Layer struct {
	Neurons []*Neuron `json:"neurons"`
}

func NewLayer(numNeurons, numInputs int) *Layer {
	n := make([]*Neuron, numNeurons)
	for i := 0; i < numNeurons; i++ {
		n[i] = NewNeuron(numInputs)
	}
	return &Layer{
		Neurons: n,
	}
}

func (l *Layer) Process(inputs []float32) []float32 {
	outputs := make([]float32, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Process(inputs)
	}
	return outputs
}
