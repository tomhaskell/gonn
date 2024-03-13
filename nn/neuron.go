package nn

type Neuron struct {
	Weights []float64 `json:"weights"`
	Biases []float64 `json:"biases"`
}

func NewNeuron(numInputs int) *Neuron {
	w := make([]float64, numInputs)
	b := make([]float64, numInputs)
	for i := 0; i < numInputs; i++ {
		w[i] = 1.0 / float64(numInputs)
		b[i] = 0.0
	}
	return &Neuron{
		Weights: w,
		Biases: b,
	}
}

func (n *Neuron) Process(inputs []float64) float64 {	
	sum := 0.0
	for i, input := range inputs {
		sum += n.Biases[i] + input * n.Weights[i]
	}
	return sum
}