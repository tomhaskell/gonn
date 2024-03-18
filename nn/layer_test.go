package nn

import (
	"testing"
)

func TestLayerProcess(t *testing.T) {
	l := &Layer{
		Neurons: []*Neuron{
			{
				Type:    LINEAR,
				Weights: []float64{0.3, -0.8},
				Bias:    0.7,
			}, {
				Type:    LINEAR,
				Weights: []float64{0.7, 0.15},
				Bias:    -0.57,
			},
		},
	}

	out := l.Process([]float64{0.1, 0.2})

	expected := []float64{
		0.1*0.3 + 0.2*-0.8 + 0.7,   // 0.57
		0.1*0.7 + 0.2*0.15 + -0.57, // -0.47
	}

	for i := range out {
		if out[i] != expected[i] {
			t.Errorf("Expected %v, got %v", expected[i], out[i])
		}
	}
}

func TestNewLayer(t *testing.T) {
	l := NewLayer(LINEAR, 3, 4)

	// check for 3 neurons
	if len(l.Neurons) != 3 {
		t.Errorf("Expected 3 neurons, got %v", len(l.Neurons))
	}

	// check for 4 inputs per neuron
	for _, n := range l.Neurons {
		if len(n.Weights) != 4 {
			t.Errorf("Expected 4 weights, got %v", len(n.Weights))
		}
	}

}
