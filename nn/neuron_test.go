package nn

import (
	"testing"
)

func TestNeuronProcess(t *testing.T) {
	n := &Neuron{
		Weights: []float32{0.3, -0.8},
		Bias:    0.7,
	}

	out := n.Process([]float32{0.1, 0.2})

	expected := float32(0.1*0.3 + 0.2*-0.8 + 0.7) // 0.57

	if out != expected {
		t.Errorf("Expected %v, got %v", expected, out)
	}

}

func TestNewNeuron(t *testing.T) {
	n := NewNeuron(3)
	n2 := NewNeuron(3)

	if len(n.Weights) != 3 {
		t.Errorf("Expected 3 weights, got %v", len(n.Weights))
	}

	for i := range n.Weights {
		if n.Weights[i] == n2.Weights[i] {
			t.Errorf("Expected random weights, got %v", n.Weights[i])
		}
		if n.Weights[i] < -1.0 || n.Weights[i] > 1.0 {
			t.Errorf("Expected weights between -1 and 1, got %v", n.Weights[i])
		}
	}

	if n.Bias != 0.0 {
		t.Errorf("Expected 0.0 bias, got %v", n.Bias)
	}
}