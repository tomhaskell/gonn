package nn

import (
	"math"
	"testing"
)

func TestNeuronOutputLinear(t *testing.T) {
	n := &Neuron{
		Type:    LINEAR,
		Weights: []float64{0.3, -0.8},
		Bias:    0.7,
	}

	n.Update([]float64{0.1, 0.2})
	out := n.Activation()

	expected := 0.1*0.3 + 0.2*-0.8 + 0.7 // 0.57

	if out != expected {
		t.Errorf("Expected %v, got %v", expected, out)
	}

}

func TestNeuronOutputSigmoid(t *testing.T) {
	n := &Neuron{
		Type:    SIGMOID,
		Weights: []float64{0.3, -0.8},
		Bias:    0.7,
	}

	n.Update([]float64{0.1, 0.2})
	out := n.Activation()

	expected := 1.0 / (1.0 + math.Exp(-0.57)) // ~0.638763

	if out != expected {
		t.Errorf("Expected %v, got %v", expected, out)
	}
}

func TestNeuronOutputRelu(t *testing.T) {
	n := &Neuron{
		Type:    RELU,
		Weights: []float64{0.3, -0.8},
		Bias:    0.7,
	}

	n.Update([]float64{0.1, 0.2})
	out := n.Activation()

	expected := 0.1*0.3 + 0.2*-0.8 + 0.7 // 0.57

	if out != expected {
		t.Errorf("Expected %v, got %v", expected, out)
	}

	n.Update([]float64{-1, 2})
	out = n.Activation()
	expected = 0.0

	if out != expected {
		t.Errorf("Expected %v, got %v", expected, out)
	}

}

func TestNewNeuron(t *testing.T) {
	n := NewNeuron(LINEAR, 3)
	n2 := NewNeuron(LINEAR, 3)

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
