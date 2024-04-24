package training

import (
	"fmt"
	"math/rand"

	"github.com/tomhaskell/gonn"
	"github.com/tomhaskell/gonn/nn"
	"github.com/tomhaskell/gonn/nnmath"
)

type BackProp struct {
	// learningRate is the rate at which the network learns
	learningRate float64
	// momentum is the amount of momentum to apply to the learning rate
	momentum float64
	// batchSize is the number of inputs to process before updating the weights. A batchSize of 1 is
	// equivalent to online learning
	batchSize int
}

// NewBackProp returns a new BackProp trainer with the given learning rate and momentum
func NewBackProp(learningRate, momentum float64, batchSize int) *BackProp {
	return &BackProp{
		learningRate: learningRate,
		momentum:     momentum,
		batchSize:    batchSize,
	}
}

// Train trains the network using the backpropagation algorithm
//
// epochs is the number of times to train the network on the entire dataset
func (b *BackProp) Train(net *gonn.Net, inputs, targets *[][]float64, epochs int) {
	for e := 0; e < epochs; e++ {
		b.TrainEpoch(net, inputs, targets)
	}
}

// TrainEpoch trains the network using the backpropagation algorithm for a single epoch. This is
// mainly for use during testing
func (b *BackProp) TrainEpoch(net *gonn.Net, inputs, targets *[][]float64) {
	in := *inputs
	targ := *targets
	// consts
	L := len(net.Layers) // number of layers
	N := make([]int, L)  // number of neurons per layer
	for i, l := range net.Layers {
		N[i] = len(l.Neurons)
	}

	// randomise training set
	rand.Shuffle(len(in), func(i, j int) {
		in[i], in[j] = in[j], in[i]
		targ[i], targ[j] = targ[j], targ[i]
	})

	// create slice to store the weight changes so we can use them with momentum for faster gradient descent
	weightChanges := make([][][]float64, L)
	for l := 0; l < L; l++ {
		weightChanges[l] = make([][]float64, N[l])
		for n := 0; n < N[l]; n++ {
			if l == 0 {
				weightChanges[l][n] = make([]float64, net.NumInputs)
			} else {
				weightChanges[l][n] = make([]float64, N[l-1])
			}
		}
	}

	i := 0
	// all training sets
	for i < len(in) {
		batchCost := 0.0
		batchDeltas := make([][]float64, L)             // holds delta values for each neuron for the batch
		batchDeltaActivations := make([][][]float64, L) // holds delta * input activation values for each neuron input weight for the batch
		for d := 0; d < L; d++ {
			batchDeltas[d] = make([]float64, N[d])
			batchDeltaActivations[d] = make([][]float64, N[d])
			for n := 0; n < N[d]; n++ {
				if d == 0 {
					batchDeltaActivations[d][n] = make([]float64, net.NumInputs)
				} else {
					batchDeltaActivations[d][n] = make([]float64, N[d-1])
				}
			}
		}
		// process a batch
		for j := 0; j < b.batchSize && i < len(in); j++ {
			zs, acts := net.Calculate(in[i])
			// calculate the cost for the batch
			// TODO: implement other cost functions
			batchCost += nnmath.MeanSquaredError(acts[L-1], targ[i])
			deltas := make([][]float64, L)
			for d := 0; d < L; d++ {
				deltas[d] = make([]float64, N[d])
			}

			// calulate the error in the output layer
			for n := 0; n < N[L-1]; n++ {
				p := activationDerivative(net.Layers[L-1].Type, zs[L-1][n])
				d := p * (acts[L-1][n] - targ[i][n])
				deltas[L-1][n] = d
				batchDeltas[L-1][n] += d
				if L == 1 { // single layer network
					for k := 0; k < len(in[i]); k++ {
						batchDeltaActivations[L-1][n][k] += d * in[i][k]
					}
				} else {
					for k := 0; k < N[L-2]; k++ {
						batchDeltaActivations[L-1][n][k] += d * acts[L-2][k]
					}
				}
			}

			// calulate the error in the hidden layer(s) by backpropagating the error
			for l := L - 2; l >= 0; l-- {
				for n := 0; n < N[l]; n++ {
					d := 0.0
					for m := 0; m < N[l+1]; m++ {
						d += deltas[l+1][m] * net.Layers[l+1].Neurons[m].Weights[n]
					}
					d *= activationDerivative(net.Layers[l].Type, zs[l][n])
					deltas[l][n] = d
					batchDeltas[l][n] += d
					if l == 0 { // input layer
						for k := 0; k < len(in[i]); k++ {
							batchDeltaActivations[l][n][k] += d * in[i][k]
						}
					} else {
						for k := 0; k < N[l-1]; k++ {
							batchDeltaActivations[l][n][k] += d * acts[l-1][k]
						}
					}
				}
			}

			i++
		}

		// update the weights and biases using gradient descent based on the batch deltas
		alpha := b.learningRate / float64(b.batchSize)
		m := b.momentum / float64(b.batchSize)
		// each layer
		for l := 0; l < L; l++ {
			// each Neuron
			for n := 0; n < N[l]; n++ {
				neuron := net.Layers[l].Neurons[n]
				// each input weight to Neuron
				for i := 0; i < len(neuron.Weights); i++ {
					weightChanges[l][n][i] = -alpha*batchDeltaActivations[l][n][i] + m*weightChanges[l][n][i]
					neuron.Weights[i] += weightChanges[l][n][i]
				}
				// bias of Neuron
				neuron.Bias -= alpha * batchDeltas[l][n]
			}
		}

	}

}

func activationDerivative(actType string, value float64) float64 {
	switch actType {
	case nn.SIGMOID:
		return nnmath.SigmoidPrime(value)
	case nn.RELU:
		if value > 0 {
			return 1
		} else {
			return 0
		}
	case nn.LINEAR:
		return 1
	}
	// unknown
	fmt.Println("Error: unknown activation function, cannot calc derivative")
	return 0
}
