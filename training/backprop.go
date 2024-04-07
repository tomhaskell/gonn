package training

import (
	"fmt"
	"math/rand"

	"github.com/tomhaskell/gonn"
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
	in := *inputs
	targ := *targets
	// consts
	L := len(net.Layers)
	N := make([]int, L)
	for i, l := range net.Layers {
		N[i] = len(l.Neurons)
	}

	for e := 0; e < epochs; e++ {
		// randomise training set
		rand.Shuffle(len(in), func(i, j int) {
			in[i], in[j] = in[j], in[i]
			targ[i], targ[j] = targ[j], targ[i]
		})

		i := 0
		for i < len(in) {
			// process the batch
			batchCost := 0.0
			batchDeltas := make([][]float64, L) // holds delta values for each neuron for the batch
			batchDeltaActivations := make([][]float64, L) // holds delta * activation values for each neuron for the batch
			for d := 0; d < L; d++ {
				batchDeltas[d] = make([]float64, N[d])
				batchDeltaActivations[d] = make([]float64, N[d])
			}
			for j := 0; j < b.batchSize && i < len(in); j++ {
				zs, acts := net.Calculate(in[i])
				// calculate the cost for the batch
				// TODO: implement other cost functions
				batchCost +=  nnmath.MeanSquaredError(acts[L-1], targ[i])
				// calulate the error in the output
				deltas := make([][]float64, L)
				for d := 0; d < L; d++ {
					deltas[d] = make([]float64, N[d])
				}
				for n := 0; n < N[L-1]; n++ {
					d := nnmath.SigmoidPrime(zs[L-1][n]) * (acts[L-1][n] - targ[i][n])
					deltas[L-1][n] = d
					batchDeltas[L-1][n] += d
					batchDeltaActivations[L-1][n] += d * acts[L-1][n]
				}
				
				// calulate the error in the hidden layer(s) by backpropagating the error
				for l := L-2; l >= 0; l-- {
					for n := 0; n < N[l]; n++ {
						d := 0.0
						for m := 0; m < N[l+1]; m++ {
							d += deltas[l+1][m] * net.Layers[l+1].Neurons[m].Weights[n]
						}
						d *= nnmath.SigmoidPrime(zs[l][n])
						deltas[l][n] = d
						batchDeltas[l][n] += d
						batchDeltaActivations[l][n] += d * acts[l][n]
					}
				}
			
				i++
			}

			// update the weights and biases using gradient descent based on the batch deltas
			alpha := b.learningRate / float64(b.batchSize)
			for l := 0; l < L; l++ {
				for n := 0; n < N[l]; n++ {
					neuron := net.Layers[l].Neurons[n]
					for i := 0; i < len(neuron.Weights); i++ {
						neuron.Weights[i] -= alpha * batchDeltaActivations[l][n]
					}
					neuron.Bias -= alpha * batchDeltas[l][n]
				}
			}
			
			fmt.Printf("Epoch %d, Batch %d, Cost: %f\n", e, i, batchCost)
		}

	}
}
