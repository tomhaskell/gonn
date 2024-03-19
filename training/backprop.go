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
// costFunc is the cost function used to calculate the error: func (outputs, expected [][]float64) float64
func (b *BackProp) Train(net *gonn.Net, inputs, targets *[][]float64, epochs int, costFunc func([]float64, []float64) float64) {
	in := *inputs
	targ := *targets
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
			for j := 0; j < b.batchSize && i < len(in); j++ {
				out := net.Process(in[i])
				// calculate the error for the batch
				batchCost += costFunc(out, targ[i])
				i++
			}

			// calculate the deltas for the output layer - ?????????????????????
			deltas := make([][]float64, len(outputs))
			for j, out := range outputs {
				deltas[j] = make([]float64, len(out))
				for k, o := range out {
					a := nnmath.SigmoidPrime(o)
					deltas[j][k] = expected[j][k] - a
				}
			}
			
		}

	}
}
