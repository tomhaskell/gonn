package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/tomhaskell/gonn/slownn"
	"github.com/tomhaskell/gonn/slownn/training"
)

var (
	layers    = flag.String("layers", "24 16", "list specifying number of neurons in each hidden layer")
	epochs    = flag.Int("epochs", 30, "number of training epochs to train the network for")
	learnRate = flag.Float64("learnRate", 0.7, "the learning rate to use in the backprop algorithm")
	batchSize = flag.Int("batchSize", 20, "the size of mini-batch to use for stochastic gradient descent")
	act       = flag.String("act", "sigmoid", "the activation function to use (\"sigmoid\",\"relu\",\"linear\")")
	momentum  = flag.Float64("momentum", 0.0, "the momentum to use for the gradient descent algorithm")
)

func main() {
	flag.Parse()

	// load the data
	trainInputs, trainTargets := parseDataFile(".data/mnist_train.csv")
	testInputs, testTargets := parseDataFile(".data/mnist_test.csv")

	// create a new neural network with 784 input neurons
	nb := slownn.NewBuilder().SetDefaultActivation(*act).SetInputCount(784)

	for _, s := range strings.Split(*layers, " ") {
		l, err := strconv.ParseInt(s, 10, 32)
		if err != nil {
			panic(fmt.Errorf("error parsing layers: %w", err))
		}
		nb = nb.AddLayer(int(l))
	}

	// add output neurons - always use Sigmoid function (until softmax is available)
	nb = nb.AddLayerWithActivation(10, slownn.SIGMOID)

	net := nb.Build()

	fmt.Println("training net: ", net)

	// train the network
	t := training.NewBackProp(*learnRate, *momentum, *batchSize)

	for e := 0; e < *epochs; e++ { // 30 epochs
		t.TrainEpoch(net, &trainInputs, &trainTargets)

		// test network for accuracy after each epoch
		if testInputs != nil && testTargets != nil {
			correct := 0
			for i, in := range testInputs {
				output := net.FeedForward(in)
				if maxIndex(output) == maxIndex((testTargets)[i]) {
					correct++
				}
			}
			fmt.Printf("Epoch %d: %d/%d correct (%d%%)\n", e, correct, len(testInputs), 100*correct/len(testInputs))
		}

	}

}

// returns the index of the slice f with the highest value
func maxIndex(f []float64) int {
	index := 0
	max := f[0]
	for i, val := range f {
		if val > max {
			index = i
			max = val
		}
	}
	return index
}

// parseDataFile reads one of the data csv files and returns the inputs and targets
func parseDataFile(filename string) ([][]float64, [][]float64) {
	f, err := os.Open(filename)
	if err != nil {
		panic(fmt.Errorf("error opening file: %v", err))
	}
	defer f.Close()

	csvReader := csv.NewReader(f)
	r, err := csvReader.ReadAll()
	if err != nil {
		panic(fmt.Errorf("error reading csv: %v", err))
	}

	inputs := make([][]float64, len(r))
	targets := make([][]float64, len(r))

	for i, line := range r {
		inputs[i] = make([]float64, len(line)-1)
		targets[i] = make([]float64, 10)
		for j, field := range line {
			if j == 0 {
				// the first column is the target
				val, err := strconv.Atoi(strings.TrimSpace(field))
				if err != nil {
					panic(fmt.Errorf("error parsing int: %v", err))
				}
				targets[i][val] = 1.0
			} else {
				// the rest are inputs
				val, err := strconv.ParseFloat(strings.TrimSpace(field), 32)
				if err != nil {
					panic(fmt.Errorf("error parsing float: %v", err))
				}
				inputs[i][j-1] = float64(val / 255.0) // normalise the input
			}
		}
	}

	return inputs, targets
}
