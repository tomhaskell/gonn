package main

import (
	"encoding/csv"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/tomhaskell/gonn"
	"github.com/tomhaskell/gonn/training"
)

func main() {

	// load the data
	trainInputs, trainTargets := parseDataFile(".data/mnist_train.csv")
	testInputs, testTargets := parseDataFile(".data/mnist_test.csv")
	
	// create a new neural network with 784 input neurons, 2 x 16 hidden neurons, and 10 output neurons (one for each digit)
	net := gonn.NewBuilder().SetInputCount(784).AddLayer(30).AddLayer(10).Build()

	// train the network
	var t training.Trainer = training.NewBackProp(0.1, 0, 10)
	
	for e := 0; e < 30; e++ { // 30 epochs
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
			fmt.Printf("Epoch %d: %d/%d correct\n", e, correct, len(testInputs))
		}

	}


	// test the network
	//test(net, testInputs, testTargets)

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
