package training

import "github.com/tomhaskell/gonn"

type Trainer interface {
	Train(net *gonn.Net, inputs, targets *[][]float64, epochs int)
}

