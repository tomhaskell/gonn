package nn

type Trainer interface {
	Train(inputs, targets []float32, learningRate float32)
}