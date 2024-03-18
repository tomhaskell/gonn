package nnmath

import "math"

// Sigmoid performs the calculation of 1 / (1 + e^-x)
func Sigmoid(x float64) float64 {
	return float64(1.0 / (1.0 + math.Exp(float64(-x))))
}

// MeanSquaredError returns the average of the squared differences between the actual and expected values
func MeanSquaredError(actual, expected []float64) float64 {
	var sum float64
	for i := range actual {
		sum += math.Pow(float64(expected[i] - actual[i]), 2)
	}
	return sum / float64(len(actual))
}