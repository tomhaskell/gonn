package nnmath

import "math"

// MeanSquaredError returns the average of the squared differences between the actual and expected values
func MeanSquaredError(actual, expected []float64) float64 {
	var sum float64
	for i := range actual {
		sum += math.Pow(float64(expected[i]-actual[i]), 2)
	}
	return sum / float64(len(actual)*2)
}
