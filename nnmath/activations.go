package nnmath

import "math"

// Activation function type consts
const (
	// SIGMOID is the sigmoid activation function
	SIGMOID = "sigmoid"
	// RELU is the rectified linear unit activation function
	RELU = "relu"
	// LINEAR is the linear activation function
	LINEAR = "linear"
)

// Sigmoid performs the calculation of 1 / (1 + e^-x)
func Sigmoid(x float64) float64 {
	return float64(1.0 / (1.0 + math.Exp(-x)))
}

// SigmoidPrime returns the derivative of the sigmoid function at x
func SigmoidPrime(x float64) float64 {
	s := Sigmoid(x)
	return s * (1 - s)
}

// Relu uses the Rectified Linear Unit function f(x) = max(0,x) which essentially just drops
// negative values
func Relu(x float64) float64 {
	if x < 0 {
		return 0
	}
	return x
}

// ReluPrime returns the derivative of the Relu function at x
func ReluPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

// LeakyRelu is a "leaky" version of the ReLU function to avoid the dead weight problem. This
// version returns a very small amount (0.01*x) of the value if x is less the 0.
func LeakyRelu(x float64) float64 {
	if x < 0 {
		// leak negative values
		return 0.01 * x
	}
	return x
}

// LeakyReluPrime returns the derivative of the LeakyRelu function at x
func LeakyReluPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0.1
}

// Linear activation function is simply the identity function f(x) = x
func Linear(x float64) float64 {
	return x
}

// LinearPrime returns the derivative of th Linear function - it's always 1, included for completeness
func LinearPrime(x float64) float64 {
	return 1
}
