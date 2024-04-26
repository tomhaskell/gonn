package nnmath

import "fmt"

type Activator struct {
	Activate   func(x float64) float64
	Derivative func(x float64) float64
}

func NewActivator(type_ string) (*Activator, error) {
	switch type_ {
	case SIGMOID:
		return &Activator{
			Activate:   Sigmoid,
			Derivative: SigmoidPrime,
		}, nil
	case RELU:
		return &Activator{
			Activate:   LeakyRelu,
			Derivative: LeakyReluPrime,
		}, nil
	case LINEAR:
		return &Activator{
			Activate:   Linear,
			Derivative: LinearPrime,
		}, nil
	}
	return nil, fmt.Errorf("Unkown activator type: %s", type_)
}
