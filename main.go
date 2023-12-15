package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Sample network learning basic addition. (terrible use case for LSTM since next results don't rely on previous results)
func main() {
	network := New()
	const trainCount int = 1000

	// Generate training data
	X := make([][]float64, trainCount)
	Y := make([][]float64, trainCount)
	for idx := 0; idx < trainCount; idx++ {
		a := rand.Float64()
		b := rand.Float64()
		c := rand.Float64()
		X[idx] = []float64{a, b, c}
		Y[idx] = []float64{math.Min(math.Max(a+b+c, -1), 1)}
	}

	// Train the network
	for idx := 0; idx < trainCount; idx++ {
		// Set input x to network
		xB := X[idx]
		xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(3))
		if err := g.Let(network.X, xT); err != nil {
			panic(err)
		}

		// Define validation data set
		yB := Y[idx]
		yT := tensor.New(tensor.WithBacking(yB), tensor.WithShape(1))
		if err := g.Let(network.Y, yT); err != nil {
			panic(err)
		}

		network.VM.Reset()
		if err = network.VM.RunAll(); err != nil {
			log.Fatalf("Failed at inter  %d: %v", idx, err)
		}
		network.Solver.Step(g.NodesToValueGrads(network.learnables()))
	}
	network.VM.Reset()

	// Test the network
	xB := []float64{0.5, -0.6, 0.2}
	xT := tensor.New(tensor.WithBacking(xB), tensor.WithShape(3))
	if err := g.Let(network.X, xT); err != nil {
		panic(err)
	}
	network.VM.Reset()
	if err = network.VM.RunAll(); err != nil {
		log.Fatalf("Failed at inter test: %v", err)
	}

	fmt.Println("Correct output:", xB[0]+xB[1]+xB[2])
	fmt.Println("Network output:", network.predVal)
}
