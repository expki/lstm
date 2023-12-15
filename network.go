package main

import (
	"fmt"
	"log"

	g "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var err error
var dt tensor.Dtype = tensor.Float64

type Network struct {
	graph *g.ExprGraph

	X, Y *g.Node

	w0, w1, w2, w3 *g.Node
	b0, b1, b2, b3 *g.Node

	memory *g.Node

	pred    *g.Node
	predVal g.Value

	VM     TapeMachine
	Solver *g.VanillaSolver
}

func New() *Network {
	// Create graph and network
	graph := g.NewGraph()

	// Create node for w/weight
	weight0 := g.NewMatrix(graph, dt, g.WithShape(3, 4), g.WithName("weight0"), g.WithInit(g.GlorotN(1.0)))
	bias0 := g.NewVector(graph, dt, g.WithShape(4), g.WithName("bias0"), g.WithInit(g.GlorotN(1.0)))
	weight1 := g.NewMatrix(graph, dt, g.WithShape(8, 8), g.WithName("weight1"), g.WithInit(g.GlorotN(1.0)))
	bias1 := g.NewVector(graph, dt, g.WithShape(8), g.WithName("bias1"), g.WithInit(g.GlorotN(1.0)))
	weight2 := g.NewMatrix(graph, dt, g.WithShape(8, 8), g.WithName("weight2"), g.WithInit(g.GlorotN(1.0)))
	bias2 := g.NewVector(graph, dt, g.WithShape(8), g.WithName("bias2"), g.WithInit(g.GlorotN(1.0)))
	weight := g.NewMatrix(graph, dt, g.WithShape(4, 1), g.WithName("weight3"), g.WithInit(g.GlorotN(1.0)))
	bias3 := g.NewVector(graph, dt, g.WithShape(1), g.WithName("bias3"), g.WithInit(g.GlorotN(1.0)))
	memory := g.NewVector(graph, dt, g.WithShape(4), g.WithName("memory"), g.WithInit(g.GlorotN(1.0)))

	// Create input and output nodes
	X := g.NewVector(graph, tensor.Float64, g.WithName("X"), g.WithShape(3))
	Y := g.NewVector(graph, tensor.Float64, g.WithName("Y"), g.WithShape(1))

	// Create network
	network := &Network{
		X:      X,
		Y:      Y,
		graph:  graph,
		w0:     weight0,
		b0:     bias0,
		w1:     weight1,
		b1:     bias1,
		w2:     weight2,
		b2:     bias2,
		w3:     weight,
		b3:     bias3,
		memory: memory,
	}

	// Run forward pass
	if err := network.fwd(X); err != nil {
		log.Fatalf("%+v", err)
	}

	// Calculate Cost w/MSE
	losses := g.Must(g.Sub(Y, network.pred))
	square := g.Must(g.Square(losses))
	cost := g.Must(g.Mean(square))

	// Do Gradient updates
	if _, err = g.Grad(cost, network.learnables()...); err != nil {
		log.Fatal(err)
	}

	// Instantiate VM and Solver
	vm := g.NewTapeMachine(graph, g.BindDualValues(network.learnables()...))
	solver := g.NewVanillaSolver(g.WithLearnRate(0.1))
	network.VM = vm
	network.Solver = solver

	return network
}

func (m *Network) learnables() g.Nodes {
	return g.Nodes{m.w0, m.b1, m.w1, m.b1, m.w2, m.b2, m.w3, m.b3}
}

func (m *Network) fwd(layerInput *g.Node) (err error) {
	var h1, hm1, h2, h3, hl3, layerOutput *g.Node

	// Input → Hidden 1
	h1 = g.Must(g.Mul(layerInput, m.w0))
	h1 = g.Must(g.Add(h1, m.b0))
	fmt.Printf("(%v × %v) + %v = %v\n", layerInput.Shape(), m.w0.Shape(), m.b0.Shape(), h1.Shape())

	// Hidden 1 + Memory → Hidden 2
	hm1 = g.Must(g.Concat(0, h1, m.memory))
	h2 = g.Must(g.Mul(hm1, m.w1))
	h2 = g.Must(g.Add(h2, m.b1))
	fmt.Printf("%v,%v = %v\n", h1.Shape(), m.memory.Shape(), hm1.Shape())
	fmt.Printf("(%v × %v) + %v = %v\n", hm1.Shape(), m.w1.Shape(), m.b1.Shape(), h2.Shape())

	// Hidden 2 → Hidden 3
	h3 = g.Must(g.Mul(h2, m.w2))
	h3 = g.Must(g.Add(h3, m.b2))

	// Hidden 3 → Memory
	m.memory = g.Must(g.Slice(h3, g.S(0, 4)))

	// Hidden 3 → Output
	hl3 = g.Must(g.Slice(h3, g.S(4, 8)))
	layerOutput = g.Must(g.Mul(hl3, m.w3))
	layerOutput = g.Must(g.Add(layerOutput, m.b3))
	layerOutput = g.Must(g.Tanh(layerOutput))
	fmt.Printf("(%v × %v) + %v = %v\n", hl3.Shape(), m.w3.Shape(), m.b3.Shape(), layerOutput.Shape())

	m.pred = layerOutput
	g.Read(m.pred, &m.predVal)
	return nil

}
