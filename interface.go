package main

import (
	g "gorgonia.org/gorgonia"
)

type TapeMachine interface {
	// Reset resets the run state of the machine by changing the instruction pointer back to 0
	// and reseting the registry
	Reset()
	Close() error
	// Let wraps the Let() function of the package, with additional checks that n is in the machine
	Let(n *g.Node, be interface{}) (err error)
	// Set wraps the Set() function of this package, with additional checks that both a and b are in the machine
	Set(a, b *g.Node) (err error)
	RunAll() (err error)
}
