package nn

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Config for Network
type Config struct {
	Epochs    int
	BatchSize int
	Eta       float64
}

// MLP Holds Architectural Values of Network
type MLP struct {
	numLayers int
	sizes     []int
	biases    []*mat.Dense
	weights   []*mat.Dense
	config    Config
}

// New Network
func New(c Config, sizes ...int) *MLP {
	// Generate Random Weights/Biases
	bs := []*mat.Dense{}
	ws := []*mat.Dense{}

	l := len(sizes) - 1

	for j := 0; j < l; j++ {
		y := sizes[1:][j]
		x := sizes[:l][j]

		// Make Random Init Biases Matrix
		// of y * 1
		b := make([]float64, y)
		for i := range b {
			b[i] = rand.NormFloat64()
		}

		bs = append(bs, mat.NewDense(y, 1, b))

		// Make Random Init Weights Matrix
		// of y * x
		w := make([]float64, y*x)
		for i := range w {
			w[i] = rand.NormFloat64()
		}

		ws = append(ws, mat.NewDense(x, y, w))
	}

	return &MLP{
		numLayers: len(sizes),
		sizes:     sizes,
		biases:    bs,
		weights:   ws,
		config:    c,
	}
}
