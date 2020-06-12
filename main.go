package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// Type Used As Flag Variable
type ints []int

func (i *ints) String() string {
	return fmt.Sprintf("%d", *i)
}

func (i *ints) Set(s string) error {
	vals := strings.Split(s, ",")
	for _, v := range vals {
		tmp, err := strconv.Atoi(v)
		if err != nil {
			return err
		}

		*i = append(*i, tmp)
	}

	return nil
}

var (
	epochs    int
	eta       float64
	batchSize int
	arch      = ints{}
	train     string
	test      string
)

func main() {
	// Args
	flag.IntVar(&epochs, "Epochs", 1000, "Number of Training Epochs")
	flag.Float64Var(&eta, "ETA", 0.3, "Learning Rate")
	flag.IntVar(&batchSize, "Batch", 32, "Size of Training Batch")
	flag.Var(&arch, "Arch", "Architecture of Neurons")
	flag.StringVar(&train, "Train", "train.csv", "Path of Training CSV")
	flag.StringVar(&test, "Test", "test.csv", "Path of Testing CSV")
	flag.Parse()

	if len(arch) < 2 {
		panic("...Neural Network Must Have at Least 2 Layers Minimum for Input and Output...")
	}

	i := arch[0]
	o := arch[len(arch)-1]

	con := nn.Config{
		Epochs:    epochs,
		Eta:       eta,
		BatchSize: batchSize,
	}

	// Make New Network
	n := nn.New(con, arch...)

	// Train Network
	x, y := load(train, i, o)
	n.Train(x, y)

	// Evaluate
	x, y = load(test, i, o)
	accuracy := n.Evaluate(x, y)
	fmt.Printf("Accuracy = %0.1f%%\n", accuracy)
}

func load(path string, xFields, yFields int) (*mat.Dense, *mat.Dense) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatal(err)
	}

	defer f.Close()

	r := csv.NewReader(f)

	x := []float64{}
	y := []float64{}

	n := 0
	for {
		fields, err := r.Read()
		if err == io.EOF {
			break
		} else if err != nil {
			panic("Error Parsing CSV: " + err.Error())
		}

		for i, v := range fields {
			fl := string2float64(v)
			if i < xFields {
				x = append(x, fl)
			} else {
				y = append(y, fl)
			}
		}

		n++
	}

	return mat.NewDense(n, xFields, x), mat.NewDense(n, yFields, y)
}

func string2float64(v string) float64 {
	f, err := strconv.ParseFloat(v, 64)
	if err != nil {
		panic("...Error Parsing Field as Float...")
	}

	return f
}
