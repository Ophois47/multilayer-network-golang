package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strconv"
)

func main() {
	xCols := 4

	// Create "data.csv"
	resp, err := http.Get("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data")

	if err != nil {
		panic("Error Downloading Data!")
	}

	lines := []string{}
	r := csv.NewReader(resp.Body)
	for {
		fields, err := r.Read()
		if err == io.EOF {
			break
		}

		if err != nil {
			panic("Error Reading From CSV!" + err.Error())
		}

		line := ""
		for i, f := range fields {
			if i < xCols {
				v := parseX(f)
				m, sd := colStats(i)
				v = normalize(v, m, sd)

				line += fmt.Sprintf("%0.5f,", v)
			} else {
				v := parseY(f)
				line += fmt.Sprintf("%s\n", v)
				lines = append(lines, line)
			}
		}
	}

	train, err := os.Create("train.csv")
	if err != nil {
		panic("Error Creating Out Of File!")
	}

	defer train.Close()

	test, err := os.Create("test.csv")
	if err != nil {
		panic("Error Creating Out Of File!")
	}

	defer train.Close()

	counts := make(map[int]int, 0)
	for _, i := range rand.Perm(len(lines)) {
		var c int
		if i < 50 {
			c = 0
		} else if i > 100 {
			c = 2
		} else {
			c = 1
		}

		if counts[c] < 33 {
			train.WriteString(lines[i])
		} else {
			test.WriteString(lines[i])
		}

		counts[c]++
	}

	train.Sync()
	test.Sync()
}

var means = []float64{5.84, 3.05, 3.76, 1.2}
var sds = []float64{0.83, 0.43, 1.76, 0.76}

func colStats(i int) (mean float64, sd float64) {
	return means[i], sds[i]
}

func parseX(s string) float64 {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		panic("Error Parsing Float!")
	}

	return f
}
