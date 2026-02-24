package embeddings

type Float interface{ ~float32 | ~float64 }

type EmbeddingsPrecision uint8

const (
	_         EmbeddingsPrecision = iota
	Micro                         // 4
	Mini                          // 8
	Half                          // 16
	Single                        // 32
	Double                        // 64
	Quadruple                     // 128
)

func AsFloat32(data []float64) []float32 {

	e32 := make([]float32, len(data))

	for idx, v := range data {
		// Really, check for max float32here...
		e32[idx] = float32(v)
	}

	return e32
}

func AsFloat64(data []float32) []float64 {

	e64 := make([]float64, len(data))

	for idx, v := range data {
		e64[idx] = float64(v)
	}

	return e64
}
