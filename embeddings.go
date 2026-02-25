package embeddings

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

func toFloat64Slice[T Float](src []float64) []T {
	dst := make([]T, len(src))
	for i, v := range src {
		dst[i] = T(v) // float64 → T (float32 or float64)
	}
	return dst
}

func toFloat32Slice[T Float](src []float32) []T {
	dst := make([]T, len(src))
	for i, v := range src {
		dst[i] = T(v) // float32 → T (float32 or float64)
	}
	return dst
}
