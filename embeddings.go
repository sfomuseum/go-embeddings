package embeddings

func EmbeddingsResponseToEmbeddingsResponse32(rsp64 *EmbeddingResponse) *EmbeddingResponse32 {

	e32 := AsFloat32(rsp64.Embeddings)

	rsp32 := &EmbeddingsResponse{
		Id:         rsp64.Id,
		Embeddings: e32,
		Dimensions: rsp64.Dimensions,
		Created:    rsp64.Created,
		Model:      rsp64.Model,
	}

	return rsp32, nil
}

func EmbeddingsResponse32ToEmbeddingsResponse(rsp32 *EmbeddingResponse32) *EmbeddingResponse {

	e64 := AsFloat64(rsp32.Embeddings)

	rsp64 := &EmbeddingsResponse{
		Id:         rsp32.Id,
		Embeddings: e64,
		Dimensions: rsp32.Dimensions,
		Created:    rsp32.Created,
		Model:      rsp32.Model,
	}

	return rsp32, nil
}

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
