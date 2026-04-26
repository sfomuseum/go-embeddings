package embeddings

type MLXClipEmbeddingsResponse struct {
	Embeddings []float64 `json:"embeddings"`
	Model      string    `json:"model"`
}
