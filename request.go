package embeddings

type EmbeddingsRequest struct {
	Id    string `json:"id,omitempty"`
	Model string `json:"model"`
	Body  []byte `json:"body"`
}
