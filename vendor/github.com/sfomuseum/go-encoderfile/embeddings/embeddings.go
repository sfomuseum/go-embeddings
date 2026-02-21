package embeddings

type EmbeddingsInput struct {
	Inputs    []string       `json:"inputs"`
	Normalize bool           `json:"normalize"`
	Metadata  map[string]any `json:"metadata,omitempty"`
}

type EmbeddingsOutput struct {
	Results  []*Result      `json:"results"`
	ModelId  string         `json:"model_id"`
	Metadata map[string]any `json:"metadata"`
}

type Result struct {
	Embeddings []*Embedding `json:"embeddings"`
}

type Embedding struct {
	Values    []float32  `json:"embedding"`
	TokenInfo *TokenInfo `json:"token_info"`
}

type TokenInfo struct {
	Token   string `json:"token"`
	TokenId int    `json:"token_id"`
	Start   int    `json:"start"`
	End     int    `json:"end"`
}
