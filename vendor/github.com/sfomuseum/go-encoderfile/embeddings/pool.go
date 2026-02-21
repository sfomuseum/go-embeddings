package embeddings

import (
	"fmt"
	"math"
)

type PooledResults struct {
	Embeddings []float32 `json:"embeddings"`
	Tokens     []string  `json:"tokens"`
	ModelId    string    `json:"model_id"`
}

func PoolOutputResults(output *EmbeddingsOutput) (*PooledResults, error) {

	if output == nil || len(output.Results) == 0 {
		return nil, fmt.Errorf("Missing results")
	}

	results := output.Results
	first := results[0]

	if first == nil || len(first.Embeddings) == 0 {
		return nil, fmt.Errorf("Missing embeddings")
	}

	pooled := PoolEmbeddings(first.Embeddings)
	tokens := make([]string, 0)

	for _, r := range results {
		for _, e := range r.Embeddings {
			tokens = append(tokens, e.TokenInfo.Token)
		}
	}

	rsp := &PooledResults{
		Embeddings: pooled,
		Tokens:     tokens,
		ModelId:    output.ModelId,
	}

	return rsp, nil
}

func PoolEmbeddings(vecs []*Embedding) []float32 {

	hidden_sz := len(vecs[0].Values)
	pooled := make([]float32, hidden_sz)

	// 1. Mean Pooling: Sum all token vectors

	for _, entry := range vecs {

		for i := 0; i < hidden_sz; i++ {
			pooled[i] += entry.Values[i]
		}
	}

	// 2. Average the sum

	count := float32(len(vecs))

	for i := 0; i < hidden_sz; i++ {
		pooled[i] /= count
	}

	// 3. L2 Normalization

	var sq float64

	for _, v := range pooled {
		sq += float64(v * v)
	}

	norm := float32(math.Sqrt(sq))

	if norm > 0 {
		for i := 0; i < hidden_sz; i++ {
			pooled[i] /= norm
		}
	}

	return pooled
}
