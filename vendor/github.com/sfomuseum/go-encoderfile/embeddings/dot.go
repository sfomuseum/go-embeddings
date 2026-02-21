package embeddings

import (
	"fmt"
)

// DotProduct calculates the similarity between two normalized vectors.
// Returns a score where 1.0 is perfectly similar and -1.0 is perfectly opposite.
func DotProduct(vecA, vecB []float32) (float32, error) {
	if len(vecA) != len(vecB) {
		return 0, fmt.Errorf("vector dimensions must match")
	}

	var score float32
	for i := 0; i < len(vecA); i++ {
		score += vecA[i] * vecB[i]
	}

	return score, nil
}
