package embeddings

import (
	"fmt"
)

// CosineDistance calculates the semantic distance between two normalized vectors.
// 0.0 means the phrases are identical (perfectly similar).
// 1.0 means the phrases are orthogonal (unrelated).
// 2.0 means the phrases are perfect opposites.
func CosineDistance(vecA, vecB []float32) (float32, error) {
	if len(vecA) != len(vecB) {
		return 0, fmt.Errorf("vector dimensions must match")
	}

	// 1. Calculate the Dot Product (Similarity)
	var similarity float32
	for i := 0; i < len(vecA); i++ {
		similarity += vecA[i] * vecB[i]
	}

	// 2. Convert Similarity to Distance
	// Since the vectors are normalized to 1.0:
	// Distance = 1.0 - Similarity
	distance := 1.0 - similarity

	return distance, nil
}
