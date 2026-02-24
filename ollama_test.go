//go:build ollama

package embeddings

import (
	"context"
	"testing"
)

func TestOllamaEmbeddings(t *testing.T) {

	ctx := context.Background()

	emb, err := NewEmbedder(ctx, "ollama://?model=embeddinggemma")

	if err != nil {
		t.Fatalf("Failed to create embedder, %v", err)
	}

	req := &EmbeddingsRequest{
		Body: []byte("Hello world"),
	}

	rsp, err := emb.Embeddings32(ctx, req)

	if err != nil {
		t.Fatalf("Failed to derive embeddings, %v", err)
	}

	if len(rsp.Embeddings) == 0 {
		t.Fatalf("Empty embedding")
	}
}

func TestOllamaImageEmbeddings(t *testing.T) {
	t.Skip()
}
