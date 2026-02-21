//go:build encoderfile

package embeddings

import (
	"context"
	"testing"
)

func TestEncoderfileEmbeddings(t *testing.T) {

	ctx := context.Background()

	emb, err := NewEmbedder(ctx, "encoderfile://?client-uri=http://localhost:8080")

	if err != nil {
		t.Fatalf("Failed to create embedder, %v", err)
	}

	rsp, err := emb.Embeddings(ctx, "Hello world")

	if err != nil {
		t.Fatalf("Failed to derive embeddings, %v", err)
	}

	if len(rsp) == 0 {
		t.Fatalf("Empty embedding")
	}
}

func TestEncoderfileImageEmbeddings(t *testing.T) {
	t.Skip()
}
