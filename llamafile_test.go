//go:build llamafile

package embeddings

import (
	"context"
	"io"
	"os"
	"testing"
)

func TestLlamafileEmbeddings(t *testing.T) {

	ctx := context.Background()

	emb, err := NewEmbedder(ctx, "llamafile://")

	if err != nil {
		t.Fatalf("Failed to create embedder, %v", err)
	}

	res := &EmbeddingsRequest{
		Body: []byte("Hello world"),
	}

	rsp, err := emb.Embeddings(ctx, req)

	if err != nil {
		t.Fatalf("Failed to derive embeddings, %v", err)
	}

	if len(rsp.Embeddings) == 0 {
		t.Fatalf("Empty embedding")
	}
}

func TestLlamafileImageEmbeddings(t *testing.T) {

	ctx := context.Background()

	emb, err := NewEmbedder(ctx, "llamafile://")

	if err != nil {
		t.Fatalf("Failed to create embedder, %v", err)
	}

	im_path := "../fixtures/1527845303_walrus.jpg"

	im_r, err := os.Open(im_path)

	if err != nil {
		t.Fatalf("Failed to open %s for reading, %v", im_path, err)
	}

	defer im_r.Close()

	im_body, err := io.ReadAll(im_r)

	if err != nil {
		t.Fatalf("Failed to read data from %s, %v", im_path, err)
	}

	res := &EmbeddingsRequest{
		Body: im_body,
	}

	rsp, err := emb.ImageEmbeddings(ctx, im_body)

	if err != nil {
		t.Fatalf("Failed to derive embeddings, %v", err)
	}

	if len(rsp.Embeddings) == 0 {
		t.Fatalf("Empty embedding")
	}
}
