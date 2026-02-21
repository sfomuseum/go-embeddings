//go:build mobileclip

package embeddings

import (
	"context"
	"io"
	"os"
	"testing"
)

func TestMobileCLIPEmbeddings(t *testing.T) {

	ctx := context.Background()

	emb, err := NewEmbedder(ctx, "mobileclip://?client-uri=grpc://localhost:8080&model=s0")

	if err != nil {
		t.Fatalf("Failed to create embedder, %v", err)
	}

	rsp, err := emb.Embeddings32(ctx, "Hello world")

	if err != nil {
		t.Fatalf("Failed to derive embeddings, %v", err)
	}

	if len(rsp) == 0 {
		t.Fatalf("Empty embedding")
	}
}

func TestMobileCLIPImageEmbeddings(t *testing.T) {

	ctx := context.Background()

	emb, err := NewEmbedder(ctx, "mobileclip://?client-uri=grpc://localhost:8080&model=s0")

	if err != nil {
		t.Fatalf("Failed to create embedder, %v", err)
	}

	im_path := "fixtures/1527845303_walrus.jpg"

	im_r, err := os.Open(im_path)

	if err != nil {
		t.Fatalf("Failed to open %s for reading, %v", im_path, err)
	}

	defer im_r.Close()

	im_body, err := io.ReadAll(im_r)

	if err != nil {
		t.Fatalf("Failed to read data from %s, %v", im_path, err)
	}

	rsp, err := emb.ImageEmbeddings32(ctx, im_body)

	if err != nil {
		t.Fatalf("Failed to derive embeddings, %v", err)
	}

	if len(rsp) == 0 {
		t.Fatalf("Empty embedding")
	}
}
