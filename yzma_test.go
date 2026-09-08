//go:build yzma

package embeddings

import (
	"context"
	"log/slog"
	"testing"
)

func TestYzmaEmbedder(t *testing.T) {

	slog.SetLogLoggerLevel(slog.LevelDebug)
	slog.Debug("Verbose logging enabled")

	ctx := context.Background()

	emb, err := NewEmbedder32(ctx, "yzma://")

	if err != nil {
		t.Fatalf("Failed to create embedder, %v", err)
	}

	req := &EmbeddingsRequest{
		Model: "https://huggingface.co/QuantFactory/SmolLM2-135M-GGUF/resolve/main/SmolLM2-135M.Q4_K_M.gguf",
		Body:  []byte("Hello world"),
	}

	rsp, err := emb.TextEmbeddings(ctx, req)

	if err != nil {
		t.Fatalf("Failed to derive embeddings, %v", err)
	}

	if len(rsp.Embeddings()) == 0 {
		t.Fatalf("Empty embedding")
	}
}
