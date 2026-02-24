package embeddings

import (
	"context"
	"time"
)

// NullEmbedder implements the `Embedder` interface using an Null API endpoint to derive embeddings.
type NullEmbedder struct {
	Embedder
}

func init() {
	ctx := context.Background()
	err := RegisterEmbedder(ctx, "null", NewNullEmbedder)

	if err != nil {
		panic(err)
	}
}

func NewNullEmbedder(ctx context.Context, uri string) (Embedder, error) {

	e := &NullEmbedder{}
	return e, nil
}

func (e *NullEmbedder) Embeddings(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse, error) {

	now := time.Now()
	ts := now.Unix()

	rsp := &EmbeddingsResponse{
		Embeddings: make([]float64, 0),
		Model:      "null",
		Created:    ts,
	}

	return rsp, nil
}

func (e *NullEmbedder) Embeddings32(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse32, error) {

	rsp64, err := e.Embeddings(ctx, req)

	if err != nil {
		return nil, err
	}

	return EmbeddingsResponseToEmbeddingsResponse32(rsp64), nil
}

func (e *NullEmbedder) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse, error) {

	now := time.Now()
	ts := now.Unix()

	rsp := &EmbeddingsResponse{
		Embeddings: make([]float64, 0),
		Model:      "null",
		Created:    ts,
	}

	return rsp, nil
}

func (e *NullEmbedder) ImageEmbeddings32(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse32, error) {

	rsp64, err := e.Embeddings(ctx, req)

	if err != nil {
		return nil, err
	}

	return EmbeddingsResponseToEmbeddingsResponse32(rsp64), nil
}
