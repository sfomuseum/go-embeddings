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

func (e *NullEmbedder) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse, error) {
	return e.nullEmbeddings(ctx, req)
}

func (e *NullEmbedder) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse, error) {
	return e.nullEmbeddings(ctx, req)
}

func (e *NullEmbedder) nullEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse, error) {

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse{
		CommonId:           req.Id,
		CommonEmbeddings64: make([]float64, 0),
		CommonModel:        "null",
		CommonCreated:      ts,
		CommonPrecision:    64,
	}

	return rsp, nil
}
