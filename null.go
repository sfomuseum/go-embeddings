package embeddings

import (
	"context"
	"time"
)

// NullEmbedder implements the `Embedder` interface using an Null API endpoint to derive embeddings.
type NullEmbedder[T Float] struct {
	Embedder[T]
}

func init() {
	ctx := context.Background()

	err := RegisterEmbedder(ctx, "null", NewNullEmbedder[float64])

	if err != nil {
		panic(err)
	}
}

func NewNullEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	e := &NullEmbedder[T]{}
	return e, nil
}

func (e *NullEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {
	return e.nullEmbeddings(ctx, req)
}

func (e *NullEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {
	return e.nullEmbeddings(ctx, req)
}

func (e *NullEmbedder[T]) nullEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:         req.Id,
		CommonEmbeddings: make([]T, 0),
		CommonModel:      "null",
		CommonCreated:    ts,
		CommonPrecision:  64,
	}

	return rsp, nil
}
