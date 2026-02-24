package embeddings

import (
	"context"
	"net/url"
	"time"
)

// NullEmbedder implements the `Embedder` interface using an Null API endpoint to derive embeddings.
type NullEmbedder[T Float] struct {
	Embedder[T]
	precision string
	scheme    string
}

func init() {
	ctx := context.Background()

	RegisterEmbedder[float64](ctx, "null", NewNullEmbedder[float64])
	RegisterEmbedder[float64](ctx, "null64", NewNullEmbedder[float64])
	RegisterEmbedder[float32](ctx, "null32", NewNullEmbedder[float32])
}

func NewNullEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	precision := "64"

	switch u.Scheme {
	case "null32":
		precision = "64#32"
	}

	e := &NullEmbedder[T]{
		precision: precision,
		scheme:    u.Scheme,
	}

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
		CommonPrecision:  e.precision,
	}

	return rsp, nil
}
