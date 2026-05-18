package embeddings

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	_ "time"
)

// RouteEmbedder implements the `Embedder` interface using an Null API endpoint to derive embeddings.
type RouteEmbedder[T Float] struct {
	Embedder[T]
	precision string
	scheme    string
	clients   map[string]Embedder[T]
}

func init() {
	ctx := context.Background()

	RegisterEmbedder[float32](ctx, "route", NewRouteEmbedder[float32])
	RegisterEmbedder[float32](ctx, "route32", NewRouteEmbedder[float32])
	RegisterEmbedder[float64](ctx, "route64", NewRouteEmbedder[float64])
}

func NewRouteEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	precision := "float32"

	switch {
	case strings.HasSuffix(u.Scheme, "64"):
		precision = "%s#as-float64"
	}

	clients := make(map[string]Embedder[T])

	e := &RouteEmbedder[T]{
		precision: precision,
		scheme:    u.Scheme,
		clients:   clients,
	}

	return e, nil
}

func (e *RouteEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	client, exists := e.clients[req.Model]

	if !exists {
		return nil, fmt.Errorf("Model not found")
	}

	return client.TextEmbeddings(ctx, req)
}

func (e *RouteEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	client, exists := e.clients[req.Model]

	if !exists {
		return nil, fmt.Errorf("Model not found")
	}

	return client.ImageEmbeddings(ctx, req)
}
