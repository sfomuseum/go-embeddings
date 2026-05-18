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

	q := u.Query()

	precision := "float32"

	switch {
	case strings.HasSuffix(u.Scheme, "64"):
		precision = "%s#as-float64"
	}

	clients := make(map[string]Embedder[T])

	client_uris := q["client-uri"]

	if len(client_uris) == 0 {
		return nil, fmt.Errorf("A minimum of (1) ?client-uri= parameters is required")
	}

	for _, str_spec := range client_uris {

		spec := strings.SplitN(str_spec, " ", 2)

		if len(spec) != 2 {
			return nil, fmt.Errorf("?client-uri= parameter must be in the form of '{MODEL} {CLIENT_URI}'")
		}

		model := spec[0]
		client_uri := spec[1]

		_, exists := clients[model]

		if exists {
			return nil, fmt.Errorf("Model %s already registered", model)
		}

		cl, err := NewEmbedder[T](ctx, client_uri)

		if err != nil {
			return nil, fmt.Errorf("Failed to create new client for %s, %v", client_uri, err)
		}

		clients[model] = cl
	}

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
