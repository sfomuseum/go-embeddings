package embeddings

import (
	"context"
	"fmt"
	"net/url"
	"strings"
)

// For example:


// ROUTE_SEPARATOR is the token that separates the client URI from the
// list of model names in the `client-uri` query parameter.  It is
// defined here to make it easy to change the separator without
// touching every place in the code that uses it.
const ROUTE_SEPARATOR string = "..."

// RouteEmbedder implements the Embedder interface by routing
// requests to different underlying clients depending on the
// requested model.  The generic type parameter T must satisfy the
// Float constraint defined in another file of this package.
//
// The struct embeds an Embedder[T] interface to satisfy the
// interface but the methods below provide the actual routing logic.
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

// NewRouteEmbedder creates a new RouteEmbedder from the supplied URI.
// The URI must be in the form:
//
//     route://?client-uri=CLIENT_URI…MODEL…MODEL
//
// The client URI may be repeated to register multiple clients.
// Each client URI is passed to NewEmbedder64 or NewEmbedder32
// depending on the precision requested by the scheme suffix.
func NewRouteEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, fmt.Errorf("Failed to parse URI, %w", err)
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

		spec := strings.Split(str_spec, ROUTE_SEPARATOR)

		if len(spec) != 2 {
			return nil, fmt.Errorf("?client-uri= parameter must be in the form of '{CLIENT_URI}%s{MODEL}%s{MODEL}'", ROUTE_SEPARATOR, ROUTE_SEPARATOR)
		}

		client_uri := spec[0]
		models := spec[1:]

		if len(models) == 0 {
			return nil, fmt.Errorf("No models defined for client. ?client-uri= parameter must be in the form of '{CLIENT_URI}%s{MODEL}%s{MODEL}'", ROUTE_SEPARATOR, ROUTE_SEPARATOR)
		}

		var cl Embedder[T]
		var stub T

		switch any(stub).(type) {
		case float64:

			cl64, err := NewEmbedder64(ctx, client_uri)

			if err != nil {
				return nil, fmt.Errorf("Failed to create new client for %s: %w", client_uri, err)
			}

			cl = any(cl64).(Embedder[T])

		default:

			cl32, err := NewEmbedder32(ctx, client_uri)

			if err != nil {
				return nil, fmt.Errorf("Failed to create new client for %s: %w", client_uri, err)
			}

			cl = any(cl32).(Embedder[T])
		}

		for _, m := range models {

			has_client, exists := clients[m]

			if exists {
				return nil, fmt.Errorf("Model %s already registered for client (%s)", m, has_client)
			}

			clients[m] = cl
		}
	}

	e := &RouteEmbedder[T]{
		precision: precision,
		scheme:    u.Scheme,
		clients:   clients,
	}

	return e, nil
}

// TextEmbeddings implements the Embedder interface.  It forwards the
// request to the underlying client that matches the requested model.
func (e *RouteEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	client, exists := e.clients[req.Model]

	if !exists {
		return nil, fmt.Errorf("Model not found")
	}

	return client.TextEmbeddings(ctx, req)
}

// ImageEmbeddings implements the Embedder interface.  It forwards the
// request to the underlying client that matches the requested model.
func (e *RouteEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	client, exists := e.clients[req.Model]

	if !exists {
		return nil, fmt.Errorf("Model not found")
	}

	return client.ImageEmbeddings(ctx, req)
}
