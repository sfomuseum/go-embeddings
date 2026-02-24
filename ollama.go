//go:build ollama

package embeddings

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// OllamaEmbedder implements the `Embedder` interface using an Ollama API endpoint to derive embeddings.
type OllamaEmbedder[T Float] struct {
	Embedder[T]
	client    *ollamaClient
	model     string
	precision string
}

func init() {
	ctx := context.Background()

	schemes := []string{
		"ollama",
		"ollamas",
	}

	for _, s := range schemes {

		s32 := fmt.Sprintf("%s32", s)
		s64 := fmt.Sprintf("%s64", s)

		RegisterEmbedder[float32](ctx, s, NewOllamaEmbedder[float32])
		RegisterEmbedder[float32](ctx, s32, NewOllamaEmbedder[float32])
		RegisterEmbedder[float64](ctx, s64, NewOllamaEmbedder[float64])
	}
}

func NewOllamaEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	q := u.Query()

	scheme := "http"

	if strings.HasPrefix(u.Scheme, "ollamas") {
		scheme = "https"
	}

	host := u.Host

	if host == "" {
		host = "localhost:11434"
	}

	client_uri := url.URL{}
	client_uri.Scheme = scheme
	client_uri.Host = host

	cl, err := newOllamaClient(ctx, client_uri.String())

	if err != nil {
		return nil, err
	}

	model := q.Get("model")

	precision := "32"

	if strings.HasSuffix(u.Scheme, "64") {
		scheme = fmt.Sprintf("%s#%d", precision, 64)
	}

	e := &OllamaEmbedder[T]{
		client:    cl,
		model:     model,
		precision: precision,
	}

	return e, nil
}

func (e *OllamaEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	cl_rsp, err := e.client.embeddings(ctx, e.model, string(req.Body))

	if err != nil {
		return nil, err
	}

	e32 := cl_rsp.Embeddings[0]

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonModel:     e.model,
		CommonCreated:   ts,
		CommonPrecision: "32",
	}

	switch {
	case strings.HasSuffix(e.precision, "64"):
		rsp.CommonEmbeddings = toFloat64Slice[T](AsFloat64(e32))
	default:
		rsp.CommonEmbeddings = toFloat32Slice[T](e32)
	}

	return rsp, nil
}

func (e *OllamaEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {
	return nil, NotImplemented
}
