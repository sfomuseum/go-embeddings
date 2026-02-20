package embeddings

import (
	"context"
	"net/url"
)

// OllamaEmbedder implements the `Embedder` interface using an Ollama API endpoint to derive embeddings.
type OllamaEmbedder struct {
	Embedder
	client *ollamaClient
	model  string
}

func init() {
	ctx := context.Background()

	schemes := []string{
		"ollama",
		"ollamas",
	}

	for _, s := range schemes {
		err := RegisterEmbedder(ctx, s, NewOllamaEmbedder)

		if err != nil {
			panic(err)
		}
	}
}

func NewOllamaEmbedder(ctx context.Context, uri string) (Embedder, error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	q := u.Query()

	scheme := "http"

	if u.Scheme == "ollamas" {
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

	e := &OllamaEmbedder{
		client: cl,
		model:  model,
	}

	return e, nil
}

func (e *OllamaEmbedder) Embeddings(ctx context.Context, content string) ([]float64, error) {

	e32, err := e.Embeddings32(ctx, content)

	if err != nil {
		return nil, err
	}

	return AsFloat64(e32), nil
}

func (e *OllamaEmbedder) Embeddings32(ctx context.Context, content string) ([]float32, error) {

	rsp, err := e.client.embeddings(ctx, e.model, content)

	if err != nil {
		return nil, err
	}

	return rsp.Embeddings[0], nil
}

func (e *OllamaEmbedder) ImageEmbeddings(ctx context.Context, data []byte) ([]float64, error) {
	return nil, NotImplemented
}

func (e *OllamaEmbedder) ImageEmbeddings32(ctx context.Context, data []byte) ([]float32, error) {
	return nil, NotImplemented
}
