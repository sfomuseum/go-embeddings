//go:build ollama

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

func (e *OllamaEmbedder) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse, error) {

	cl_rsp, err := e.client.embeddings(ctx, e.model, string(req.Body))

	if err != nil {
		return nil, err
	}

	now := time.Now()
	ts := now.Unix()

	rsp32 := &CommonEmbeddingsResponse{
		CommonId:         req.Id,
		CommonModel:      e.model,
		CommonEmbeddings: cl_rsp.Embeddings[0],
		CommonCreated:    ts,
		CommonPrecision:  32,
	}

	return rsp32, nil
}

func (e *OllamaEmbedder) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse, error) {
	return nil, NotImplemented
}
