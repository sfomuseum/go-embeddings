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

func (e *OllamaEmbedder) Embeddings(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse, error) {

	rsp32, err := e.Embeddings32(ctx, req)

	if err != nil {
		return nil, err
	}

	return Embeddings32ResponseToEmbeddingsResponse(rsp32)
}

func (e *OllamaEmbedder) Embeddings32(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse32, error) {

	cl_rsp, err := e.client.embeddings(ctx, e.model, string(req.Body))

	if err != nil {
		return nil, err
	}

	now := time.Now()
	ts := now.Unix()

	rsp32 := &EmbeddingsResponse32{
		Id:         req.Id,
		Model:      e.model,
		Embeddings: cl_rsp.Embeddings[0],
		Created:    ts,
	}

	return rsp32, nil
}

func (e *OllamaEmbedder) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse64, error) {
	return nil, NotImplemented
}

func (e *OllamaEmbedder) ImageEmbeddings32(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse32, error) {
	return nil, NotImplemented
}
