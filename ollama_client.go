package embeddings

// https://docs.ollama.com/api/introduction

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/url"
)

type ollamaEmbeddingsRequest struct {
	Model string `json:"model"`
	Input string `json:"input"`
}

type ollamaEmbeddingsResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float32 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration"`
	LoadDuration    int64       `json:"load_duration"`
	PromptEvalCount int64       `json:"prompt_eval_count"`
}

type ollamaClient struct {
	scheme string
	host   string
	client *http.Client
}

func newOllamaClient(ctx context.Context, uri string) (*ollamaClient, error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	http_cl := &http.Client{}

	cl := &ollamaClient{
		scheme: u.Scheme,
		host:   u.Host,
		client: http_cl,
	}

	return cl, nil
}

func (o *ollamaClient) embeddings(ctx context.Context, model string, input string) (*ollamaEmbeddingsResponse, error) {

	req := &ollamaEmbeddingsRequest{
		Model: model,
		Input: input,
	}

	enc, err := json.Marshal(req)

	if err != nil {
		return nil, err
	}

	body := bytes.NewReader(enc)

	rsp, err := o.execute(ctx, "/api/embed", body)

	if err != nil {
		return nil, err
	}

	defer rsp.Close()

	var emb_rsp *ollamaEmbeddingsResponse

	dec := json.NewDecoder(rsp)
	err = dec.Decode(&emb_rsp)

	if err != nil {
		return nil, err
	}

	return emb_rsp, nil
}

func (o *ollamaClient) execute(ctx context.Context, path string, r io.Reader) (io.ReadCloser, error) {

	// http_ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	// defer cancel()

	u := url.URL{}
	u.Scheme = o.scheme
	u.Host = o.host
	u.Path = path

	req, err := http.NewRequestWithContext(ctx, "POST", u.String(), r)

	if err != nil {
		return nil, err
	}

	rsp, err := o.client.Do(req)

	if err != nil {
		return nil, err
	}

	return rsp.Body, nil
}
