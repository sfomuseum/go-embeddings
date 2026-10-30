package embeddings

// https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md#post-v1embeddings-openai-compatible-embeddings-api

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
)

type llamaCppEmbeddingsInput struct {
	Content []llamaCppEmbeddingsContentItem `json:"content"`
}

type llamaCppEmbeddingsContentItem struct {
	Type     string                      `json:"type"`
	Text     string                      `json:"text,omitempty"`
	ImageURL *llamaCppEmbeddingsImageURL `json:"image_url,omitempty"`
}

type llamaCppEmbeddingsImageURL struct {
	URL string `json:"url"`
}

type llamaCppEmbeddingsRequest struct {
	Input          []llamaCppEmbeddingsInput `json:"input"`
	Model          string                    `json:"model"`
	EncodingFormat string                    `json:"encoding_format"`
}

type llamaCppEmbeddingsResponse struct {
	Model           string      `json:"model"`
	Embeddings      [][]float32 `json:"embeddings"`
	TotalDuration   int64       `json:"total_duration"`
	LoadDuration    int64       `json:"load_duration"`
	PromptEvalCount int64       `json:"prompt_eval_count"`
}

type llamaCppClient struct {
	scheme string
	host   string
	client *http.Client
}

func newLlamaCppClient(ctx context.Context, uri string) (*llamaCppClient, error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	http_cl := &http.Client{}

	cl := &llamaCppClient{
		scheme: u.Scheme,
		host:   u.Host,
		client: http_cl,
	}

	return cl, nil
}

func (o *llamaCppClient) embeddings(ctx context.Context, model string, embeddings_type string, body []byte) (*llamaCppEmbeddingsResponse, error) {

	var content llamaCppEmbeddingsContentItem

	switch embeddings_type {
	case "text":

		content = llamaCppEmbeddingsContentItem{
			Type: "text",
			Text: string(body),
		}

	case "image":

		b64 := base64.StdEncoding.EncodeToString(body)

		content = llamaCppEmbeddingsContentItem{
			Type: "image",
			ImageURL: &llamaCppEmbeddingsImageURL{
				URL: fmt.Sprintf("data:image/jpeg;base64,%s", b64),
			},
		}

	default:
		return nil, fmt.Errorf("Invalid embeddings type")
	}

	input := []llamaCppEmbeddingsInput{
		llamaCppEmbeddingsInput{
			Content: []llamaCppEmbeddingsContentItem{
				content,
			},
		},
	}

	req := &llamaCppEmbeddingsRequest{
		Model:          model,
		Input:          input,
		EncodingFormat: "float",
	}

	enc, err := json.Marshal(req)

	if err != nil {
		return nil, err
	}

	br := bytes.NewReader(enc)

	rsp, err := o.execute(ctx, "/v1/embeddings", br)

	if err != nil {
		return nil, err
	}

	defer rsp.Close()

	var emb_rsp *llamaCppEmbeddingsResponse

	dec := json.NewDecoder(rsp)
	err = dec.Decode(&emb_rsp)

	if err != nil {
		return nil, err
	}

	return emb_rsp, nil
}

func (o *llamaCppClient) execute(ctx context.Context, path string, r io.Reader) (io.ReadCloser, error) {

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

	if rsp.StatusCode != http.StatusOK {
		rsp.Body.Close()
		return nil, fmt.Errorf("HTTP request failed, %s", rsp.Status)
	}

	return rsp.Body, nil
}
