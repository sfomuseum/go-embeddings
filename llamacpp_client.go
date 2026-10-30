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
	"strings"

	"github.com/gabriel-vasile/mimetype"
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
	Input          []any  `json:"input"` // Flexible to match string vs multimodal spec
	Model          string `json:"model,omitempty"`
	EncodingFormat string `json:"encoding_format"`
}

// llamaCppEmbeddingsEmbeddingsResponse represents the top-level JSON response from the /v1/embeddings endpoint.
type llamaCppEmbeddingsResponse struct {
	Object string                            `json:"object"`
	Data   []llamaCppEmbeddingsEmbeddingData `json:"data"`
	Model  string                            `json:"model"`
	Usage  llamaCppEmbeddingsUsage           `json:"usage"`
}

// llamaCppEmbeddingsEmbeddingData represents an individual vector and its position index.
type llamaCppEmbeddingsEmbeddingData struct {
	Object    string    `json:"object"`
	Index     int       `json:"index"`
	Embedding []float32 `json:"embedding"`
}

// llamaCppEmbeddingsUsage captures the token metrics for billing and limit tracking.
type llamaCppEmbeddingsUsage struct {
	PromptTokens int `json:"prompt_tokens"`
	TotalTokens  int `json:"total_tokens"`
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

	// Note: Leaving model in the method signature for a time when llama.cpp
	// allows Ollama-style model toggling

	var content any

	switch embeddings_type {
	case "text":

		content = llamaCppEmbeddingsContentItem{
			Type: "text",
			Text: string(body),
		}

	case "image":

		m := mimetype.Detect(body)

		if !strings.HasPrefix(m.String(), "image/") {
			return nil, fmt.Errorf("Failed to determine image mimetype")
		}

		b64 := base64.StdEncoding.EncodeToString(body)

		content = llamaCppEmbeddingsContentItem{
			Type: "image_url",
			ImageURL: &llamaCppEmbeddingsImageURL{
				URL: fmt.Sprintf("data:%s;base64,%s", m.String(), b64),
			},
		}

	default:
		return nil, fmt.Errorf("Invalid embeddings type")
	}

	input := map[string]any{
		"content": []any{
			content,
		},
	}

	req := &llamaCppEmbeddingsRequest{
		Input:          []any{input},
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

func (o *llamaCppClient) execute(ctx context.Context, path string, r io.ReadSeeker) (io.ReadCloser, error) {

	u := url.URL{}
	u.Scheme = o.scheme
	u.Host = o.host
	u.Path = path

	req, err := http.NewRequestWithContext(ctx, "POST", u.String(), r)

	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	// r2 := req.Clone(ctx)
	// r2.Write(os.Stdout)
	// r.Seek(0, 0)

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
