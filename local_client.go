package embeddings

// For talking to the simple HTTP "server" wrappers for non-Go embeddings interface.
// Typically these are Flask-based servers.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

type LocalClientImageDataEmbeddingRequest struct {
	Id   int64  `json:"id"`
	Data string `json:"data"`
}

type LocalClientEmbeddingRequest struct {
	Content   string                                  `json:"content,omitempty"`
	ImageData []*LocalClientImageDataEmbeddingRequest `json:"image_data,omitempty"`
}

type LocalClientEmbeddingResponse struct {
	Model      string    `json:"model,omitempty"`
	Embeddings []float64 `json:"embedding,omitempty"`
}

type LocalClient struct {
	client *http.Client
	host   string
	port   string
	tls    bool
}

func NewLocalClient(ctx context.Context, uri string) (*LocalClient, error) {

	host := "127.0.0.1"
	port := "5000"
	tls := false

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	if u.Host != "" {
		host = u.Host

		parts := strings.Split(host, ":")

		if len(parts) < 1 {
			return nil, fmt.Errorf("Failed to parse host component of URI")
		}

		host = parts[0]
	}

	if u.Port() != "" {
		port = u.Port()
	}

	slog.Debug("URL", "host", host, "port", port)

	q := u.Query()

	if q.Has("tls") {

		v, err := strconv.ParseBool("tls")

		if err != nil {
			return nil, fmt.Errorf("Invalid ?tls= parameter, %w", err)
		}

		tls = v
	}

	http_cl := &http.Client{}

	cl := &LocalClient{
		client: http_cl,
		host:   host,
		port:   port,
		tls:    tls,
	}

	return cl, nil
}

func (e *LocalClient) embeddings(ctx context.Context, local_req *LocalClientEmbeddingRequest) (*LocalClientEmbeddingResponse, error) {

	u := url.URL{}
	u.Scheme = "http"
	u.Host = fmt.Sprintf("%s:%s", e.host, e.port)

	if len(local_req.ImageData) > 0 {
		u.Path = "/embeddings/image"
	} else {
		u.Path = "/embeddings"
	}

	if e.tls {
		u.Scheme = "https"
	}

	endpoint := u.String()

	enc_msg, err := json.Marshal(local_req)

	if err != nil {
		return nil, fmt.Errorf("Failed to encode message, %w", err)
	}

	br := bytes.NewReader(enc_msg)

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, br)

	if err != nil {
		return nil, fmt.Errorf("Failed to create new request, %w", err)
	}

	req.Header.Set("Content-type", "application/json")

	rsp, err := e.client.Do(req)

	if err != nil {
		return nil, fmt.Errorf("Failed to execute request, %w", err)
	}

	defer rsp.Body.Close()

	if rsp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Embeddings request failed %d: %s", rsp.StatusCode, rsp.Status)
	}

	var local_rsp *LocalClientEmbeddingResponse

	dec := json.NewDecoder(rsp.Body)
	err = dec.Decode(&local_rsp)

	if err != nil {
		return nil, fmt.Errorf("Failed to unmarshal embeddings, %w", err)
	}

	return local_rsp, nil
}
