package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

type llamafileImageDataEmbeddingRequest struct {
	Id   int64  `json:"id"`
	Data string `json:"data"`
}

type llamafileEmbeddingRequest struct {
	Content   string                                `json:"content,omitempty"`
	ImageData []*llamafileImageDataEmbeddingRequest `json:"image_data,omitempty"`
}

type llamafileEmbeddingResponse struct {
	Embeddings []float64 `json:"embedding,omitempty"`
}

type llamafileClient struct {
	client *http.Client
	host   string
	port   string
	tls    bool
}

func newLlamafileClient(ctx context.Context, uri string) (*llamafileClient, error) {

	host := "localhost"
	port := "8080"
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

	q := u.Query()

	if q.Has("tls") {

		v, err := strconv.ParseBool("tls")

		if err != nil {
			return nil, fmt.Errorf("Invalid ?tls= parameter, %w", err)
		}

		tls = v
	}

	cl := &http.Client{}

	e := &llamafileClient{
		client: cl,
		host:   host,
		port:   port,
		tls:    tls,
	}

	return e, nil
}

func (e *llamafileClient) embeddings(ctx context.Context, llamafile_req *llamafileEmbeddingRequest) (*llamafileEmbeddingResponse, error) {

	u := url.URL{}
	u.Scheme = "http"
	u.Host = fmt.Sprintf("%s:%s", e.host, e.port)
	u.Path = "/embedding"

	if e.tls {
		u.Scheme = "https"
	}

	endpoint := u.String()

	enc_msg, err := json.Marshal(llamafile_req)

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

	// body, _ := io.ReadAll(rsp.Body)
	// fmt.Println("WUT", string(body))

	var llamafile_rsp *llamafileEmbeddingResponse

	dec := json.NewDecoder(rsp.Body)
	err = dec.Decode(&llamafile_rsp)

	if err != nil {
		return nil, fmt.Errorf("Failed to unmarshal embeddings, %w", err)
	}

	return llamafile_rsp, nil
}
