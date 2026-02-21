package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"

	_ "os"

	"github.com/sfomuseum/go-encoderfile/embeddings"
)

type HTTPClient struct {
	Client
	endpoint    *url.URL
	http_client *http.Client
}

func init() {

	schemes := []string{
		"http",
		"https",
	}

	for _, scheme := range schemes {

		err := RegisterClient(context.Background(), scheme, NewHTTPClient)

		if err != nil {
			panic(err)
		}
	}
}

func NewHTTPClient(ctx context.Context, uri string) (Client, error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, fmt.Errorf("Failed to parse URI, %w", err)
	}

	cl := &HTTPClient{
		http_client: &http.Client{},
		endpoint:    u,
	}

	return cl, nil
}

func (cl *HTTPClient) Embeddings(ctx context.Context, inputs []string, normalize bool) (*embeddings.EmbeddingsOutput, error) {

	emb_input := &embeddings.EmbeddingsInput{
		Inputs:    inputs,
		Normalize: normalize,
	}

	enc, err := json.Marshal(emb_input)

	if err != nil {
		return nil, err
	}

	body := bytes.NewReader(enc)

	rsp, err := cl.execute(ctx, "POST", "/predict", body)

	if err != nil {
		return nil, err
	}

	defer rsp.Close()

	var output *embeddings.EmbeddingsOutput

	dec := json.NewDecoder(rsp)
	err = dec.Decode(&output)

	if err != nil {
		return nil, err
	}

	return output, nil
}

func (cl *HTTPClient) execute(ctx context.Context, method string, path string, body io.Reader) (io.ReadCloser, error) {

	u := cl.newURL(path)

	req, err := http.NewRequestWithContext(ctx, method, u.String(), body)

	if err != nil {
		return nil, err
	}

	req.Header.Set("Content-Type", "application/json")

	// req2 := req.Clone(ctx)
	// req2.Write(os.Stderr)

	rsp, err := cl.http_client.Do(req)

	if err != nil {
		return nil, err
	}

	if rsp.StatusCode != http.StatusOK {
		rsp.Body.Close()
		return nil, fmt.Errorf("%d %s", rsp.StatusCode, rsp.Status)
	}

	return rsp.Body, nil
}

func (cl *HTTPClient) newURL(path string) *url.URL {

	u := &url.URL{}
	u.Scheme = cl.endpoint.Scheme
	u.Host = cl.endpoint.Host
	u.Path = path

	return u
}
