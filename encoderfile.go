//go:build encoderfile

package embeddings

// go run -mod vendor -tags encoderfile cmd/embeddings/main.go -client-uri 'encoderfile://?client-uri=http://localhost:8080' text ./README.md
// go run -mod vendor -tags encoderfile cmd/embeddings/main.go -client-uri 'encoderfile://?client-uri=http://localhost:8080' image ~/Desktop/test22.png
// 2026/02/20 16:56:51 Failed to derive embeddings, Not implemented

import (
	"context"
	"net/url"
	"time"

	"github.com/sfomuseum/go-encoderfile/client"
	"github.com/sfomuseum/go-encoderfile/embeddings"
)

// EncoderfileEmbedder implements the `Embedder` interface using an Encoderfile API endpoint to derive embeddings.
type EncoderfileEmbedder struct {
	Embedder

	client    client.Client
	normalize bool
}

func init() {
	ctx := context.Background()
	err := RegisterEmbedder(ctx, "encoderfile", NewEncoderfileEmbedder)

	if err != nil {
		panic(err)
	}
}

func NewEncoderfileEmbedder(ctx context.Context, uri string) (Embedder, error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	q := u.Query()

	client_uri := q.Get("client-uri")

	cl, err := client.NewClient(ctx, client_uri)

	if err != nil {
		return nil, err
	}

	e := &EncoderfileEmbedder{
		client:    cl,
		normalize: true,
	}

	return e, nil
}

func (e *EncoderfileEmbedder) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse, error) {

	input := []string{
		string(req.Body),
	}

	cl_rsp, err := e.client.Embeddings(ctx, input, e.normalize)

	if err != nil {
		return nil, err
	}

	pooled, err := embeddings.PoolOutputResults(cl_rsp)

	if err != nil {
		return nil, err
	}

	e32 := pooled.Embeddings

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse{
		Id:           req.Id,
		Embeddings32: e32,
		Precision:    32,
		Model:        "fixme",
		Created:      ts,
	}

	return rsp, nil
}

func (e *EncoderfileEmbedder) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse, error) {
	return nil, NotImplemented
}
