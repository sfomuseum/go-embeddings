package embeddings

import (
	"context"
	"net/url"

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

func (e *EncoderfileEmbedder) Embeddings(ctx context.Context, content string) ([]float64, error) {

	e32, err := e.Embeddings32(ctx, content)

	if err != nil {
		return nil, err
	}

	return AsFloat64(e32), nil
}

func (e *EncoderfileEmbedder) Embeddings32(ctx context.Context, content string) ([]float32, error) {

	input := []string{content}

	rsp, err := e.client.Embeddings(ctx, input, e.normalize)

	if err != nil {
		return nil, err
	}

	pooled, err := embeddings.PoolOutputResults(rsp)

	if err != nil {
		return nil, err
	}

	return pooled.Embeddings, nil
}

func (e *EncoderfileEmbedder) ImageEmbeddings(ctx context.Context, data []byte) ([]float64, error) {

	return nil, NotImplemented
}

func (e *EncoderfileEmbedder) ImageEmbeddings32(ctx context.Context, data []byte) ([]float32, error) {

	return nil, NotImplemented
}
