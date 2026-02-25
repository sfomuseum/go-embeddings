package embeddings

// go run -mod vendor -tags encoderfile cmd/embeddings/main.go -client-uri 'encoderfile://?client-uri=http://localhost:8080' text ./README.md
// go run -mod vendor -tags encoderfile cmd/embeddings/main.go -client-uri 'encoderfile://?client-uri=http://localhost:8080' image ~/Desktop/test22.png
// 2026/02/20 16:56:51 Failed to derive embeddings, Not implemented

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/sfomuseum/go-encoderfile/client"
	"github.com/sfomuseum/go-encoderfile/embeddings"
)

// EncoderfileEmbedder implements the `Embedder` interface using an Encoderfile API endpoint to derive embeddings.
type EncoderfileEmbedder[T Float] struct {
	Embedder[T]

	client    client.Client
	precision string
	normalize bool
}

func init() {
	ctx := context.Background()
	RegisterEmbedder[float32](ctx, "encoderfile", NewEncoderfileEmbedder)
	RegisterEmbedder[float32](ctx, "encoderfile32", NewEncoderfileEmbedder)
	RegisterEmbedder[float64](ctx, "encoderfile64", NewEncoderfileEmbedder)
}

func NewEncoderfileEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	q := u.Query()

	client_uri := "http://localhost:8080"

	if q.Has("client-uri") {
		client_uri = q.Get("client-uri")
	}

	cl, err := client.NewClient(ctx, client_uri)

	if err != nil {
		return nil, err
	}

	precision := "float32"

	if strings.HasSuffix(u.Scheme, "64") {
		precision = fmt.Sprintf("%s#as-float%d", precision, 64)
	}

	e := &EncoderfileEmbedder[T]{
		client:    cl,
		normalize: true,
		precision: precision,
	}

	return e, nil
}

func (e *EncoderfileEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

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

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonPrecision: e.precision,
		CommonModel:     cl_rsp.ModelId,
		CommonCreated:   ts,
	}

	switch {
	case strings.HasSuffix(e.precision, "64"):
		rsp.CommonEmbeddings = toFloat64Slice[T](AsFloat64(e32))
	default:
		rsp.CommonEmbeddings = toFloat32Slice[T](e32)
	}

	return rsp, nil
}

func (e *EncoderfileEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {
	return nil, NotImplemented
}
