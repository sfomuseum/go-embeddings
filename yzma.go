package embeddings

import (
	"context"
	"net/url"
	"strings"
	"time"

	"github.com/hybridgroup/yzma/pkg/llama"
	"github.com/hybridgroup/yzma/pkg/download"	
)

// YzmaEmbedder implements the `Embedder` interface using an Yzma API endpoint to derive embeddings.
type YzmaEmbedder[T Float] struct {
	Embedder[T]
	precision string
	scheme    string
}

func init() {
	ctx := context.Background()

	RegisterEmbedder[float32](ctx, "yzma", NewYzmaEmbedder[float32])
	RegisterEmbedder[float32](ctx, "yzma32", NewYzmaEmbedder[float32])
	RegisterEmbedder[float64](ctx, "yzma64", NewYzmaEmbedder[float64])
}

func NewYzmaEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	precision := "float32"

	switch {
	case strings.HasSuffix(u.Scheme, "64"):
		precision = "%s#as-float64"
	}

	lib_path := u.Path

	if !download.AlreadyInstalled(lib_path){
		// https://pkg.go.dev/github.com/hybridgroup/yzma@v1.25.0/pkg/download#Install
	}
	
	err = llama.Load(lib_path)

	if err != nil {
		return nil, err
	}

	llama.Init()
	
	e := &YzmaEmbedder[T]{
		precision: precision,
		scheme:    u.Scheme,
	}

	return e, nil
}

func (e *YzmaEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	// https://github.com/hybridgroup/yzma/blob/main/examples/embeddings/main.go
	return nil, NotImplemented	
}

func (e *YzmaEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {
	return nil, NotImplemented
}

func (e *YzmaEmbedder[T]) yzmaEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:         req.Id,
		CommonEmbeddings: make([]T, 0),
		CommonModel:      "yzma",
		CommonCreated:    ts,
		CommonPrecision:  e.precision,
	}

	return rsp, nil
}

func (e *YzmaEmbedder[T]) Close() error {
	llama.Close()
	return nil
}
