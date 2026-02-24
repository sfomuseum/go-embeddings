//go:build mobileclip

package embeddings

// go run -mod vendor -tags mobileclip cmd/embeddings/main.go -client-uri 'mobileclip://?client-uri=grpc://localhost:8080&model=s0' text hello world
// go run -mod vendor -tags mobileclip cmd/embeddings/main.go -client-uri 'mobileclip://?client-uri=grpc://localhost:8080&model=s0' image ~/Desktop/test22.png

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"

	"github.com/sfomuseum/go-mobileclip"
)

// MobileCLIPEmbedder implements the `Embedder` interface using an MobileCLIP API endpoint to derive embeddings.
type MobileCLIPEmbedder[T Float] struct {
	Embedder[T]
	client    mobileclip.EmbeddingsClient
	model     string
	precision string
}

func init() {
	ctx := context.Background()
	RegisterEmbedder[float32](ctx, "mobileclip", NewMobileCLIPEmbedder)
	RegisterEmbedder[float32](ctx, "mobileclip32", NewMobileCLIPEmbedder)
	RegisterEmbedder[float64](ctx, "mobileclip64", NewMobileCLIPEmbedder)
}

func NewMobileCLIPEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	q := u.Query()

	client_uri := q.Get("client-uri")
	model := q.Get("model")

	cl, err := mobileclip.NewEmbeddingsClient(ctx, client_uri)

	if err != nil {
		return nil, err
	}

	precision := "32"

	if strings.HasSuffix(u.Scheme, "64") {
		precision = fmt.Sprintf("%s#%d", precision, 64)
	}

	e := &MobileCLIPEmbedder[T]{
		client:    cl,
		model:     model,
		precision: precision,
	}

	return e, nil
}

func (e *MobileCLIPEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	mc_req := &mobileclip.EmbeddingsRequest{
		Model: e.model,
		Body:  req.Body,
	}

	mc_rsp, err := e.client.ComputeTextEmbeddings(ctx, mc_req)

	if err != nil {
		return nil, err
	}

	rsp := e.mobileCLIPResponseToEmbeddingsResponse(req, mc_rsp)
	return rsp, nil
}

func (e *MobileCLIPEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	mc_req := &mobileclip.EmbeddingsRequest{
		Model: e.model,
		Body:  req.Body,
	}

	mc_rsp, err := e.client.ComputeImageEmbeddings(ctx, mc_req)

	if err != nil {
		return nil, err
	}

	rsp := e.mobileCLIPResponseToEmbeddingsResponse(req, mc_rsp)
	return rsp, nil
}

func (e *MobileCLIPEmbedder[T]) mobileCLIPResponseToEmbeddingsResponse(req *EmbeddingsRequest, mc_rsp *mobileclip.Embeddings) EmbeddingsResponse[T] {

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonPrecision: e.precision,
		CommonCreated:   ts,
		CommonModel:     e.model,
	}

	e32 := mc_rsp.Embeddings

	switch {
	case strings.HasSuffix(e.precision, "64"):
		rsp.CommonEmbeddings = toFloat64Slice[T](AsFloat64(e32))
	default:
		rsp.CommonEmbeddings = toFloat32Slice[T](e32)
	}

	return rsp
}
