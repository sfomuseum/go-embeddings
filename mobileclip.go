//go:build mobileclip

package embeddings

// go run -mod vendor -tags mobileclip cmd/embeddings/main.go -client-uri 'mobileclip://?client-uri=grpc://localhost:8080&model=s0' text hello world
// go run -mod vendor -tags mobileclip cmd/embeddings/main.go -client-uri 'mobileclip://?client-uri=grpc://localhost:8080&model=s0' image ~/Desktop/test22.png

import (
	"context"
	"net/url"

	"github.com/sfomuseum/go-mobileclip"
)

// MobileCLIPEmbedder implements the `Embedder` interface using an MobileCLIP API endpoint to derive embeddings.
type MobileCLIPEmbedder struct {
	Embedder

	client mobileclip.EmbeddingsClient
	model  string
}

func init() {
	ctx := context.Background()
	err := RegisterEmbedder(ctx, "mobileclip", NewMobileCLIPEmbedder)

	if err != nil {
		panic(err)
	}
}

func NewMobileCLIPEmbedder(ctx context.Context, uri string) (Embedder, error) {

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

	e := &MobileCLIPEmbedder{
		client: cl,
		model:  model,
	}

	return e, nil
}

func (e *MobileCLIPEmbedder) Embeddings(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse, error) {

	rsp32, err := e.Embeddings32(ctx, req)

	if err != nil {
		return nil, err
	}

	return EmbeddingsResponse32AsEmbeddingsResponse(rsp32)
}

func (e *MobileCLIPEmbedder) Embeddings32(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse32, error) {

	mc_req := &mobileclip.EmbeddingsRequest{
		Model: e.model,
		Body:  req.Body,
	}

	mc_rsp, err := e.client.ComputeTextEmbeddings(ctx, mc_req)

	if err != nil {
		return nil, err
	}

	now := time.Now()
	ts := now.Unix()

	rsp32 := &EmbeddingsResponse32{
		Id:         req.Id,
		Embeddings: mc_rsp.Embeddings,
		Model:      e.model,
		Created:    ts,
	}

	return rsp32, nil
}

func (e *MobileCLIPEmbedder) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse, error) {

	rsp32, err := e.ImageEmbeddings32(ctx, req)

	if err != nil {
		return nil, err
	}

	return EmbeddingsResponse32AsEmbeddingsResponse(rsp32)
}

func (e *MobileCLIPEmbedder) ImageEmbeddings32(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse32, error) {

	mc_req := &mobileclip.EmbeddingsRequest{
		Model: e.model,
		Body:  req.Body,
	}

	mc_rsp, err := e.client.ComputeImageEmbeddings(ctx, req)

	if err != nil {
		return nil, err
	}

	now := time.Now()
	ts := now.Unix()

	rsp32 := &EmbeddingsResponse32{
		Id:         req.Id,
		Embeddings: mc_rsp.Embeddings,
		Model:      e.model,
		Created:    ts,
	}

	return rsp32, nil
}
