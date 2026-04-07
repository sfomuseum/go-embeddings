package embeddings

// https://github.com/mlfoundations/open_clip

import (
	"context"
	"encoding/base64"
	"net/url"
	"strings"
	"time"
)

// OpenCLIPEmbedder implements the `Embedder` interface using an OpenCLIP API endpoint to derive embeddings.
type OpenCLIPEmbedder[T Float] struct {
	Embedder[T]
	client    *LocalClient
	precision string
}

func init() {
	ctx := context.Background()
	RegisterEmbedder[float64](ctx, "openclip", NewOpenCLIPEmbedder)
	RegisterEmbedder[float32](ctx, "openclip32", NewOpenCLIPEmbedder)
	RegisterEmbedder[float64](ctx, "openclip64", NewOpenCLIPEmbedder)
}

func NewOpenCLIPEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	q := u.Query()

	client_uri := "http://localhost:5000"

	if q.Has("client-uri") {
		client_uri = q.Get("client-uri")
	}

	local_cl, err := NewLocalClient(ctx, client_uri)

	if err != nil {
		return nil, err
	}

	e := &OpenCLIPEmbedder[T]{
		client: local_cl,
	}

	return e, nil
}

func (e *OpenCLIPEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	cl_req := &LocalClientEmbeddingRequest{
		Content: string(req.Body),
	}

	cl_rsp, err := e.client.embeddings(ctx, cl_req)

	if err != nil {
		return nil, err
	}

	rsp := e.localClientResponseToEmbeddingsResponse(req, cl_rsp)
	return rsp, nil
}

func (e *OpenCLIPEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	data_b64 := base64.StdEncoding.EncodeToString(req.Body)

	now := time.Now()
	ts := now.Unix()

	image_req := &LocalClientImageDataEmbeddingRequest{
		Data: data_b64,
		Id:   ts,
	}

	cl_req := &LocalClientEmbeddingRequest{
		ImageData: []*LocalClientImageDataEmbeddingRequest{
			image_req,
		},
	}

	cl_rsp, err := e.client.embeddings(ctx, cl_req)

	if err != nil {
		return nil, err
	}

	rsp := e.localClientResponseToEmbeddingsResponse(req, cl_rsp)
	return rsp, nil
}

func (e *OpenCLIPEmbedder[T]) localClientResponseToEmbeddingsResponse(req *EmbeddingsRequest, cl_rsp *LocalClientEmbeddingResponse) EmbeddingsResponse[T] {

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonPrecision: e.precision,
		CommonCreated:   ts,
		CommonModel:     "openclip",
	}

	e64 := cl_rsp.Embeddings

	switch {
	case strings.HasSuffix(e.precision, "32"):
		rsp.CommonEmbeddings = toFloat32Slice[T](AsFloat32(e64))
	default:
		rsp.CommonEmbeddings = toFloat64Slice[T](e64)
	}

	return rsp
}
