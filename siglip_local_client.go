package embeddings

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/url"
	"strings"
	"time"
)

func init() {
	ctx := context.Background()
	RegisterEmbedder[float32](ctx, "siglip-client", NewSigLIPLocalClientEmbedder)
	RegisterEmbedder[float32](ctx, "siglip-client32", NewSigLIPLocalClientEmbedder)
	RegisterEmbedder[float64](ctx, "siglip-client64", NewSigLIPLocalClientEmbedder)
}

type SigLIPLocalClientEmbedder[T Float] struct {
	Embedder[T]
	client    *LocalClient
	precision string
}

func NewSigLIPLocalClientEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	q := u.Query()

	client_uri := "http://localhost:5000"

	if q.Has("client-uri") {
		client_uri = q.Get("client-uri")
	}
	
	cl, err := NewLocalClient(ctx, client_uri)

	if err != nil {
		return nil, err
	}

	precision := "float32"

	if strings.HasSuffix(u.Scheme, "64") {
		precision = fmt.Sprintf("%s#as-float%d", precision, 64)
	}

	e := &SigLIPLocalClientEmbedder[T]{
		client:    cl,
		precision: precision,
	}

	return e, nil
}

func (e *SigLIPLocalClientEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

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

func (e *SigLIPLocalClientEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

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

func (e *SigLIPLocalClientEmbedder[T]) localClientResponseToEmbeddingsResponse(req *EmbeddingsRequest, cl_rsp *LocalClientEmbeddingResponse) EmbeddingsResponse[T] {

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonPrecision: e.precision,
		CommonCreated:   ts,
		CommonModel:     cl_rsp.Model,
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
