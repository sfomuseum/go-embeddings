package embeddings

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/url"
	"strings"
	"time"
)

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

	cl, err := NewLocalClient(ctx, uri)

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

	cl_req := &LocalEmbeddingRequest{
		Content: string(req.Body),
	}

	cl_rsp, err := e.client.embeddings(ctx, cl_req)

	if err != nil {
		return nil, err
	}

	rsp := e.localResponseToEmbeddingsResponse(req, cl_rsp)
	return rsp, nil
}

func (e *SigLIPLocalClientEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	data_b64 := base64.StdEncoding.EncodeToString(req.Body)

	now := time.Now()
	ts := now.Unix()

	image_req := &LocalImageDataEmbeddingRequest{
		Data: data_b64,
		Id:   ts,
	}

	cl_req := &LocalEmbeddingRequest{
		ImageData: []*LocalImageDataEmbeddingRequest{
			image_req,
		},
	}

	cl_rsp, err := e.client.embeddings(ctx, cl_req)

	if err != nil {
		return nil, err
	}

	rsp := e.localResponseToEmbeddingsResponse(req, cl_rsp)
	return rsp, nil
}

func (e *SigLIPLocalClientEmbedder[T]) localResponseToEmbeddingsResponse(req *EmbeddingsRequest, cl_rsp *LocalEmbeddingResponse) EmbeddingsResponse[T] {

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
