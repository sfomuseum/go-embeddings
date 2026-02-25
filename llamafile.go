package embeddings

// Note that newer versions of llamafile no longer generate embeddings. See encoderfile.go

// https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints
// https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#other-example-llamafiles
//
// curl --request POST --url http://localhost:8080/embedding --header "Content-Type: application/json" --data '{"content": "Hello world" }'

import (
	"context"
	"encoding/base64"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// LlamafileEmbedder implements the `Embedder` interface using an Llamafile API endpoint to derive embeddings.
type LlamafileEmbedder[T Float] struct {
	Embedder[T]
	client    *llamafileClient
	precision string
}

func init() {
	ctx := context.Background()
	RegisterEmbedder[float64](ctx, "llamafile", NewLlamafileEmbedder)
	RegisterEmbedder[float32](ctx, "llamafile32", NewLlamafileEmbedder)
	RegisterEmbedder[float64](ctx, "llamafile64", NewLlamafileEmbedder)
}

func NewLlamafileEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	q := u.Query()

	client_uri := "http://localhost:8080"

	if q.Has("client-uri") {
		client_uri = q.Get("client-uri")
	}

	llamafile_cl, err := newLlamafileClient(ctx, client_uri)

	if err != nil {
		return nil, err
	}

	precision := "float64"

	if strings.HasSuffix(u.Scheme, "32") {
		precision = fmt.Sprintf("%s#as%d", precision, 32)
	}

	e := &LlamafileEmbedder[T]{
		client:    llamafile_cl,
		precision: precision,
	}

	return e, nil
}

func (e *LlamafileEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	ll_req := &llamafileEmbeddingRequest{
		Content: string(req.Body),
	}

	ll_rsp, err := e.client.embeddings(ctx, ll_req)

	if err != nil {
		return nil, err
	}

	rsp := e.llamafileResponseToEmbeddingsResponse(req, ll_rsp)
	return rsp, nil
}

func (e *LlamafileEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	data_b64 := base64.StdEncoding.EncodeToString(req.Body)

	now := time.Now()
	ts := now.Unix()

	image_req := &llamafileImageDataEmbeddingRequest{
		Data: data_b64,
		Id:   ts,
	}

	ll_req := &llamafileEmbeddingRequest{
		ImageData: []*llamafileImageDataEmbeddingRequest{
			image_req,
		},
	}

	ll_rsp, err := e.client.embeddings(ctx, ll_req)

	if err != nil {
		return nil, err
	}

	rsp := e.llamafileResponseToEmbeddingsResponse(req, ll_rsp)
	return rsp, nil
}

func (e *LlamafileEmbedder[T]) llamafileResponseToEmbeddingsResponse(req *EmbeddingsRequest, ll_rsp *llamafileEmbeddingResponse) EmbeddingsResponse[T] {

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonPrecision: e.precision,
		CommonCreated:   ts,
		CommonModel:     "",
	}

	e64 := ll_rsp.Embeddings

	switch {
	case strings.HasSuffix(e.precision, "32"):
		rsp.CommonEmbeddings = toFloat32Slice[T](AsFloat32(e64))
	default:
		rsp.CommonEmbeddings = toFloat64Slice[T](e64)
	}

	return rsp
}
