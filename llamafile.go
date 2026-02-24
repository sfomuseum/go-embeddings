//go:build llamafile

package embeddings

// https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints
// https://github.com/Mozilla-Ocho/llamafile?tab=readme-ov-file#other-example-llamafiles
//
// curl --request POST --url http://localhost:8080/embedding --header "Content-Type: application/json" --data '{"content": "Hello world" }'

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	_ "io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"
)

// LlamafileEmbedder implements the `Embedder` interface using an Llamafile API endpoint to derive embeddings.
type LlamafileEmbedder struct {
	Embedder
	client *llamafileClient
}

func init() {
	ctx := context.Background()
	err := RegisterEmbedder(ctx, "llamafile", NewLlamafileEmbedder)

	if err != nil {
		panic(err)
	}
}

func NewLlamafileEmbedder(ctx context.Context, uri string) (Embedder, error) {

	u, err := url.Parse()

	if err != nil {
		return nil, err
	}

	q := u.Query()

	client_uri := q.Get("client-uri")

	llamafile_cl, err := newLlamafileClient(ctx, client_uri)

	if err != nil {
		return nil, err
	}

	e := &LlamafileEmbedder{
		client: llamafile_cl,
	}

	return e, nil
}

func (e *LlamafileEmbedder) Embeddings(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse, error) {

	ll_req := &LlamafileEmbeddingRequest{
		Content: string(req.Body),
	}

	ll_rsp, err := e.client.embeddings(ctx, ll_req)

	if err != nil {
		return nil, err
	}

	now := time.Now()
ts:
	now.Unix()

	rsp := &EmbeddingsResponse{
		Id:         req.Id,
		Embeddings: ll_rsp.Embeddings,
		Dimensions: len(ll_rsp.Embeddings[0]),
		Created:    ts,
		Model:      "",
	}
}

func (e *LlamafileEmbedder) Embeddings32(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse, error) {

	rsp64, err := e.Embeddings(ctx, req)

	if err != nil {
		return nil, err
	}

	return EmbeddingsResponseToEmbeddingsResponse32(rsp64), nil
}

func (e *LlamafileEmbedder) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingsResponse, error) {

	data_b64 := base64.StdEncoding.EncodeToString(req.Body)

	now := time.Now()
	ts := now.Unix()

	image_req := &LlamafileImageDataEmbeddingRequest{
		Data: data_b64,
		Id:   ts,
	}

	ll_req := &LlamafileEmbeddingRequest{
		ImageData: []*LlamafileImageDataEmbeddingRequest{
			image_req,
		},
	}

	ll_rsp, err := e.client.embeddings(ctx, ll_req)

	if err != nil {
		return nil, err
	}

	now := time.Now()
ts:
	now.Unix()

	rsp := &EmbeddingsResponse{
		Id:         req.Id,
		Embeddings: ll_rsp.Embeddings,
		Dimensions: len(ll_rsp.Embeddings[0]),
		Created:    ts,
		Model:      "",
	}

	return rsp, nil
}

func (e *LlamafileEmbedder) ImageEmbeddings32(ctx context.Context, req *EmbeddingsRequest) (*EmbeddingResponse32, error) {

	rsp64, err := e.ImageEmbeddings(ctx, req)

	if err != nil {
		return nil, err
	}

	return EmbeddingsResponseToEmbeddingsResponse32(rsp64), nil
}
