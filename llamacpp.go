package embeddings

// llama serve -hf ggml-org/embeddinggemma-2-GGUF:Q8_0 --embeddings

import (
	"context"
	"fmt"
	"net/url"
	"strings"
	"time"
)

// LlamaCppEmbedder implements the `Embedder` interface using an LlamaCpp API endpoint to derive embeddings.
type LlamaCppEmbedder[T Float] struct {
	Embedder[T]
	client    *llamaCppClient
	model     string
	precision string
}

func init() {
	ctx := context.Background()

	schemes := []string{
		"llamacpp",
	}

	for _, s := range schemes {

		s32 := fmt.Sprintf("%s32", s)
		s64 := fmt.Sprintf("%s64", s)

		MustRegisterEmbedder[float32](ctx, s, NewLlamaCppEmbedder[float32])
		MustRegisterEmbedder[float32](ctx, s32, NewLlamaCppEmbedder[float32])
		MustRegisterEmbedder[float64](ctx, s64, NewLlamaCppEmbedder[float64])
	}
}

func NewLlamaCppEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	q := u.Query()

	client_uri := "http://localhost:9931"

	if q.Has("client-uri") {
		client_uri = q.Get("client-uri")
	}

	cl, err := newLlamaCppClient(ctx, client_uri)

	if err != nil {
		return nil, err
	}

	model := q.Get("model")

	precision := "float32"

	if strings.HasSuffix(u.Scheme, "64") {
		precision = fmt.Sprintf("%s#as-float%d", precision, 64)
	}

	e := &LlamaCppEmbedder[T]{
		client:    cl,
		model:     model,
		precision: precision,
	}

	return e, nil
}

func (e *LlamaCppEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {
	return e.getEmbeddings(ctx, "text", req)
}

func (e *LlamaCppEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {
	return e.getEmbeddings(ctx, "image", req)
}

func (e *LlamaCppEmbedder[T]) getEmbeddings(ctx context.Context, embeddings_type string, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	model := e.model

	if req.Model != "" {
		model = req.Model
	}

	// Note: Munging model in anticipation of a time when llama.cpp allows Ollama-style model toggling

	cl_rsp, err := e.client.embeddings(ctx, model, embeddings_type, req.Body)

	if err != nil {
		return nil, err
	}

	if len(cl_rsp.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("Model returned no embeddings")
	}

	e32 := cl_rsp.Data[0].Embedding

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonModel:     cl_rsp.Model,
		CommonCreated:   ts,
		CommonPrecision: e.precision,
	}

	switch {
	case strings.HasSuffix(e.precision, "64"):
		rsp.CommonEmbeddings = toFloat64Slice[T](AsFloat64(e32))
	default:
		rsp.CommonEmbeddings = toFloat32Slice[T](e32)
	}

	return rsp, nil
}

func (e *LlamaCppEmbedder[T]) Close() error {
	return nil
}
