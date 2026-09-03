package embeddings

import (
	"context"
	"fmt"
	"math"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
)

// YzmaEmbedder implements the `Embedder` interface using an Yzma API endpoint to derive embeddings.
type YzmaEmbedder[T Float] struct {
	Embedder[T]
	precision    string
	scheme       string
	context_sz   int
	batch_sz     int
	pooling      string
	pooling_type llama.PoolingType
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

	if !download.AlreadyInstalled(lib_path) {

		// https://pkg.go.dev/github.com/hybridgroup/yzma@v1.25.0/pkg/download#Install

		arch := runtime.GOARCH
		os := runtime.GOOS
		proc := ""
		version := "latest"

		err := download.GetWithContext(ctx, arch, os, proc, version, lib_path, nil)

		if err != nil {
			return nil, err
		}
	}

	err = llama.Load(lib_path)

	if err != nil {
		return nil, err
	}

	llama.Init()

	e := &YzmaEmbedder[T]{
		precision:    precision,
		scheme:       u.Scheme,
		context_sz:   0,
		batch_sz:     2048,
		pooling:      "mean",
		pooling_type: llama.PoolingTypeUnspecified,
	}

	return e, nil
}

func (e *YzmaEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	// https://github.com/hybridgroup/yzma/blob/main/examples/embeddings/main.go

	model, err := e.getModelForRequest(ctx, req)

	if err != nil {
		return nil, err
	}

	if model == 0 {
		return nil, fmt.Errorf("Unable to load model")
	}

	defer llama.ModelFree(model)

	model_ctx := llama.ContextDefaultParams()
	model_ctx.NCtx = uint32(e.context_sz)
	model_ctx.NBatch = uint32(e.batch_sz)
	model_ctx.PoolingType = e.pooling_type
	model_ctx.Embeddings = 1

	lctx, err := llama.InitFromModel(model, model_ctx)

	if err != nil {
		return nil, fmt.Errorf("Unable to initialize context from model, %w", err)
	}

	defer llama.Free(lctx)

	vocab := llama.ModelGetVocab(model)
	tokens := llama.Tokenize(vocab, string(req.Body), true, true)

	batch := llama.BatchGetOne(tokens)

	ret, err := llama.Decode(lctx, batch)

	if err != nil {
		return nil, fmt.Errorf("decode failed: %w", err)
	}

	if ret != 0 {
		return nil, fmt.Errorf("decode returned non-zero: %d", ret)
	}

	nEmbd := llama.ModelNEmbd(model)

	vec, err := llama.GetEmbeddingsSeq(lctx, 0, nEmbd)

	if err != nil {
		return nil, fmt.Errorf("unable to get embeddings: %v", err)
	}

	// normalize embeddings
	var sum float64

	for _, v := range vec {
		sum += float64(v * v)
	}

	sum = math.Sqrt(sum)
	norm := float32(1.0 / sum)

	e32 := make([]float32, len(vec))

	for i, v := range vec {
		e32[i] = v * norm
	}

	now := time.Now()
	ts := now.Unix()

	model_name := filepath.Base(req.Model)

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonModel:     model_name,
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

func (e *YzmaEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {
	return nil, NotImplemented
}

func (e *YzmaEmbedder[T]) Close() error {
	llama.Close()
	return nil
}

func (e *YzmaEmbedder[T]) getModelForRequest(ctx context.Context, req *EmbeddingsRequest) (llama.Model, error) {

	var model_root string
	var model_path string

	_, err := os.Stat(model_path)

	if err != nil {

		if !os.IsNotExist(err) {
			return 0, err
		}

		err := download.GetModelWithContext(ctx, req.Model, model_root, nil)

		if err != nil {
			return 0, err
		}
	}

	params := llama.ModelDefaultParams()

	model, err := llama.ModelLoadFromFile(model_path, params)

	if err != nil {
		return 0, err
	}

	return model, nil
}
