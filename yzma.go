package embeddings

import (
	"context"
	"fmt"
	"log/slog"
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

// YzmaEmbedder implements the `Embedder` interface using the `hybridgroup/yzma` package to derive embeddings.
type YzmaEmbedder[T Float] struct {
	Embedder[T]
	precision    string
	scheme       string
	context_sz   int
	batch_sz     int
	pooling      string
	pooling_type llama.PoolingType
	model_root   string
	tmp_dir      string
	lib_path     string
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
		return nil, fmt.Errorf("Failed to parse URI, %w", err)
	}

	q := u.Query()

	precision := "float32"

	switch {
	case strings.HasSuffix(u.Scheme, "64"):
		precision = "%s#as-float64"
	}

	lib_path := u.Path
	tmp_dir := ""

	if lib_path == "" {
		lib_path = os.Getenv("YZMA_LIB")
	}

	if lib_path == "" {

		dir, err := os.MkdirTemp("", "yzma")

		if err != nil {
			return nil, fmt.Errorf("Failed to create tmp llama root, %w", err)
		}

		lib_path = dir
		tmp_dir = dir

	} else {

		err := os.MkdirAll(lib_path, 0750)

		if err != nil {
			return nil, fmt.Errorf("Failed to create llama root, %w", err)
		}
	}

	model_root := filepath.Join(lib_path, "models")

	if q.Has("model-root") && q.Get("model-root") != "" {
		model_root = q.Get("model-root")
	}

	err = os.MkdirAll(model_root, 0750)

	if err != nil {
		return nil, fmt.Errorf("Failed to create model root, %w", err)
	}

	if !download.AlreadyInstalled(lib_path) {

		// https://pkg.go.dev/github.com/hybridgroup/yzma@v1.25.0/pkg/download#Install

		arch := runtime.GOARCH
		os := runtime.GOOS
		proc := q.Get("processor")
		version := q.Get("version")

		// To do: Better progress indicator...

		slog.Debug("Download llama", "arch", arch, "os", os, "proc", proc, "version", version)
		err := download.GetWithContext(ctx, arch, os, proc, version, lib_path, nil)

		if err != nil {
			return nil, fmt.Errorf("Failed to download llama, %w", err)
		}
	}

	err = llama.Load(lib_path)

	if err != nil {
		return nil, fmt.Errorf("Failed to load %s, %w", lib_path, err)
	}

	llama.LogSet(llama.LogSilent())
	llama.Init()

	e := &YzmaEmbedder[T]{
		precision:    precision,
		scheme:       u.Scheme,
		context_sz:   0,
		batch_sz:     2048,
		pooling:      "mean",
		pooling_type: llama.PoolingTypeMean,
		model_root:   model_root,
		tmp_dir:      tmp_dir,
		lib_path:     lib_path,
	}

	return e, nil
}

func (e *YzmaEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	// https://github.com/hybridgroup/yzma/blob/main/examples/embeddings/main.go

	model, err := e.getModelForRequest(ctx, req)

	if err != nil {
		return nil, fmt.Errorf("Failed to instantiate model, %w", err)
	}

	defer llama.ModelFree(model)

	model_ctx := llama.ContextDefaultParams()
	model_ctx.NCtx = uint32(e.context_sz)
	model_ctx.NBatch = uint32(e.batch_sz)
	model_ctx.PoolingType = e.pooling_type
	model_ctx.Embeddings = 1

	emb_ctx, err := llama.InitFromModel(model, model_ctx)

	if err != nil {
		return nil, fmt.Errorf("Unable to initialize context from model, %w", err)
	}

	defer llama.Free(emb_ctx)

	vocab := llama.ModelGetVocab(model)
	tokens := llama.Tokenize(vocab, string(req.Body), true, true)

	batch := llama.BatchGetOne(tokens)

	ret, err := llama.Decode(emb_ctx, batch)

	if err != nil {
		return nil, fmt.Errorf("Failed to decode token btach, %w", err)
	}

	if ret != 0 {
		return nil, fmt.Errorf("Decode operation returned non-zero response, %d", ret)
	}

	nEmbd := llama.ModelNEmbd(model)

	vec, err := llama.GetEmbeddingsSeq(emb_ctx, 0, nEmbd)

	if err != nil {
		return nil, fmt.Errorf("Failed to derive embeddings, %w", err)
	}

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

	// To do: Try to capture source/publisher for the model URI

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

	if e.tmp_dir != "" {
		os.RemoveAll(e.tmp_dir)
	}

	return nil
}

func (e *YzmaEmbedder[T]) getModelForRequest(ctx context.Context, req *EmbeddingsRequest) (llama.Model, error) {

	model_fname := filepath.Base(req.Model)
	model_path := filepath.Join(e.model_root, model_fname)

	_, err := os.Stat(model_path)

	if err != nil {

		if !os.IsNotExist(err) {
			return 0, err
		}

		u, err := url.Parse(req.Model)

		if err != nil {
			return 0, fmt.Errorf("Failed to parse model URI, %w", err)
		}

		switch u.Scheme {
		case "http", "https":

			// To do: Better progress indicator...

			slog.Debug("Download model", "model", req.Model)
			err := download.GetModelWithContext(ctx, req.Model, e.model_root, nil)

			if err != nil {
				return 0, fmt.Errorf("Failed to retrieve model, %w", err)
			}

		case "file":
			model_path = u.Path
		default:
			return 0, fmt.Errorf("Unsupported model scheme")
		}

		_, err = os.Stat(model_path)

		if err != nil {
			return 0, fmt.Errorf("Model path does not exist, %w", err)
		}
	}

	params := llama.ModelDefaultParams()

	model, err := llama.ModelLoadFromFile(model_path, params)

	if err != nil {
		return 0, fmt.Errorf("Failed to load model, %w", err)
	}

	if model == 0 {
		return 0, fmt.Errorf("Unable to load model")
	}

	return model, nil
}
