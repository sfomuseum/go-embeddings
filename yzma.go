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
	"strconv"
	"strings"
	"time"

	"github.com/hybridgroup/yzma/pkg/download"
	"github.com/hybridgroup/yzma/pkg/llama"
)

// YzmaEmbedder implements the `Embedder` interface using the `hybridgroup/yzma` package to derive embeddings.
type YzmaEmbedder[T Float] struct {
	Embedder[T]
	// precision is the string representation of the precision used for
	// the embeddings.  It is either `"float32"` or a string ending in
	// `"64"` for `float64`.
	precision string
	// scheme is the URI scheme that was used to construct the embedder.
	// It is typically `"yzma"` or `"yzma64"`.
	scheme string
	// context_sz is the maximum number of tokens in a llama context.
	context_sz int
	// batch_sz is the maximum number of tokens processed per batch.
	batch_sz int
	// pooling_type is the `llama.PoolingType` corresponding to `pooling`.
	pooling_type llama.PoolingType
	// model_root is the directory where the llama model files are stored.
	model_root string
	// tmp_dir is a temporary directory created for downloading the yzma binary; it is removed when the embedder is closed.
	tmp_dir string
	// lib_path is the path to the yzma library binaries.
	lib_path string
}

func init() {
	ctx := context.Background()

	RegisterEmbedder[float32](ctx, "yzma", NewYzmaEmbedder[float32])
	RegisterEmbedder[float32](ctx, "yzma32", NewYzmaEmbedder[float32])
	RegisterEmbedder[float64](ctx, "yzma64", NewYzmaEmbedder[float64])
}

// NewYzmaEmbedder creates a new YzmaEmbedder based on the provided URI.
//
// The URI may include query parameters that configure the embedder.  The function downloads the yzma binary if
// it is not already present and ensures that the model root directory exists.  It returns an error if the URI
// is malformed, the binary cannot be downloaded, or the configuration parameters are invalid. YzmaEmbedder is
// instantiated by 'uri' which is expected to take the form of:
// The URI may contain query parameters that configure the embedder.  The
// supported parameters are:
//
//	yzma://{PATH_TO_YZMA_LIB}?{QUERY_PARAMETERS)
//
// Where `{PATH_TO_YZMA}` is the path the yzma-specific llama.cpp build. If empty then the code will check for
// a `YZMA_LIB` environment variable. If the path remains empty then a temporary directory will be created and
// an device-specific build will be downloaded. Valid query parameters are:
//
//   - **context-size**   – Maximum number of tokens in a llama context. If omitted, the default of 0 (use library default) is used.
//   - **batch-size**     – Maximum number of tokens processed per batch. If omitted, the default of 0 (use library default) is used.
//   - **pooling**        – Pooling strategy used to aggregate token embeddings.  Accepted values are the same strings that the
//     `llama.PoolingType` type understands (e.g. `"mean"`, `"sum"`).  The default is `"mean"`.
//   - **processor**      – Target CPU instruction set for the downloaded yzma binary (e.g. `avx`, `neon`).  If omitted, the library will
//     automatically select an appropriate processor.
//   - **version**        – The yzma release version to download.  The  default is `"v0.3.0"`.  The value should be a full git tag (e.g.
//     `v0.3.0`), not an empty string.
//   - **model-root**     – Directory where the llama model files are stored.  If omitted, a subdirectory `models` of the library root
//     (`lib_path`) is used.
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

		if proc == "" && os == "darwin" {
			proc = "metal"
		}

		// Don't be surprised if "latest" is out of sync with the versioned release of yzma.
		// A version number of "" does not seem to work as advertised so the safest thing is
		// to reference an explcit version number specific to the yzma release:
		// https://github.com/hybridgroup/yzma#required-versions-of-llamacpp

		version := "v0.3.0"

		if q.Has("version") {
			version = q.Get("version")
		}

		slog.Debug("Download llama", "arch", arch, "os", os, "proc", proc, "version", version)

		err := download.GetWithContext(ctx, arch, os, proc, version, lib_path, download.DefaultProgressTracker())

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

	context_sz := 0
	batch_sz := 0
	pooling := "mean"

	if q.Has("context-size") {

		v, err := strconv.Atoi(q.Get("context-size"))

		if err != nil {
			return nil, fmt.Errorf("Invalid ?context-size parameter, %w", err)
		}

		context_sz = v
	}

	if q.Has("batch-size") {

		v, err := strconv.Atoi(q.Get("batch-size"))

		if err != nil {
			return nil, fmt.Errorf("Invalid ?batch-size parameter, %w", err)
		}

		batch_sz = v
	}

	if q.Has("pooling") {
		pooling = q.Get("pooling")
	}

	var pooling_type llama.PoolingType

	switch pooling {
	case "mean":
		pooling_type = llama.PoolingTypeMean
	case "cls":
		pooling_type = llama.PoolingTypeCLS
	case "rank":
		pooling_type = llama.PoolingTypeRank
	case "last":
		pooling_type = llama.PoolingTypeLast
	case "none":
		pooling_type = llama.PoolingTypeNone
	default:
		pooling_type = llama.PoolingTypeUnspecified
	}

	e := &YzmaEmbedder[T]{
		precision:    precision,
		scheme:       u.Scheme,
		context_sz:   context_sz,
		batch_sz:     batch_sz,
		pooling_type: pooling_type,
		model_root:   model_root,
		tmp_dir:      tmp_dir,
		lib_path:     lib_path,
	}

	return e, nil
}

// TextEmbeddings derives embeddings for the text contained in the request.
// It uses the yzma backend to tokenize the input, create a llama context, and retrieve the embedding vector.  The vector is normalised to unit
// length and returned in the `EmbeddingsResponse`.  The response includes the model name, the timestamp of creation, and the precision used.
func (e *YzmaEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	// https://github.com/hybridgroup/yzma/blob/main/examples/embeddings/main.go

	model, err := e.getModelForRequest(ctx, req)

	if err != nil {
		return nil, fmt.Errorf("Failed to instantiate model, %w", err)
	}

	defer llama.ModelFree(model)

	body := string(req.Body)

	body_sz := len(body)
	batch_sz := e.batch_sz

	if batch_sz < body_sz {
		new_sz := int(float64(body_sz) / 3.0)

		if new_sz > batch_sz {
			batch_sz = new_sz
			slog.Debug("Reassign batch size based on input", "old", e.batch_sz, "new", batch_sz)
		}
	}

	model_ctx := llama.ContextDefaultParams()
	model_ctx.NCtx = uint32(e.context_sz)
	model_ctx.NBatch = uint32(batch_sz)
	model_ctx.PoolingType = e.pooling_type
	model_ctx.Embeddings = 1

	emb_ctx, err := llama.InitFromModel(model, model_ctx)

	if err != nil {
		return nil, fmt.Errorf("Unable to initialize context from model, %w", err)
	}

	defer llama.Free(emb_ctx)

	vocab := llama.ModelGetVocab(model)

	tokens := llama.Tokenize(vocab, body, true, true)

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

// ImageEmbeddings is not implemented for the yzma backend. It returns the `NotImplemented` error to
// indicate that image embeddings are unsupported.
func (e *YzmaEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {
	return nil, NotImplemented
}

// Close releases resources held by the yzma backend.  It also removes any temporary directories that were created for downloading the yzma binary.
func (e *YzmaEmbedder[T]) Close() error {

	llama.Close()

	if e.tmp_dir != "" {

		slog.Debug("Remove tmp dir", "path", e.tmp_dir)
		err := os.RemoveAll(e.tmp_dir)

		if err != nil {
			slog.Error("Failed to remove tmp dir", "path", e.tmp_dir, "error", err)
		}
	}

	return nil
}

// getModelForRequest ensures that the model referenced in the request exists
// locally.  If the model is not present, it is downloaded (for HTTP/HTTPS
// URLs) or resolved from a file path.  The function then loads the model
// using `llama.ModelLoadFromFile` and returns the handle.
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

			slog.Debug("Download model", "model", req.Model, "target", e.model_root)
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
