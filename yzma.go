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
	"github.com/hybridgroup/yzma/pkg/mtmd"
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
	// ngpu_layers stores the GPU offload layer count
	ngpu_layers int
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
	// enable verbose llama.cpp logging
	llama_verbose bool
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
//   - **ngl**            - GPU offload layer count. Default is -1.
//   - **processor**      – Target CPU instruction set for the downloaded yzma binary (e.g. `avx`, `neon`).  If omitted, the library will
//     automatically select an appropriate processor.
//   - **version**        – The yzma release version to download.  The  default is `"v0.3.0"`.  The value should be a full git tag (e.g.
//     `v0.3.0`), not an empty string.
//   - **model-root**     – Directory where the llama model files are stored.  If omitted, a subdirectory `models` of the library root
//     (`lib_path`) is used.
//   - **llama-verbose**  - Enable verbose llama.cpp logging.
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

		// "A tagged release of yzma installs its own llama.cpp release by default, so yzma install
		// without the -version flag gets the version in this table. Use -version latest to get the
		// most recent nightly build instead. A build from the main branch always uses the most recent
		// nightly build." - https://github.com/hybridgroup/yzma#required-versions-of-llamacpp

		version := ""

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

	llama_verbose := false

	if q.Has("llama-verbose") {

		v, err := strconv.ParseBool(q.Get("llama-verbose"))

		if err != nil {
			return nil, fmt.Errorf("Failed to parse ?llama-verbose query parameter, %w", err)
		}

		llama_verbose = v
	}

	if !llama_verbose {
		llama.LogSet(llama.LogSilent())
	}

	llama.Init()

	// We call llama.Close in Close() below

	context_sz := 0
	batch_sz := 2048
	ngpu_layers := -1
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

	if q.Has("ngl") {

		v, err := strconv.Atoi(q.Get("ngl"))

		if err != nil {

			return nil, fmt.Errorf("Invalid ?ngl parameter, %w", err)
		}

		ngpu_layers = v
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
		precision:     precision,
		scheme:        u.Scheme,
		context_sz:    context_sz,
		batch_sz:      batch_sz,
		ngpu_layers:   ngpu_layers,
		pooling_type:  pooling_type,
		model_root:    model_root,
		tmp_dir:       tmp_dir,
		lib_path:      lib_path,
		llama_verbose: llama_verbose,
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

	return e.generateResponse(req, vec)
}

// ImageEmbeddings derives embeddings for the image contained in the request.
// It uses the correct yzma/mtmd pipeline variables and function signatures.
func (e *YzmaEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	err := mtmd.Load(e.lib_path)

	if err != nil {
		return nil, fmt.Errorf("Failed to load mtmd library from %s, %w", e.lib_path, err)
	}

	if !e.llama_verbose {
		mtmd.LogSet(llama.LogSilent())
	}

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

	// 4. Resolve the location framework matching the projector binary

	u, err := url.Parse(req.Model)

	if err != nil {
		return nil, err
	}

	var mmproj_uri string

	switch u.Scheme {
	case "http", "https":

		model_root := filepath.Dir(u.Path)
		model_fname := filepath.Base(u.Path)
		mmproj_fname := "mmproj-" + model_fname

		u.Path = filepath.Join(model_root, mmproj_fname)
		mmproj_uri = u.String()

	case "file":

		model_root := filepath.Dir(req.Model)
		model_fname := filepath.Base(req.Model)
		mmproj_fname := "mmproj-" + model_fname

		mmproj_uri = filepath.Join(model_root, mmproj_fname)
	default:
		return nil, fmt.Errorf("Unsupported model scheme")
	}

	mmproj_path, err := e.fetchModel(ctx, mmproj_uri)

	if err != nil {
		return nil, fmt.Errorf("Failed to fetch multimodal model %s, %w", mmproj_uri, err)
	}

	// 5. Instantiate multimodal parsing layers

	mtmd_params := mtmd.ContextParamsDefault()

	mtmd_ctx, err := mtmd.InitFromFile(mmproj_path, model, mtmd_params)

	if err != nil {
		return nil, fmt.Errorf("Failed to initialize multimodal context, %w", err)
	}

	defer mtmd.Free(mtmd_ctx)

	tmp_im, err := os.CreateTemp("", "yzma-img-*.jpg")

	if err != nil {
		return nil, fmt.Errorf("Failed to create temporary image cache file, %w", err)
	}

	defer os.Remove(tmp_im.Name())
	defer tmp_im.Close()

	_, err = tmp_im.Write(req.Body)

	if err != nil {
		return nil, fmt.Errorf("Failed to write image data to temporary file, %w", err)
	}

	err = tmp_im.Close()

	if err != nil {
		return nil, fmt.Errorf("Failed to close temporary file after writing, %w", err)
	}

	bitmap := mtmd.BitmapInitFromFile(mtmd_ctx, tmp_im.Name(), false, mtmd.InitOptDefault())

	if bitmap.Bitmap == 0 {
		return nil, fmt.Errorf("Failed to construct image layout matrix from file")
	}

	defer mtmd.BitmapFree(bitmap.Bitmap)

	// 8. Allocate target output allocation tracking layout chunks

	output_chunks := mtmd.InputChunksInit()
	defer mtmd.InputChunksFree(output_chunks)

	// 9. Prepare standard token marker structure mapping the image boundaries

	text_marker := mtmd.DefaultMarker()
	input_text := mtmd.NewInputText(text_marker, true, true)

	// 10. Format and ingest the text/image markers into the target processing chunk

	tokenize_rsp := mtmd.Tokenize(mtmd_ctx, output_chunks, input_text, []mtmd.Bitmap{bitmap.Bitmap})

	if tokenize_rsp != 0 {
		return nil, fmt.Errorf("Multimodal Tokenize operations returned error status: %d", tokenize_rsp)
	}

	// 11. Run evaluation matrices using explicit sequence tracking arguments

	var last_pos llama.Pos

	eval_rsp := mtmd.HelperEvalChunks(mtmd_ctx, emb_ctx, output_chunks, 0, 0, int32(model_ctx.NBatch), true, &last_pos)

	if eval_rsp != 0 {
		return nil, fmt.Errorf("Multimodal image token layer evaluation failed with status, %d", eval_rsp)
	}

	// 12. Read vector mapping state layers out from your execution block

	nEmbd := llama.ModelNEmbd(model)

	vec, err := llama.GetEmbeddingsSeq(emb_ctx, 0, nEmbd)

	if err != nil {
		return nil, fmt.Errorf("Failed to extract multimodal embeddings from context, %w", err)
	}

	return e.generateResponse(req, vec)
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

	slog.Debug("Get model", "model", req.Model)

	model_path, err := e.fetchModel(ctx, req.Model)

	if err != nil {
		return 0, err
	}

	params := llama.ModelDefaultParams()
	params.NGpuLayers = int32(e.ngpu_layers)

	slog.Debug("Load model from path", "path", model_path)

	model, err := llama.ModelLoadFromFile(model_path, params)

	if err != nil {
		return 0, fmt.Errorf("Failed to load model, %w", err)
	}

	if model == 0 {
		return 0, fmt.Errorf("Unable to load model")
	}

	return model, nil
}

func (e *YzmaEmbedder[T]) fetchModel(ctx context.Context, model_uri string) (string, error) {

	model_fname := filepath.Base(model_uri)
	model_path := filepath.Join(e.model_root, model_fname)

	slog.Debug("Fetch model", "uri", model_uri, "path", model_path)

	_, err := os.Stat(model_path)

	if err != nil {

		if !os.IsNotExist(err) {
			return "", err
		}

		u, err := url.Parse(model_uri)

		if err != nil {
			return "", fmt.Errorf("Failed to parse model URI, %w", err)
		}

		switch u.Scheme {
		case "http", "https":

			// To do: Better progress indicator...

			slog.Debug("Download model", "model", model_uri, "target", e.model_root)
			err := download.GetModelWithContext(ctx, model_uri, e.model_root, nil)

			if err != nil {
				return "", fmt.Errorf("Failed to retrieve model, %w", err)
			}

		case "file":
			model_path = u.Path
		default:
			return "", fmt.Errorf("Unsupported model scheme")
		}

		_, err = os.Stat(model_path)

		if err != nil {
			return "", fmt.Errorf("Model path does not exist, %w", err)
		}
	}

	slog.Debug("Got model", "path", model_path)
	return model_path, nil
}

// generateResponse handles L2 normalization, model path extraction, and response formatting
// consistently for both text and image vectors.
func (e *YzmaEmbedder[T]) generateResponse(req *EmbeddingsRequest, vec []float32) (EmbeddingsResponse[T], error) {

	var sum float64

	for _, v := range vec {
		sum += float64(v * v)
	}

	if sum > 0 {
		sum = math.Sqrt(sum)
		norm := float32(1.0 / sum)

		for i := range vec {
			vec[i] = vec[i] * norm
		}
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
		rsp.CommonEmbeddings = toFloat64Slice[T](AsFloat64(vec))
	default:
		rsp.CommonEmbeddings = toFloat32Slice[T](vec)
	}

	return rsp, nil
}
