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
	"sync"
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
	// the maximum number of pixels an image can have before it is automatically resized
	llama_max_pixels int
	// enable verbose llama.cpp logging
	llama_verbose bool
	// max_slots is the maximum number of llama contexts (concurrent requests) per model.
	max_slots int
	// models caches loaded models, keyed by the model URI in the request.
	models    map[string]*yzmaModel
	models_mu sync.Mutex
	// mtmd_once ensures the mtmd library is loaded only once.
	mtmd_once sync.Once
	mtmd_err  error
}

// yzmaModel is a loaded model plus a pool of reusable slots. Weights are shared; each slot has its own
// llama context (and lazily created multimodal context) because those are not thread-safe.
type yzmaModel struct {
	model llama.Model
	// idle holds slots that are not in use.
	idle chan *yzmaSlot
	// mu guards `created`.
	mu      sync.Mutex
	created int
	max     int
}

type yzmaSlot struct {
	ctx      llama.Context
	mtmd_ctx mtmd.Context
}

func (s *yzmaSlot) free() {

	if s.mtmd_ctx != 0 {
		mtmd.Free(s.mtmd_ctx)
		s.mtmd_ctx = 0
	}

	if s.ctx != 0 {
		llama.Free(s.ctx)
		s.ctx = 0
	}
}

// reset clears any state left in the context's memory by a previous request.
func (s *yzmaSlot) reset() {

	mem, err := llama.GetMemory(s.ctx)

	if err != nil || mem == 0 {
		// Non-causal embedding models may have no KV cache to clear.
		return
	}

	// Metadata only; zeroing the (potentially huge) data buffers is needlessly slow.
	err = llama.MemoryClear(mem, false)

	if err != nil {
		slog.Debug("Failed to clear context memory", "error", err)
	}
}

func init() {
	ctx := context.Background()

	MustRegisterEmbedder[float32](ctx, "yzma", NewYzmaEmbedder[float32])
	MustRegisterEmbedder[float32](ctx, "yzma32", NewYzmaEmbedder[float32])
	MustRegisterEmbedder[float64](ctx, "yzma64", NewYzmaEmbedder[float64])
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
//   - **version**        – The yzma release version to use. The  default is whatever is associated with the current version of
//     `hybridgroup/yzma`.  The value should be a full git tag (e.g. `v0.3.0`).
//   - **model-root**     – Directory where the llama model files are stored.  If omitted, a subdirectory `models` of the library root
//     (`lib_path`) is used.
//   - **llama-max-pixels** - The maximum number of pixels in an image to be embedded before it is automatically resized. Default is 400, 000.
//   - **llama-verbose**  - Enable verbose llama.cpp logging.
//   - **slots**         - Maximum number of concurrent llama contexts per model. Default is 4.
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

	// https://github.com/hybridgroup/yzma#required-versions-of-llamacpp
	// For the time being this will need to be manually kept in sync with
	// the version of hybridgroup/yzma in go.mod. The only way to derive
	// this programmatically is _after_ llama itself is loaded.
	version := "v0.9.0"

	if q.Has("version") {

		v := q.Get("version")

		err := download.VersionIsValid(v)

		if err != nil {
			return nil, fmt.Errorf("Invalid version, %w", err)
		}

		slog.Debug("Assign yzma version", "version", v)
		version = v
	}

	var model_root string

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

		model_root = filepath.Join(lib_path, "models")

	} else {

		// Ensure model root is outside of any versioned yzma folder

		model_root = filepath.Join(lib_path, "models")

		lib_path = filepath.Join(lib_path, version)
		slog.Debug("Update lib path with version", "version", q.Get("version"), "path", lib_path)

		err := os.MkdirAll(lib_path, 0750)

		if err != nil {
			slog.Error("Failed to create llama root", "path", lib_path, "error", err)
			return nil, fmt.Errorf("Failed to create llama root, %w", err)
		}
	}

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

		slog.Debug("Download llama", "arch", arch, "os", os, "proc", proc, "version", version)

		err := download.GetWithContext(ctx, arch, os, proc, version, lib_path, download.DefaultProgressTracker())

		if err != nil {
			return nil, fmt.Errorf("Failed to download llama, %w", err)
		}
	}

	slog.Debug("Load llama", "path", lib_path)

	err = llama.Load(lib_path)

	if err != nil {
		return nil, fmt.Errorf("Failed to load %s, %w", lib_path, err)
	}

	slog.Debug("llama initialized", "version", llama.Version())

	context_sz := 0
	batch_sz := 2048
	ngpu_layers := -1
	pooling := "mean"
	max_pixels := 400000
	llama_verbose := false
	max_slots := 4

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

	if q.Has("context-size") {

		v, err := strconv.Atoi(q.Get("context-size"))

		if err != nil {
			return nil, fmt.Errorf("Invalid ?context-size parameter, %w", err)
		}

		slog.Debug("Assign context size", "value", v)
		context_sz = v
	}

	if q.Has("batch-size") {

		v, err := strconv.Atoi(q.Get("batch-size"))

		if err != nil {
			return nil, fmt.Errorf("Invalid ?batch-size parameter, %w", err)
		}

		slog.Debug("Assign batch size", "value", v)
		batch_sz = v
	}

	if q.Has("ngl") {

		v, err := strconv.Atoi(q.Get("ngl"))

		if err != nil {

			return nil, fmt.Errorf("Invalid ?ngl parameter, %w", err)
		}

		slog.Debug("Assign ngpu layers", "value", v)
		ngpu_layers = v
	}

	if q.Has("slots") {

		v, err := strconv.Atoi(q.Get("slots"))

		if err != nil || v < 1 {
			return nil, fmt.Errorf("Invalid ?slots parameter, %w", err)
		}

		max_slots = v
	}

	if q.Has("pooling") {
		pooling = q.Get("pooling")

		slog.Debug("Assign pooling", "value", pooling)
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

	if q.Has("llama-max-pixels") {

		v, err := strconv.Atoi(q.Get("llama-max-pixels"))

		if err != nil {
			return nil, fmt.Errorf("Failed to parse ?llama-max-pixels query parameter, %w", err)
		}

		max_pixels = v
	}

	slog.Debug("Create embedder", "precision", precision, "context size", context_sz, "batch size", batch_sz, "pooling", pooling_type)

	e := &YzmaEmbedder[T]{
		precision:        precision,
		scheme:           u.Scheme,
		context_sz:       context_sz,
		batch_sz:         batch_sz,
		ngpu_layers:      ngpu_layers,
		pooling_type:     pooling_type,
		model_root:       model_root,
		tmp_dir:          tmp_dir,
		lib_path:         lib_path,
		llama_verbose:    llama_verbose,
		llama_max_pixels: max_pixels,
		models:           make(map[string]*yzmaModel),
		max_slots:        max_slots,
	}

	return e, nil
}

// TextEmbeddings derives embeddings for the text contained in the request.
// It uses the yzma backend to tokenize the input, create a llama context, and retrieve the embedding vector.  The vector is normalised to unit
// length and returned in the `EmbeddingsResponse`.  The response includes the model name, the timestamp of creation, and the precision used.
func (e *YzmaEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	// https://github.com/hybridgroup/yzma/blob/main/examples/embeddings/main.go

	entry, slot, err := e.acquireSlot(ctx, req)

	if err != nil {
		return nil, fmt.Errorf("Failed to instantiate model, %w", err)
	}

	defer entry.release(slot)

	model := entry.model
	emb_ctx := slot.ctx

	slot.reset()

	body := string(req.Body)

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

	// Use the output size: it can differ from n_embd (e.g. embeddinggemma-2: 768 vs 512).
	nEmbd := llama.ModelNEmbdOut(model)

	vec, err := llama.GetEmbeddingsSeq(emb_ctx, 0, nEmbd)

	if err != nil {
		return nil, fmt.Errorf("Failed to derive embeddings, %w", err)
	}

	return e.generateResponse(req, vec)
}

// ImageEmbeddings derives embeddings for the image contained in the request.
// It uses the correct yzma/mtmd pipeline variables and function signatures.
func (e *YzmaEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	entry, slot, err := e.acquireSlot(ctx, req)

	if err != nil {
		return nil, fmt.Errorf("Failed to instantiate model, %w", err)
	}

	defer entry.release(slot)

	model := entry.model
	emb_ctx := slot.ctx

	mtmd_ctx, err := e.getMtmdContext(ctx, req, entry, slot)

	if err != nil {
		return nil, err
	}

	slot.reset()

	im_body, err := ResizeIfOversized(req.Body, e.llama_max_pixels)

	if err != nil {
		return nil, err
	}

	if len(im_body) == 0 {
		return nil, fmt.Errorf("Empty image body")
	}

	// Decode directly from memory rather than round-tripping through a temporary file.
	bitmap := mtmd.BitmapInitFromBuf(mtmd_ctx, &im_body[0], uint64(len(im_body)), false, mtmd.InitOptDefault())

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

	// 11. Like llama-server for non-causal embedding models, put the text tokens (BOS) and the
	// projected image embeddings in a single batch so the pooled vector covers all of them.

	batch, err := llama.BatchExtInit(emb_ctx)

	if err != nil {
		return nil, fmt.Errorf("Failed to create batch, %w", err)
	}

	defer llama.BatchExtFree(batch)

	n_embd_inp := int(llama.ModelNEmbdInp(model))

	var pos llama.Pos

	add := func(idx int32, err error) error {

		if err != nil {
			return err
		}

		err = llama.BatchExtSetPos(batch, idx, pos)

		if err != nil {
			return err
		}

		pos++
		return llama.BatchExtSetOutputEmbd(batch, idx, true)
	}

	for i := uint64(0); i < mtmd.InputChunksSize(output_chunks); i++ {

		chunk := mtmd.InputChunksGet(output_chunks, i)

		switch mtmd.InputChunkGetType(chunk) {
		case mtmd.InputChunkTypeText:

			for _, token := range mtmd.InputChunkGetTokensText(chunk) {

				err := add(llama.BatchExtAddToken(batch, 0, token))

				if err != nil {
					return nil, fmt.Errorf("Failed to add text token to batch, %w", err)
				}
			}

		default:

			err := mtmd.EncodeChunk(mtmd_ctx, chunk)

			if err != nil {
				return nil, fmt.Errorf("Failed to encode image chunk, %w", err)
			}

			n_tokens := int(mtmd.InputChunkGetNTokens(chunk))

			image_embd, err := mtmd.GetOutputEmbd(mtmd_ctx, int32(n_embd_inp*n_tokens))

			if err != nil {
				return nil, fmt.Errorf("Failed to read image embeddings, %w", err)
			}

			for row := 0; row < n_tokens; row++ {

				err := add(llama.BatchExtAddEmbd(batch, 0, image_embd[row*n_embd_inp:(row+1)*n_embd_inp], n_embd_inp))

				if err != nil {
					return nil, fmt.Errorf("Failed to add image embedding to batch, %w", err)
				}
			}
		}
	}

	eval_rsp, err := llama.Process(emb_ctx, llama.ProcessTypeDecode, batch)

	if err != nil || eval_rsp != 0 {
		return nil, fmt.Errorf("Multimodal image token layer evaluation failed with status %d, %w", eval_rsp, err)
	}

	// 12. Read vector mapping state layers out from your execution block

	// Use the output size: it can differ from n_embd (e.g. embeddinggemma-2: 768 vs 512).
	nEmbd := llama.ModelNEmbdOut(model)

	vec, err := llama.GetEmbeddingsSeq(emb_ctx, 0, nEmbd)

	if err != nil {
		return nil, fmt.Errorf("Failed to extract multimodal embeddings from context, %w", err)
	}

	return e.generateResponse(req, vec)
}

// Close releases resources held by the yzma backend. It also removes any temporary directories that were created for downloading the yzma binary.
func (e *YzmaEmbedder[T]) Close() error {

	e.models_mu.Lock()

	for uri, entry := range e.models {
		entry.free()
		delete(e.models, uri)
	}

	e.models_mu.Unlock()

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

// acquireSlot returns the cached model for the request (loading it on first use) and a slot to run
// it on, creating a new context if none is idle and the pool isn't full, otherwise waiting for one.
// The caller must call `entry.release(slot)`.
func (e *YzmaEmbedder[T]) acquireSlot(ctx context.Context, req *EmbeddingsRequest) (*yzmaModel, *yzmaSlot, error) {

	e.models_mu.Lock()

	entry, ok := e.models[req.Model]

	if !ok {

		model, err := e.getModelForRequest(ctx, req)

		if err != nil {
			e.models_mu.Unlock()
			return nil, nil, err
		}

		entry = &yzmaModel{
			model: model,
			idle:  make(chan *yzmaSlot, e.max_slots),
			max:   e.max_slots,
		}

		e.models[req.Model] = entry
	}

	e.models_mu.Unlock()

	select {
	case slot := <-entry.idle:
		return entry, slot, nil
	default:
	}

	entry.mu.Lock()
	can_create := entry.created < entry.max

	if can_create {
		entry.created++
	}

	entry.mu.Unlock()

	if can_create {

		model_ctx := llama.ContextDefaultParams()
		model_ctx.NCtx = uint32(e.context_sz)
		model_ctx.NBatch = uint32(e.batch_sz)
		model_ctx.PoolingType = e.pooling_type
		model_ctx.Embeddings = 1

		emb_ctx, err := llama.InitFromModel(entry.model, model_ctx)

		if err != nil {

			entry.mu.Lock()
			entry.created--
			entry.mu.Unlock()

			return nil, nil, fmt.Errorf("Unable to initialize context from model, %w", err)
		}

		return entry, &yzmaSlot{ctx: emb_ctx}, nil
	}

	select {
	case slot := <-entry.idle:
		return entry, slot, nil
	case <-ctx.Done():
		return nil, nil, ctx.Err()
	}
}

func (m *yzmaModel) release(s *yzmaSlot) {
	m.idle <- s
}

// free releases all idle slots and the model. It must only be called when no requests are in flight.
func (m *yzmaModel) free() {

	for {
		select {
		case s := <-m.idle:
			s.free()
			continue
		default:
		}

		break
	}

	if m.model != 0 {
		llama.ModelFree(m.model)
		m.model = 0
	}
}

// getMtmdContext returns the multimodal context for the entry, creating it (and loading the mtmd
// library) on first use. The caller must own `slot`.
func (e *YzmaEmbedder[T]) getMtmdContext(ctx context.Context, req *EmbeddingsRequest, entry *yzmaModel, slot *yzmaSlot) (mtmd.Context, error) {

	e.mtmd_once.Do(func() {

		e.mtmd_err = mtmd.Load(e.lib_path)

		if e.mtmd_err == nil && !e.llama_verbose {
			mtmd.LogSet(llama.LogSilent())
		}
	})

	if e.mtmd_err != nil {
		return 0, fmt.Errorf("Failed to load mtmd library from %s, %w", e.lib_path, e.mtmd_err)
	}

	if slot.mtmd_ctx != 0 {
		return slot.mtmd_ctx, nil
	}

	// Resolve the location of the projector matching the model

	u, err := url.Parse(req.Model)

	if err != nil {
		return 0, err
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
		return 0, fmt.Errorf("Unsupported model scheme")
	}

	mmproj_path, err := e.fetchModel(ctx, mmproj_uri)

	if err != nil {
		return 0, fmt.Errorf("Failed to fetch multimodal model %s, %w", mmproj_uri, err)
	}

	mtmd_ctx, err := mtmd.InitFromFile(mmproj_path, entry.model, mtmd.ContextParamsDefault())

	if err != nil {
		return 0, fmt.Errorf("Failed to initialize multimodal context, %w", err)
	}

	slot.mtmd_ctx = mtmd_ctx
	return mtmd_ctx, nil
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

	fetch_model := false

	slog.Debug("Fetch model", "uri", model_uri, "path", model_path)

	info, err := os.Stat(model_path)

	if err != nil {

		if !os.IsNotExist(err) {
			return "", err
		}

		slog.Debug("Model does not exist, fetch")
		fetch_model = true
	} else {

		if info.Size() == 0 {
			slog.Debug("Model has zero size, try to fetch again")
			fetch_model = true
		}
	}

	if fetch_model {

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
