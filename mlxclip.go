package embeddings

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"strings"
	"time"
)

type MLXClipEmbedder[T Float] struct {
	Embedder[T]
	embeddings_py string
	precision     string
}

func init() {
	ctx := context.Background()
	RegisterEmbedder[float32](ctx, "mlxclip", NewMLXClipEmbedder)
	RegisterEmbedder[float32](ctx, "mlxclip32", NewMLXClipEmbedder)
	RegisterEmbedder[float64](ctx, "mlxclip64", NewMLXClipEmbedder)
}

func NewMLXClipEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, fmt.Errorf("Failed to parse URI, %w", err)
	}

	embeddings_py := u.Path

	_, err = os.Stat(embeddings_py)

	if err != nil {
		return nil, err
	}

	precision := "float32"

	if strings.HasSuffix(u.Scheme, "64") {
		precision = fmt.Sprintf("%s#as-float%d", precision, 64)
	}

	e := &MLXClipEmbedder[T]{
		embeddings_py: embeddings_py,
		precision:     precision,
	}

	return e, nil
}

func (e *MLXClipEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {
	return e.generate_embeddings(ctx, req, "text", string(req.Body))
}

func (e *MLXClipEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	tmp, err := os.CreateTemp("", "mlxclip.*.img")

	if err != nil {
		return nil, fmt.Errorf("Failed to create tmp file, %w", err)
	}

	defer os.Remove(tmp.Name()) // clean up

	_, err = tmp.Write(req.Body)

	if err != nil {
		return nil, err
	}

	err = tmp.Close()

	if err != nil {
		return nil, err
	}

	return e.generate_embeddings(ctx, req, "image", tmp.Name())
}

func (e *MLXClipEmbedder[T]) generate_embeddings(ctx context.Context, req *EmbeddingsRequest, target string, input string) (EmbeddingsResponse[T], error) {

	tmp, err := os.CreateTemp("", "mlxclip.*.json")

	if err != nil {
		return nil, fmt.Errorf("Failed to create tmp file, %w", err)
	}

	defer os.Remove(tmp.Name())

	err = tmp.Close()

	if err != nil {
		return nil, err
	}

	cmd := exec.CommandContext(ctx, "python3", e.embeddings_py, target, input, tmp.Name())
	err = cmd.Run()

	if err != nil {
		return nil, fmt.Errorf("Failed to derive embeddings, %w", err)
	}

	r, err := os.Open(tmp.Name())

	if err != nil {
		return nil, err
	}

	defer r.Close()

	var e32 []float32

	dec := json.NewDecoder(r)
	err = dec.Decode(&e32)

	if err != nil {
		return nil, fmt.Errorf("Failed to unmarshal embeddings, %w (%s)", err, tmp.Name())
	}

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonPrecision: e.precision,
		CommonCreated:   ts,
		CommonModel:     "mlxclip",
	}

	switch {
	case strings.HasSuffix(e.precision, "64"):
		rsp.CommonEmbeddings = toFloat64Slice[T](AsFloat64(e32))
	default:
		rsp.CommonEmbeddings = toFloat32Slice[T](e32)
	}

	return rsp, nil
}
