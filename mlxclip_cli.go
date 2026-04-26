package embeddings

import (
	"context"
	"encoding/json"
	"fmt"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"
)

type MLXClipEmbedder[T Float] struct {
	Embedder[T]
	python        string
	model_dir string
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

	q := u.Query()

	embeddings_py, err := filepath.Abs(u.Path)

	if err != nil {
		return nil, err
	}

	_, err = os.Stat(embeddings_py)

	if err != nil {
		return nil, err
	}

	precision := "float64"

	if strings.HasSuffix(u.Scheme, "32") {
		precision = fmt.Sprintf("%s#as-float%d", precision, 32)
	}

	if !q.Has("model"){
		return nil, fmt.Errorf("Missing ?model= parameter")
	}

	model := q.Get("model")

	model_dir, err := filepath.Abs(model)

	if err != nil {
		return nil, fmt.Errorf("Failed to determine absolute path for model, %w", err)
	}

	info, err := os.Stat(model_dir)

	if err != nil {
		return nil, fmt.Errorf("Failed to stat model directory, %w", err)
	}

	if !info.IsDir(){
		return nil, fmt.Errorf("Model directory must be... a directory")
	}
	
	python := "python"

	if q.Has("python") {

		abs_python, err := filepath.Abs(q.Get("python"))

		if err != nil {
			return nil, err
		}

		_, err = os.Stat(abs_python)

		if err != nil {
			return nil, err
		}

		python = abs_python
	}

	e := &MLXClipEmbedder[T]{
		python:        python,
		model_dir: model_dir,
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

	args := []string{
		e.embeddings_py,
		"--model_dir",
		e.model_dir,
		"--mode",
		target,
		"--input",
		input,
		"--output",
		tmp.Name(),
	}

	cmd := exec.CommandContext(ctx, e.python, args...)
	err = cmd.Run()

	if err != nil {
		return nil, fmt.Errorf("Failed to derive embeddings, %w", err)
	}

	r, err := os.Open(tmp.Name())

	if err != nil {
		return nil, err
	}

	defer r.Close()

	var emb_rsp *MLXClipEmbeddingsResponse

	dec := json.NewDecoder(r)
	err = dec.Decode(&emb_rsp)

	if err != nil {
		return nil, fmt.Errorf("Failed to unmarshal embeddings, %w (%s)", err, tmp.Name())
	}

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonPrecision: e.precision,
		CommonCreated:   ts,
		CommonModel:     emb_rsp.Model,
	}

	switch {
	case strings.HasSuffix(e.precision, "32"):
		rsp.CommonEmbeddings = toFloat32Slice[T](AsFloat32(emb_rsp.Embeddings))
	default:
		rsp.CommonEmbeddings = toFloat64Slice[T](emb_rsp.Embeddings)
	}

	return rsp, nil
}
