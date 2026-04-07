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

func init() {
	ctx := context.Background()
	RegisterEmbedder[float32](ctx, "siglip", NewSigLIPCommandLineEmbedder)
	RegisterEmbedder[float32](ctx, "siglip32", NewSigLIPCommandLineEmbedder)
	RegisterEmbedder[float64](ctx, "siglip64", NewSigLIPCommandLineEmbedder)
}

type SigLIPCommandLineEmbedder[T Float] struct {
	Embedder[T]
	python        string
	embeddings_py string
	model         string
	precision     string
}

func NewSigLIPCommandLineEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

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

	if !q.Has("model") {
		return nil, fmt.Errorf("Required model (HuggingFace checkpoint URI) missing.")
	}

	model := q.Get("model")

	precision := "float32"

	if strings.HasSuffix(u.Scheme, "64") {
		precision = fmt.Sprintf("%s#as-float%d", precision, 64)
	}

	e := &SigLIPCommandLineEmbedder[T]{
		python:        python,
		embeddings_py: embeddings_py,
		precision:     precision,
		model:         model,
	}

	return e, nil
}

func (e *SigLIPCommandLineEmbedder[T]) TextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	return e.generateEmbeddingsFromCommandLine(ctx, req, "text", string(req.Body))
}

func (e *SigLIPCommandLineEmbedder[T]) ImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (EmbeddingsResponse[T], error) {

	tmp, err := os.CreateTemp("", "siglip.*.img")

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

	return e.generateEmbeddingsFromCommandLine(ctx, req, "image", tmp.Name())
}

func (e *SigLIPCommandLineEmbedder[T]) generateEmbeddingsFromCommandLine(ctx context.Context, req *EmbeddingsRequest, target string, input string) (EmbeddingsResponse[T], error) {

	tmp, err := os.CreateTemp("", "siglip.*.json")

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
		"--model_name", e.model,
		"--embeddings_type", target,
		"--embeddings_source", input,
		"--embeddings_output", tmp.Name(),
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

	var e64 []float64

	dec := json.NewDecoder(r)
	err = dec.Decode(&e64)

	if err != nil {
		return nil, fmt.Errorf("Failed to unmarshal embeddings, %w (%s)", err, tmp.Name())
	}

	now := time.Now()
	ts := now.Unix()

	rsp := &CommonEmbeddingsResponse[T]{
		CommonId:        req.Id,
		CommonPrecision: e.precision,
		CommonCreated:   ts,
		CommonModel:     e.model,
	}

	switch {
	case strings.HasSuffix(e.precision, "32"):
		rsp.CommonEmbeddings = toFloat32Slice[T](AsFloat32(e64))
	default:
		rsp.CommonEmbeddings = toFloat64Slice[T](e64)
	}

	return rsp, nil
}
