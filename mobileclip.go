//go:build mobileclip

package embeddings

import (
	"context"
	"encoding/json"
	"fmt"
	_ "log/slog"
	"net/url"
	"os"
	"os/exec"
)

type MobileCLIPEmbeddingRequest struct {
	Content   string `json:"content,omitempty"`
	ImageData []byte `json:"image_data,omitempty"`
}

type MobileCLIPEmbeddingResponse struct {
	Embeddings []float64 `json:"embeddings"`
	Dimensions int       `json:"dimensions"`
	Model      string    `json:"model"`
	Type       string    `json:"type"`
	Created    int64     `json:"created"`
}

// MobileCLIPEmbedder implements the `Embedder` interface using an MobileCLIP API endpoint to derive embeddings.
type MobileCLIPEmbedder struct {
	Embedder
	tool        string
	encoder_uri string
}

func init() {
	ctx := context.Background()
	err := RegisterEmbedder(ctx, "mobileclip", NewMobileCLIPEmbedder)

	if err != nil {
		panic(err)
	}
}

func NewMobileCLIPEmbedder(ctx context.Context, uri string) (Embedder, error) {

	u, err := url.Parse(uri)
	q := u.Query()

	if !q.Has("tool") {
		return nil, fmt.Errorf("Missing ?tool= parameter")
	}

	tool := q.Get("tool")

	info, err := os.Stat(tool)

	if err != nil {
		return nil, fmt.Errorf("Failed to stat tool, %w", err)
	}

	if info.IsDir() {
		return nil, fmt.Errorf("Tool is a directory")
	}

	if !q.Has("encoder_uri") {
		return nil, fmt.Errorf("Missing ?encoder_uri= parameter")
	}

	encoder_uri := q.Get("encoder_uri")

	_, err = url.Parse(encoder_uri)

	if err != nil {
		return nil, fmt.Errorf("Invalid embedder URI, %w", err)
	}

	e := &MobileCLIPEmbedder{
		tool:        tool,
		encoder_uri: encoder_uri,
	}

	return e, nil
}

func (e *MobileCLIPEmbedder) Embeddings(ctx context.Context, content string) ([]float64, error) {

	req := &MobileCLIPEmbeddingRequest{
		Content: content,
	}

	rsp, err := e.embeddings(ctx, req)

	if err != nil {
		return nil, err
	}

	return rsp.Embeddings, nil
}

func (e *MobileCLIPEmbedder) Embeddings32(ctx context.Context, content string) ([]float32, error) {

	e64, err := e.Embeddings(ctx, content)

	if err != nil {
		return nil, err
	}

	return AsFloat32(e64), nil
}

func (e *MobileCLIPEmbedder) ImageEmbeddings(ctx context.Context, data []byte) ([]float64, error) {

	req := &MobileCLIPEmbeddingRequest{
		ImageData: data,
	}

	rsp, err := e.embeddings(ctx, req)

	if err != nil {
		return nil, err
	}

	return rsp.Embeddings, nil
}

func (e *MobileCLIPEmbedder) ImageEmbeddings32(ctx context.Context, data []byte) ([]float32, error) {

	e64, err := e.ImageEmbeddings(ctx, data)

	if err != nil {
		return nil, err
	}

	return AsFloat32(e64), nil
}

func (e *MobileCLIPEmbedder) embeddings(ctx context.Context, mobileclip_req *MobileCLIPEmbeddingRequest) (*MobileCLIPEmbeddingResponse, error) {

	args := make([]string, 0)

	if len(mobileclip_req.ImageData) > 0 {
		args = append(args, "image")
	} else {
		args = append(args, "text")
	}

	args = append(args, "--encoder_uri")
	args = append(args, e.encoder_uri)

	wr, err := os.CreateTemp("", "mobileclip")

	if err != nil {
		return nil, err
	}

	defer os.Remove(wr.Name())

	if len(mobileclip_req.ImageData) > 0 {

		_, err = wr.Write(mobileclip_req.ImageData)

		if err != nil {
			return nil, err
		}

	} else {

		_, err = wr.Write([]byte(mobileclip_req.Content))

		if err != nil {
			return nil, err
		}
	}

	err = wr.Close()

	if err != nil {
		return nil, err
	}

	if len(mobileclip_req.ImageData) > 0 {
		args = append(args, "--path")
	}

	args = append(args, wr.Name())

	cmd := exec.CommandContext(ctx, e.tool, args...)

	body, err := cmd.Output()

	if err != nil {
		return nil, err
	}

	var mobileclip_rsp *MobileCLIPEmbeddingResponse

	err = json.Unmarshal(body, &mobileclip_rsp)

	if err != nil {
		return nil, fmt.Errorf("Failed to unmarshal embeddings, %w", err)
	}

	return mobileclip_rsp, nil
}
