package mobileclip

import (
	"context"
)

type NullEmbeddingsClient struct{}

func init() {
	ctx := context.Background()
	RegisterEmbeddingsClient(ctx, "null", NewNullEmbeddingsClient)
}

func NewNullEmbeddingsClient(ctx context.Context, uri string) (EmbeddingsClient, error) {
	e := &NullEmbeddingsClient{}
	return e, nil
}

func (e *NullEmbeddingsClient) ComputeTextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (*Embeddings, error) {
	return new(Embeddings), nil
}

func (e *NullEmbeddingsClient) ComputeImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (*Embeddings, error) {
	return new(Embeddings), nil
}
