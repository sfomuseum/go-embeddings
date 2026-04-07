package embeddings

// https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/image_text/README_siglip2.md

import (
	"context"
	"net/url"
)

func init() {
	ctx := context.Background()
	RegisterEmbedder[float32](ctx, "siglip", NewSigLIPEmbedder)
	RegisterEmbedder[float32](ctx, "siglip32", NewSigLIPEmbedder)
	RegisterEmbedder[float64](ctx, "siglip64", NewSigLIPEmbedder)
}

func NewSigLIPEmbedder[T Float](ctx context.Context, uri string) (Embedder[T], error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	if u.Host != "" {
		return NewSigLIPLocalClientEmbedder[T](ctx, uri)
	}

	return NewSigLIPCommandLineEmbedder[T](ctx, uri)
}
