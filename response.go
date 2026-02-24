package embeddings

type Float interface{ ~float32 | ~float64 }

type EmbeddingsResponse[T Float] interface {
	Id() string
	Model() string
	Embeddings() []T
	Dimensions() int32
	Precision() uint8
	Created() int64
}

type CommonEmbeddingsResponse[T Float] struct {
	EmbeddingsResponse[T] `json:",omitempty"`
	CommonId              string `json:"id,omitempty"`
	CommonEmbeddings      []T    `json:"embeddings"`
	CommonModel           string `json:"model"`
	CommonCreated         int64  `json:"created"`
	CommonPrecision       uint8  `json:"precision"`
}

func (r *CommonEmbeddingsResponse[T]) Id() string {
	return r.CommonId
}

func (r *CommonEmbeddingsResponse[T]) Model() string {
	return r.CommonModel
}

func (r *CommonEmbeddingsResponse[T]) Created() int64 {
	return r.CommonCreated
}

func (r *CommonEmbeddingsResponse[T]) Precision() uint8 {
	return r.CommonPrecision
}

func (r *CommonEmbeddingsResponse[T]) Embeddings() []T {
	return r.CommonEmbeddings
}

func (r *CommonEmbeddingsResponse[T]) Dimensions() int32 {
	return int32(len(r.CommonEmbeddings))
}
