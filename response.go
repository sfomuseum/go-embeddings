package embeddings

type EmbeddingsResponse interface {
	Id() string
	Model() string
	Dimensions() int32
	Precision() uint8
	Created() int64
}

type CommonEmbeddingsResponse struct {
	EmbeddingsResponse
	CommonId           string    `json:"id,omitempty"`
	CommonEmbeddings32 []float32 `json:"embeddings32,omitempty"`
	CommonEmbeddings64 []float64 `json:"embeddings64,omitempty"`
	CommonModel        string    `json:"model"`
	CommonCreated      int64     `json:"created"`
	CommonPrecision    uint8     `json:"precision"`
}

func (r *CommonEmbeddingsResponse) Id() string {
	return r.CommonId
}

func (r *CommonEmbeddingsResponse) Model() string {
	return r.CommonModel
}

func (r *CommonEmbeddingsResponse) Created() int64 {
	return r.CommonCreated
}

func (r *CommonEmbeddingsResponse) Precision() uint8 {
	return r.CommonPrecision
}

func (r *CommonEmbeddingsResponse) Embeddings32() []float32 {

	switch r.CommonPrecision {
	case 32:
		return r.CommonEmbeddings32
	case 64:
		return AsFloat32(r.CommonEmbeddings64)
	default:
		return make([]float32, 0)
	}

}

func (r *CommonEmbeddingsResponse) Embeddings64() []float64 {

	switch r.CommonPrecision {
	case 32:
		return AsFloat64(r.CommonEmbeddings32)
	case 64:
		return r.CommonEmbeddings64
	default:
		return make([]float64, 0)
	}

}

func (r *CommonEmbeddingsResponse) Dimensions() int32 {

	switch r.CommonPrecision {
	case 32:
		return int32(len(r.CommonEmbeddings32))
	case 64:
		return int32(len(r.CommonEmbeddings64))
	default:
		return 0
	}

}
