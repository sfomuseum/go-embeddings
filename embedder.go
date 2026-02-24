package embeddings

import (
	"context"
	"fmt"
	"net/url"
	"sort"
	"strings"

	"github.com/aaronland/go-roster"
)

type EmbeddingsPrecision uint8

const (
	_         EmbeddingsPrecision = iota
	Micro                         // 4
	Mini                          // 8
	Half                          // 16
	Single                        // 32
	Double                        // 64
	Quadruple                     // 128
)

type EmbeddingsRequest struct {
	Id    string `json:"id,omitempty"`
	Model string `json:"model"`
	Body  []byte `json:"body"`
}

type EmbeddingsResponseTest interface {
	Id() string
	Model() string
	Embeddings32() []float32
	Embeddings64() []float64
	Dimensions() int32
	Precision() uint8
	Created() int64
}

type CommonEmbeddingsResponse struct {
	EmbeddingsResponseTest
	CommonId           string    `json:"id,omitempty"`
	CommonEmbeddings64 []float64 `json:"embeddings64,omitempty"`
	CommonEmbeddings32 []float32 `json:"embeddings32,omitempty"`
	CommonDimensions   int32     `json:"dimensions"`
	CommonModel        string    `json:"model"`
	CommonCreated      int64     `json:"created"`
	CommonPrecision    uint8     `json:"precision"`
}

type EmbeddingsResponse struct {
	Id         string    `json:"id,omitempty"`
	Embeddings []float64 `json:"embeddings"`
	Dimensions int32     `json:"dimensions"`
	Model      string    `json:"model"`
	Created    int64     `json:"created"`
}

type EmbeddingsResponse32 struct {
	Id         string    `json:"id,omitempty"`
	Embeddings []float32 `json:"embeddings"`
	Dimensions int32     `json:"dimensions"`
	Model      string    `json:"model"`
	Created    int64     `json:"created"`
}

// Embedder defines an interface for generating (vector) embeddings
type Embedder interface {
	// Embeddings returns ...
	Embeddings(context.Context, *EmbeddingsRequest) (*EmbeddingsResponse, error)
	// Embeddings32 returns ...
	Embeddings32(context.Context, *EmbeddingsRequest) (*EmbeddingsResponse32, error)
	// ImageEmbeddings returns ...
	ImageEmbeddings(context.Context, *EmbeddingsRequest) (*EmbeddingsResponse, error)
	// ImageEmbeddings32 returns ...
	ImageEmbeddings32(context.Context, *EmbeddingsRequest) (*EmbeddingsResponse32, error)
}

// EmbedderInitializationFunc is a function defined by individual embedder package and used to create
// an instance of that embedder
type EmbedderInitializationFunc func(ctx context.Context, uri string) (Embedder, error)

var embedder_roster roster.Roster

// RegisterEmbedder registers 'scheme' as a key pointing to 'init_func' in an internal lookup table
// used to create new `Embedder` instances by the `NewEmbedder` method.
func RegisterEmbedder(ctx context.Context, scheme string, init_func EmbedderInitializationFunc) error {

	err := ensureEmbedderRoster()

	if err != nil {
		return err
	}

	return embedder_roster.Register(ctx, scheme, init_func)
}

func ensureEmbedderRoster() error {

	if embedder_roster == nil {

		r, err := roster.NewDefaultRoster()

		if err != nil {
			return err
		}

		embedder_roster = r
	}

	return nil
}

// NewEmbedder returns a new `Embedder` instance configured by 'uri'. The value of 'uri' is parsed
// as a `url.URL` and its scheme is used as the key for a corresponding `EmbedderInitializationFunc`
// function used to instantiate the new `Embedder`. It is assumed that the scheme (and initialization
// function) have been registered by the `RegisterEmbedder` method.
func NewEmbedder(ctx context.Context, uri string) (Embedder, error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	scheme := u.Scheme

	i, err := embedder_roster.Driver(ctx, scheme)

	if err != nil {
		return nil, err
	}

	init_func := i.(EmbedderInitializationFunc)
	return init_func(ctx, uri)
}

// Schemes returns the list of schemes that have been registered.
func EmbedderSchemes() []string {

	ctx := context.Background()
	schemes := []string{}

	err := ensureEmbedderRoster()

	if err != nil {
		return schemes
	}

	for _, dr := range embedder_roster.Drivers(ctx) {
		scheme := fmt.Sprintf("%s://", strings.ToLower(dr))
		schemes = append(schemes, scheme)
	}

	sort.Strings(schemes)
	return schemes
}
