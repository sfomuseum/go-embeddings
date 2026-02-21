package mobileclip

import (
	"context"
	"fmt"
	"net/url"
	"sort"
	"strings"

	"github.com/aaronland/go-roster"
)

type Embeddings struct {
	Id         string    `json:"id,omitempty"`
	Embeddings []float32 `json:"embeddings"`
	Dimensions int32     `json:"dimensions"`
	Model      string    `json:"model"`
	Created    int64     `json:"created"`
}

type EmbeddingsRequest struct {
	Id    string `json:"id,omitempty"`
	Model string `json:"model"`
	Body  []byte `json:"body"`
}

type EmbeddingsClient interface {
	ComputeTextEmbeddings(context.Context, *EmbeddingsRequest) (*Embeddings, error)
	ComputeImageEmbeddings(context.Context, *EmbeddingsRequest) (*Embeddings, error)
}

var client_roster roster.Roster

// ClientInitializationFunc is a function defined by individual client package and used to create
// an instance of that client
type EmbeddingsClientInitializationFunc func(ctx context.Context, uri string) (EmbeddingsClient, error)

// RegisterEmbeddingsClient registers 'scheme' as a key pointing to 'init_func' in an internal lookup table
// used to create new `Client` instances by the `NewEmbeddingsClient` method.
func RegisterEmbeddingsClient(ctx context.Context, scheme string, init_func EmbeddingsClientInitializationFunc) error {

	err := ensureClientRoster()

	if err != nil {
		return err
	}

	return client_roster.Register(ctx, scheme, init_func)
}

func ensureClientRoster() error {

	if client_roster == nil {

		r, err := roster.NewDefaultRoster()

		if err != nil {
			return err
		}

		client_roster = r
	}

	return nil
}

// NewClient returns a new `Client` instance configured by 'uri'. The value of 'uri' is parsed
// as a `url.URL` and its scheme is used as the key for a corresponding `ClientInitializationFunc`
// function used to instantiate the new `Client`. It is assumed that the scheme (and initialization
// function) have been registered by the `RegisterClient` method.
func NewEmbeddingsClient(ctx context.Context, uri string) (EmbeddingsClient, error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	scheme := u.Scheme

	i, err := client_roster.Driver(ctx, scheme)

	if err != nil {
		return nil, err
	}

	if i == nil {
		return nil, fmt.Errorf("Unregistered client")
	}

	init_func := i.(EmbeddingsClientInitializationFunc)

	return init_func(ctx, uri)
}

// Schemes returns the list of schemes that have been registered.
func Schemes() []string {

	ctx := context.Background()
	schemes := []string{}

	err := ensureClientRoster()

	if err != nil {
		return schemes
	}

	for _, dr := range client_roster.Drivers(ctx) {
		scheme := fmt.Sprintf("%s://", strings.ToLower(dr))
		schemes = append(schemes, scheme)
	}

	sort.Strings(schemes)
	return schemes
}
