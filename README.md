# go-embeddings

Go package defining a common interface for generating text and image embeddings.

## Documentation

`godoc` is currently incomplete.

## Design

To account for the fact that most embeddings models still return `float32` vector data but an increasing number of models return `float64` vectors this package wraps both options in a `Float` interface.

```
type Float interface{ ~float32 | ~float64 }
```

That `Float` is then used as a generic value (for embeddings) in a common `EmbeddingsResponse` interface:

```
type EmbeddingsResponse[T Float] interface {
	Id() string
	Model() string
	Embeddings() []T
	Dimensions() int32
	Precision() string
	Created() int64
}
```

That interface is then used as the return value for an `Embedder` interface:

```
type Embedder[T Float] interface {
	TextEmbeddings(context.Context, *EmbeddingsRequest) (EmbeddingsResponse[T], error)
	ImageEmbeddings(context.Context, *EmbeddingsRequest) (EmbeddingsResponse[T], error)
}
```

This means that you need to specify the float type you want the interface to return when you instantiate that interface. For example:

```
ctx := context.Backgroud()

uri32 := "ollama://?model=embeddinggemma"
uri64 := "encoderfile://"

cl, _ := embeddings.NewEmbedder[float32](ctx, uri32)
cl, _ := embeddings.NewEmbedder[float64](ctx, uri64)
```

There are also handy `NewEmbedder32` and `NewEmbedder64` methods which are little more than syntactic sugar. For example:

```
ctx := context.Backgroud()

uri32 := "ollama://?model=embeddinggemma"
uri64 := "encoderfile://"

cl, _ := embeddings.NewEmbedder32(ctx, uri32)
cl, _ := embeddings.NewEmbedder64(ctx, uri64)
```

The `NewEmbedder`, `NewEmbedder32` and `NewEmbedder64` all have the same signature: A `context.Context` instance and a URI string used to configure and instantiate the underlying embeddings provider implementation. These are discussed in detail below.

Both the `TextEmbeddings` and `ImageEmbeddings` methods take the same input, a `EmbeddingsRequest` struct:

```
type EmbeddingsRequest struct {
	Id    string `json:"id,omitempty"`
	Model string `json:"model"`
	Body  []byte `json:"body"`
}
```

As mentioned both methods return an `EmbeddingsResponse[T]` instance. The default implementation of the `EmbeddingsResponse[T]` interface used by this package is the `CommonEmbeddingsResponse` type. See [response.go](response.go) for details.

## Example

_Error handling omitted for the sake of brevity._

```
import (
	"context"
	"encoding/json"
	"os"

	"github.com/sfomuseum/go-embeddings"
)

func main() {

	ctx := context.Background()

	emb, _ := embeddings.NewEmbedder32(ctx, "ollama://?model=embeddinggemma")

	req := &embeddings.EmbeddingsRequest{
		Body: []byte("Hello world"),
	}

	rsp, _ := emb.TextEmbeddings(ctx, req)

	enc := json.NewEncoder(os.Stdout)
	enc.Encode(rsp)
```

Which would return the following:

```
{
  "embeddings": [
    -0.21400317549705505,
    0.02651195414364338,
    ... more embeddings
    -0.04678588733077049,
    -0.042774248868227005
  ],
  "model": "ollama/embeddinggemma",
  "created": 1771985811,
  "precision": "float32"
}
```

## Precision

...

## Implementations

### encoderfile://

```
encoderfile://?{PARAMETERS}
```

| Name | Value | Required | Notes |
| --- | --- | --- | --- |
| client-uri | string | no | Default is `http://localhost:8080`. |

### llamafile://

```
llamafile://?{PARAMETERS}
```

| Name | Value | Required | Notes |
| --- | --- | --- | --- |
| client-uri | string | no | Default is `http://localhost:8080`. |

### mlxclip://

```
mlxclip://{PATH_TO_EMBEDDINGS_DOT_PY}
```

### mobileclip://

```
mobileclip://?{PARAMETERS}
```

| Name | Value | Required | Notes |
| --- | --- | --- | --- |
| client-uri | string | yes | ... |
| model | string | yes | ... |

### null://

```
null://
```

### ollama://

```
ollama://?{PARAMETERS}
```

| Name | Value | Required | Notes |
| --- | --- | --- | --- |
| client-uri | string | no | Default is `http://localhost:11434`. |

### openclip://

```
openclip://?{PARAMETERS}
```

| Name | Value | Required | Notes |
| --- | --- | --- | --- |
| client-uri | string | no | Default is `http://localhost:8080`. |

## Tests

