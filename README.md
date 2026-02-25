# go-embeddings

Go package defining a common interface for generating text and image embeddings.

## Documentation

`godoc` is currently incomplete.

## Motivation

This is a simple abstraction library, written in Go, around a variety of services which produce vector embeddings. There are many such libraries and this one is ours. It tries to be the "simplest dumbest" thing for the most common operations and data needs. These ideas are encapsulated in the `EmbeddingsRequest` and `EmbeddingsResponse` types.

```
type EmbeddingsRequest struct {
	Id    string `json:"id,omitempty"`
	Model string `json:"model"`
	Body  []byte `json:"body"`
}

type EmbeddingsResponse[T Float] interface {
	Id() string
	Model() string
	Embeddings() []T
	Dimensions() int32
	Precision() string
	Created() int64
}
```

The default implementation of the `EmbeddingsResponse` interface is the `CommonEmbeddingsResponse` type:

```
type CommonEmbeddingsResponse[T Float] struct {
	EmbeddingsResponse[T] `json:",omitempty"`
	CommonId              string `json:"id,omitempty"`
	CommonEmbeddings      []T    `json:"embeddings"`
	CommonModel           string `json:"model"`
	CommonCreated         int64  `json:"created"`
	CommonPrecision       string `json:"precision"`
}
```

While not specific to SFO Museum this package is targeted at the kinds of things SFO Museum needs to today meaning it may be lacking features you need or want. 

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

Derive vector embeddings from an instance of the Mozilla [encoderfile](https://www.mozilla.ai/open-tools/encoderfile) application, running as an HTTP server.

```
encoderfile://?{PARAMETERS}
```

| Name | Value | Required | Notes |
| --- | --- | --- | --- |
| client-uri | string | no | The URI for the `embedderfile` HTTP server endpoint. Default is `http://localhost:8080`. The gRPC server endpoint provided by `encoderfile` is not supported yet. |

* https://www.mozilla.ai/open-tools/encoderfile
* https://github.com/sfomuseum/go-encoderfile

### llamafile://

Derive vector embedding from an instance of the Mozilla [llamafile](#) application. Note that newer versions of `llamafile` not longer expose an interface for deriving embeddings so this implementation will only work with older builds. See the `encoderfile://` implementation for an alternative.

```
llamafile://?{PARAMETERS}
```

| Name | Value | Required | Notes |
| --- | --- | --- | --- |
| client-uri | string | no | The URI for the `llamafile` HTTP server endpoint. Default is `http://localhost:8080`. |

* https://github.com/mozilla-ai/llamafile/

### mlxclip://

Derive vector embeddings from a Python script using the [harperreed/mlx_clip](https://github.com/harperreed/mlx_clip) library.

The option requires a device using an Apple Silicon chip and involves a non-zero manual set up process discussed below.

```
mlxclip://{PATH_TO_EMBEDDINGS_DOT_PY}
```

* https://github.com/harperreed/mlx_clip

### mobileclip://

Derive vector embeddings from the MobileCLIP models exposed via an instance of the [sfomuseum/swift-mobileclip](https://github.com/sfomuseum/swift-mobileclip) gRPC endpoint. 

```
mobileclip://?{PARAMETERS}
```

| Name | Value | Required | Notes |
| --- | --- | --- | --- |
| client-uri | string | yes | The URI for the `swift-mobileclip` gRPC server endpoint. Default is `grpc://localhosr:8080`. |
| model | string | yes | The URI of the model to use for generating embeddings. |

* https://github.com/apple/ml-mobileclip
* https://github.com/sfomuseum/swift-mobileclip
* https://github.com/sfomuseum/go-mobileclip

### null://

Derive null (empty) vector embeddings. This is a "placeholder" implementation that will always return a zero-length list of embeddings.

```
null://
```

### ollama://

Derive vector embeddings from an instance of the [Ollama](https://ollama.com/) application.

```
ollama://?{PARAMETERS}
```

| Name | Value | Required | Notes |
| --- | --- | --- | --- |
| client-uri | string | no | Default is `http://localhost:11434`. |
| model | string | yes | The name of the model to use for generating embeddings. |

* https://ollama.com/
* https://docs.ollama.com/api/introduction

### openclip://

Derive vector embeddings from a web service exposing the [OpenCLIP](https://github.com/mlfoundations/open_clip) model and library.

The option involves a non-zero manual set up process discussed below.

```
openclip://?{PARAMETERS}
```

| Name | Value | Required | Notes |
| --- | --- | --- | --- |
| client-uri | string | no | The URI of the HTTP endpoint exposing the OpenCLIP model functionality. Default is `http://localhost:8080`. |

* https://github.com/mlfoundations/open_clip

#### Set up

```
$> python -m venv openclip
$> cd openclip/
$> bash bin/activate
$> bin/pip install flask
$> bin/pip install open_clip_torch
$> bin/pip install Pillow
```

Then, copy the code in [openclip_server.txt](openclip_server.txt) in to a file called openclip_server.py and launch it as follows:

```
$> bin/flask --app openclip_server run
```

## Tests

Because so many of the implementations above depend on the availability of external, third-party services their tests depend on the presence of Go build tags to run. They are :

| Implementation | Build tag |
| --- | --- |
| encoderfile:// | encoderfile |
| llamafile:// | llamafile |
| mlxclip:// | mlxclip |
| mobileclip:// | mobileclip |
| ollama:// | ollama |
| openclip:// | openclip |
