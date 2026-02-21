# go-mobileclip

Go package for deriving vector embeddings for text and images from a MobileCLIP "service".

## Motivation

This is a Go package for deriving vector embeddings for text and images from a MobileCLIP "service" where, for the time being, that "service" is expected to be a gRPC service as implemented by the [sfomuseum/swift-mobileclip](https://github.com/sfomuseum/swift-mobileclip) package.

In effect this is client-code, written in Go, for a service written in Swift.

## Documentation

`godoc` documentation is incomplete at this time.

## Usage

```
import (
	"context"
	"os"
	
	"github.com/sfomuseum/go-mobileclip"
)

func main() {

	body, _ := os.ReadAll("/path/to/image.png")
	
	cl, _ := mobileclip.NewEmbeddingsClient(ctx, "grpc://localhost:8080")

	req := &mobileclip.EmbeddingsRequest{
		Model: "s0",
		Body: body,
	}

	embeddings, _ := req.ComputeImageEmbeddings(ctx, req)
}
```

_Error handling removed for the sake of brevity._

Where `embeddings` is a struct of type `Embeddings`:

```
type Embeddings struct {
	Id         string    `json:"id,omitempty"`
	Embeddings []float32 `json:"embeddings"`
	Dimensions int32     `json:"dimensions"`
	Model      string    `json:"model"`
	Created    int64     `json:"created"`
}
```

## Tools

```
$> make cli
go build -mod vendor -ldflags="-s -w" -o bin/embeddings cmd/embeddings/main.go
```

### embeddings

Command-line tool for deriving text or image embeddings from a MobileCLIP "service". Results are written as a JSON-encoded string to STDOUT.

```
$> ./bin/embeddings -h
Command-line tool for deriving text or image embeddings from a MobileCLIP "service". Results are written as a JSON-encoded string to STDOUT.
Usage:
	./bin/embeddings [options] text|image arg(N) arg(N)

If the first argument is "image" then the body of the image to derive embeddings for will be read from the second argument.

If the first argument is "text" then the body of the text to derive embeddings for will be determined as follows: If there are only two arguments and the second argument is "-" then the body will be read from STDIN. Otherwise, if there are only two arguments the body of the text will be read from the file specified in the second argument. Finally, if there are more than two arguments then the body of the text will be the concatenation of the second to the last argument.

Valid options are:
  -client-uri string
    	A valid sfomuseum/go-mobileclip.EmbeddingsClient URI. (default "grpc://localhost:8080")
  -model string
    	The name of the MobileCLIP model to use to derive embeddings. Valid options are: s0, s1, s2, blt (default "s0")
```

#### embeddings text

```
$> echo "hello world" | ./bin/embeddings text -
{"embeddings":[-0.3161621,-0.1697998,1.4482422,0.04 ... and so on
```

#### embeddings image

```
$> ./bin/embeddings image ~/Desktop/test14.png 
{"embeddings":[-0.025161743,-0.027786255,-0.0014038086,0.0035247803,-0.006515503,-0.012527466,-0.041168213 ... and so on
```