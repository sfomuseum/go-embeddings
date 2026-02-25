package embeddings

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	sfom_embeddings "github.com/sfomuseum/go-embeddings"
	"github.com/sfomuseum/go-flags/flagset"
)

func Run(ctx context.Context) error {
	fs := DefaultFlagSet()
	return RunWithFlagSet(ctx, fs)
}

func RunWithFlagSet(ctx context.Context, fs *flag.FlagSet) error {

	flagset.Parse(fs)
	args := fs.Args()

	action := args[0]

	var embeddings_req *sfom_embeddings.EmbeddingsRequest
	var embeddings_rsp any
	var embeddings_err error

	var body []byte

	switch action {
	case "text":

		switch len(args) {
		case 2:

			switch args[1] {
			case "-":

				b, err := io.ReadAll(os.Stdin)

				if err != nil {
					return fmt.Errorf("Failed to read STDIN, %v", err)
				}

				body = b
			default:

				b, err := os.ReadFile(args[1])

				if err != nil {
					return fmt.Errorf("Failed to read file, %v", err)
				}

				body = b
			}

		default:
			body = []byte(strings.Join(args[1:], " "))
		}

		embeddings_req = &sfom_embeddings.EmbeddingsRequest{
			Body: body,
		}

	case "image":

		body, err := os.ReadFile(args[1])

		if err != nil {
			return fmt.Errorf("Failed to read file, %v", err)
		}

		embeddings_req = &sfom_embeddings.EmbeddingsRequest{
			Body: body,
		}

	default:
		return fmt.Errorf("Invalid or unsuported action")
	}

	switch precision {
	case 32:

		cl, err := sfom_embeddings.NewEmbedder32(ctx, client_uri)

		if err != nil {
			return fmt.Errorf("Failed to create embedder, %w", err)
		}

		switch action {
		case "text":
			embeddings_rsp, embeddings_err = cl.TextEmbeddings(ctx, embeddings_req)
		case "image":
			embeddings_rsp, embeddings_err = cl.ImageEmbeddings(ctx, embeddings_req)
		default:
			return fmt.Errorf("Invalid or unsupported action")
		}

	case 64:

		cl, err := sfom_embeddings.NewEmbedder64(ctx, client_uri)

		if err != nil {
			return fmt.Errorf("Failed to create embedder, %w", err)
		}

		switch action {
		case "text":
			embeddings_rsp, embeddings_err = cl.TextEmbeddings(ctx, embeddings_req)
		case "image":
			embeddings_rsp, embeddings_err = cl.ImageEmbeddings(ctx, embeddings_req)
		default:
			return fmt.Errorf("Invalid or unsupported action")
		}

	default:
		return fmt.Errorf("Invalid or unsupported precision")
	}

	if embeddings_err != nil {
		return fmt.Errorf("Failed to derive embeddings, %v", embeddings_err)
	}

	enc := json.NewEncoder(os.Stdout)
	err := enc.Encode(embeddings_rsp)

	if err != nil {
		return fmt.Errorf("Failed to encode embeddings, %v", err)
	}

	return nil
}
