package main

import (
	"context"
	"encoding/json"
	"flag"
	"io"
	"log"
	"os"
	"strings"

	"github.com/sfomuseum/go-embeddings"
)

func main() {

	var client_uri string
	// var precision int

	flag.StringVar(&client_uri, "client-uri", "null://", "...")
	// flag.IntVar(&precision, "precision", 32, "...")

	flag.Parse()

	args := flag.Args()

	ctx := context.Background()

	cl, err := embeddings.NewEmbedder32(ctx, client_uri)

	if err != nil {
		log.Fatal(err)
	}

	var embeddings_rsp any
	var embeddings_err error

	switch args[0] {
	case "text":

		var body []byte

		switch len(args) {
		case 2:

			switch args[1] {
			case "-":

				b, err := io.ReadAll(os.Stdin)

				if err != nil {
					log.Fatalf("Failed to read STDIN, %v", err)
				}

				body = b
			default:

				b, err := os.ReadFile(args[1])

				if err != nil {
					log.Fatalf("Failed to read file, %v", err)
				}

				body = b
			}

		default:
			body = []byte(strings.Join(args[1:], " "))
		}

		req := &embeddings.EmbeddingsRequest{
			Body: body,
		}

		embeddings_rsp, embeddings_err = cl.TextEmbeddings(ctx, req)

	case "image":

		body, err := os.ReadFile(args[1])

		if err != nil {
			log.Fatalf("Failed to read file, %v", err)
		}

		req := &embeddings.EmbeddingsRequest{
			Body: body,
		}

		embeddings_rsp, embeddings_err = cl.ImageEmbeddings(ctx, req)
	default:
		log.Fatal("Invalid or unsuported action")
	}

	if embeddings_err != nil {
		log.Fatalf("Failed to derive embeddings, %v", embeddings_err)
	}

	enc := json.NewEncoder(os.Stdout)
	err = enc.Encode(embeddings_rsp)

	if err != nil {
		log.Fatalf("Failed to encode embeddings, %v", err)
	}
}
