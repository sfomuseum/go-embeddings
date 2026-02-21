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

	flag.StringVar(&client_uri, "client-uri", "", "...")

	flag.Parse()

	args := flag.Args()

	ctx := context.Background()

	cl, err := embeddings.NewEmbedder(ctx, client_uri)

	if err != nil {
		log.Fatal(err)
	}

	var embeddings_rsp []float32
	var embeddings_err error

	switch args[0] {
	case "text":

		var body string

		switch len(args) {
		case 2:

			switch args[1] {
			case "-":

				b, err := io.ReadAll(os.Stdin)

				if err != nil {
					log.Fatalf("Failed to read STDIN, %v", err)
				}

				body = string(b)
			default:

				b, err := os.ReadFile(args[1])

				if err != nil {
					log.Fatalf("Failed to read file, %v", err)
				}

				body = string(b)
			}

		default:
			body = strings.Join(args[1:], " ")
		}

		embeddings_rsp, embeddings_err = cl.Embeddings32(ctx, body)

	case "image":

		body, err := os.ReadFile(args[1])

		if err != nil {
			log.Fatalf("Failed to read file, %v", err)
		}

		embeddings_rsp, embeddings_err = cl.ImageEmbeddings32(ctx, body)
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
