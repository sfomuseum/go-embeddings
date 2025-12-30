package main

import (
	"context"
	"encoding/json"
	"log"
	"os"
	"strings"

	"github.com/sfomuseum/go-embeddings"
	"github.com/sfomuseum/go-flags/flagset"
)

func main() {

	var embedder_uri string
	var mode string
	var path string

	fs := flagset.NewFlagSet("embeddings")
	fs.StringVar(&embedder_uri, "embedder-uri", "null://", "...")
	fs.StringVar(&mode, "mode", "text", "...")
	fs.StringVar(&path, "path", "", "...")

	flagset.Parse(fs)

	ctx := context.Background()

	emb, err := embeddings.NewEmbedder(ctx, embedder_uri)

	if err != nil {
		log.Fatalf("Failed to create embedder, %v", err)
	}

	var rsp []float64

	switch mode {
	case "text":

		txt := strings.Join(fs.Args(), " ")
		rsp, err = emb.Embeddings(ctx, txt)

	case "image":

		body, err := os.ReadFile(path)

		if err != nil {
			log.Fatalf("Failed to read path, %v", err)
		}

		rsp, err = emb.ImageEmbeddings(ctx, body)
	default:
		log.Fatalf("Invalid mode")
	}

	if err != nil {
		log.Fatalf("Failed to derive embeddings, %v", err)
	}

	if len(rsp) == 0 {
		log.Fatalf("Empty embedding")
	}

	enc := json.NewEncoder(os.Stdout)
	err = enc.Encode(rsp)

	if err != nil {
		log.Fatal(err)
	}
}
