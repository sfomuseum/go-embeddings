package main

import (
	"context"
	"log"

	"github.com/sfomuseum/go-embeddings/app/embeddings"
)

func main() {

	ctx := context.Background()
	err := embeddings.Run(ctx)

	if err != nil {
		log.Fatal(err)
	}
}
