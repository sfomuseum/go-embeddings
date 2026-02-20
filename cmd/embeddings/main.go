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

	// START OF make me a function

	var body string

	switch len(args) {
	case 1:

		switch args[0] {
		case "-":

			b, err := io.ReadAll(os.Stdin)

			if err != nil {
				log.Fatalf("Failed to read STDIN, %v", err)
			}

			body = string(b)
		default:

			b, err := os.ReadFile(args[0])

			if err != nil {
				log.Fatalf("Failed to read file, %v", err)
			}

			body = string(b)
		}

	default:
		body = strings.Join(args[0:], " ")
	}

	// END OF make me a function

	rsp, err := cl.Embeddings32(ctx, body)

	if err != nil {
		log.Fatal(err)
	}

	enc := json.NewEncoder(os.Stdout)
	err = enc.Encode(rsp)

	if err != nil {
		log.Fatal(err)
	}
}
