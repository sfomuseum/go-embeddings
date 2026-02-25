package embeddings

import (
	"flag"

	"github.com/sfomuseum/go-flags/flagset"
)

var client_uri string
var precision int

func DefaultFlagSet() *flag.FlagSet {

	fs := flagset.NewFlagSet("embeddings")

	fs.StringVar(&client_uri, "client-uri", "null://", "...")
	fs.IntVar(&precision, "precision", 32, "...")

	return fs
}
