package embeddings

import (
	"flag"
	"fmt"
	"os"

	"github.com/sfomuseum/go-flags/flagset"
)

var client_uri string
var precision int
var model string
var verbose bool

func DefaultFlagSet() *flag.FlagSet {

	fs := flagset.NewFlagSet("embeddings")

	fs.StringVar(&client_uri, "client-uri", "null://", "A registered sfomuseum/go-embeddings.Embedder[T] URI.")
	fs.StringVar(&model, "model", "", "An optional model to specify when generating embeddings.")
	fs.IntVar(&precision, "precision", 32, "The float-precision to use to for the embeddings that are returned.")
	fs.BoolVar(&verbose, "verbose", false, "Enable verbose (debug) logging.")

	fs.Usage = func() {
		fmt.Fprintf(os.Stderr, "Derive vector embeddings for a text string or image file.\n")
		fmt.Fprintf(os.Stderr, "Usage:\n\t%s [options] [text|image] arg(N) arg(N)\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Valid options are:\n")
		fs.PrintDefaults()
	}

	return fs
}
