package embeddings

import (
	"context"
	"embed"
)

//go:embed openclip_server.txt
var f embed.FS

func StartOpenCLIPServer(ctx context.Context) error {
	return NotImplemented
}	
