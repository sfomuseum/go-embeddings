//go:build openclip_server

package embeddings

import (
	"context"
	"embed"
)

//go:embed openclip_server.txt
var openclip_fs embed.FS

func StartOpenCLIPServer(ctx context.Context) error {
	return NotImplemented
}
