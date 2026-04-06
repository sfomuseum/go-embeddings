// go:build siglip_python

package embeddings

import (
	"embed"
)

//go:embed siglip_py.txt
var siglip_fs embed.FS
