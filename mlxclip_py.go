// go:build mlxclip_python

package embeddings

import (
	"embed"
)

//go:embed mlxclip_py.txt
var f embed.FS
