GOMOD=$(shell test -f "go.work" && echo "readonly" || echo "vendor")
LDFLAGS=-s -w

vuln:
	govulncheck -show verbose ./...

TAGS=null

cli:
	go build -tags $(TAGS) -mod $(GOMOD) -ldflags="$(LDFLAGS)" -o bin/embeddings cmd/embeddings/main.go
