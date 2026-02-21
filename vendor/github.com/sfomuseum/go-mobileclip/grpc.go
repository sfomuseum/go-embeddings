package mobileclip

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/url"
	"strconv"

	mobileclip_grpc "github.com/sfomuseum/go-mobileclip/grpc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

type GrpcEmbeddingsClient struct {
	conn   *grpc.ClientConn
	client mobileclip_grpc.EmbeddingsServiceClient
}

func init() {
	ctx := context.Background()
	RegisterEmbeddingsClient(ctx, "grpc", NewGrpcEmbeddingsClient)
}

func NewGrpcEmbeddingsClient(ctx context.Context, uri string) (EmbeddingsClient, error) {

	u, err := url.Parse(uri)

	if err != nil {
		return nil, fmt.Errorf("Failed to parse URI, %w", err)
	}

	q := u.Query()

	q_tls_cert := q.Get("tls-certificate")
	q_tls_key := q.Get("tls-key")
	q_tls_ca := q.Get("tls-ca-certificate")
	q_tls_insecure := q.Get("tls-insecure")

	addr := u.Host

	opts := make([]grpc.DialOption, 0)

	if q_tls_cert != "" && q_tls_key != "" {

		cert, err := tls.LoadX509KeyPair(q_tls_cert, q_tls_key)

		if err != nil {
			return nil, fmt.Errorf("Failed to load TLS pair, %w", err)
		}

		tls_config := &tls.Config{
			Certificates: []tls.Certificate{cert},
		}

		if q_tls_ca != "" {

			ca_cert, err := ioutil.ReadFile(q_tls_ca)

			if err != nil {
				return nil, fmt.Errorf("Failed to create CA certificate, %w", err)
			}

			cert_pool := x509.NewCertPool()

			ok := cert_pool.AppendCertsFromPEM(ca_cert)

			if !ok {
				return nil, fmt.Errorf("Failed to append CA certificate, %w", err)
			}

			tls_config.RootCAs = cert_pool

		} else if q_tls_insecure != "" {

			v, err := strconv.ParseBool(q_tls_insecure)

			if err != nil {
				return nil, fmt.Errorf("Failed to parse ?tls-insecure= parameter, %w", err)
			}

			tls_config.InsecureSkipVerify = v
		}

		tls_credentials := credentials.NewTLS(tls_config)
		opts = append(opts, grpc.WithTransportCredentials(tls_credentials))

	} else {
		opts = append(opts, grpc.WithInsecure())
	}

	conn, err := grpc.NewClient(addr, opts...)

	if err != nil {
		return nil, fmt.Errorf("Failed to dial '%s', %w", addr, err)
	}

	client := mobileclip_grpc.NewEmbeddingsServiceClient(conn)

	e := &GrpcEmbeddingsClient{
		conn:   conn,
		client: client,
	}

	return e, nil
}

func (e *GrpcEmbeddingsClient) ComputeTextEmbeddings(ctx context.Context, req *EmbeddingsRequest) (*Embeddings, error) {

	grpc_req := &mobileclip_grpc.EmbeddingsRequest{
		Id:    req.Id,
		Body:  req.Body,
		Model: req.Model,
	}

	rsp, err := e.client.ComputeTextEmbeddings(ctx, grpc_req)

	if err != nil {
		return nil, fmt.Errorf("Failed to derive embeddings, %w", err)
	}

	return embeddingsFromGrpcEmbeddingsResponse(rsp), nil
}

func (e *GrpcEmbeddingsClient) ComputeImageEmbeddings(ctx context.Context, req *EmbeddingsRequest) (*Embeddings, error) {

	grpc_req := &mobileclip_grpc.EmbeddingsRequest{
		Id:    req.Id,
		Body:  req.Body,
		Model: req.Model,
	}

	rsp, err := e.client.ComputeImageEmbeddings(ctx, grpc_req)

	if err != nil {
		return nil, fmt.Errorf("Failed to derive embeddings, %w", err)
	}

	return embeddingsFromGrpcEmbeddingsResponse(rsp), nil
}

func embeddingsFromGrpcEmbeddingsResponse(rsp *mobileclip_grpc.EmbeddingsResponse) *Embeddings {

	e := &Embeddings{
		Id:         rsp.Id,
		Model:      rsp.Model,
		Dimensions: rsp.Dimensions,
		Embeddings: rsp.Embeddings,
		Created:    rsp.Created,
	}

	return e
}
