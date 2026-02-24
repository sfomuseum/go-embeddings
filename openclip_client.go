//go:build openclip

package embeddings

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

type openCLIPImageDataEmbeddingRequest struct {
	Id   int64  `json:"id"`
	Data string `json:"data"`
}

type openCLIPEmbeddingRequest struct {
	Content   string                               `json:"content,omitempty"`
	ImageData []*openCLIPImageDataEmbeddingRequest `json:"image_data,omitempty"`
}

type openCLIPEmbeddingResponse struct {
	Embeddings []float64 `json:"embedding,omitempty"`
}

type openCLIPClient struct {
	client *http.Client
	host   string
	port   string
	tls    bool
}

func newOpenCLIPClient(ctx context.Context, uri string) (*openCLIPClient, error) {

	host := "127.0.0.1"
	port := "5000"
	tls := false

	u, err := url.Parse(uri)

	if err != nil {
		return nil, err
	}

	if u.Host != "" {
		host = u.Host

		parts := strings.Split(host, ":")

		if len(parts) < 1 {
			return nil, fmt.Errorf("Failed to parse host component of URI")
		}

		host = parts[0]
	}

	if u.Port() != "" {
		port = u.Port()
	}

	slog.Debug("URL", "host", host, "port", port)

	q := u.Query()

	if q.Has("tls") {

		v, err := strconv.ParseBool("tls")

		if err != nil {
			return nil, fmt.Errorf("Invalid ?tls= parameter, %w", err)
		}

		tls = v
	}

	http_cl := &http.Client{}

	cl := &openCLIPClient{
		client: http_cl,
		host:   host,
		port:   port,
		tls:    tls,
	}

	return cl, nil
}

func (e *openCLIPClient) embeddings(ctx context.Context, openclip_req *openCLIPEmbeddingRequest) (*openCLIPEmbeddingResponse, error) {

	u := url.URL{}
	u.Scheme = "http"
	u.Host = fmt.Sprintf("%s:%s", e.host, e.port)

	if len(openclip_req.ImageData) > 0 {
		u.Path = "/embeddings/image"
	} else {
		u.Path = "/embeddings"
	}

	if e.tls {
		u.Scheme = "https"
	}

	endpoint := u.String()

	enc_msg, err := json.Marshal(openclip_req)

	if err != nil {
		return nil, fmt.Errorf("Failed to encode message, %w", err)
	}

	br := bytes.NewReader(enc_msg)

	req, err := http.NewRequestWithContext(ctx, "POST", endpoint, br)

	if err != nil {
		return nil, fmt.Errorf("Failed to create new request, %w", err)
	}

	req.Header.Set("Content-type", "application/json")

	rsp, err := e.client.Do(req)

	if err != nil {
		return nil, fmt.Errorf("Failed to execute request, %w", err)
	}

	defer rsp.Body.Close()

	if rsp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Embeddings request failed %d: %s", rsp.StatusCode, rsp.Status)
	}

	var openclip_rsp *openCLIPEmbeddingResponse

	dec := json.NewDecoder(rsp.Body)
	err = dec.Decode(&openclip_rsp)

	if err != nil {
		return nil, fmt.Errorf("Failed to unmarshal embeddings, %w", err)
	}

	return openclip_rsp, nil
}

/*

"""
python -m venv openclip
cd openclip/
bash bin/activate
bin/pip install flask
bin/pip install open_clip_torch
bin/pip install Pillow

Then, assuming the code below is in a file called openclip_server.py:

bin/flask --app openclip_server run
"""

import tempfile
import base64
import os

from flask import Flask, request, jsonify
import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
model.eval()

tokenizer = open_clip.get_tokenizer('ViT-B-32')

app = Flask(__name__)

@app.route("/embeddings", methods=['POST'])
def embeddings():
    
    req = request.json
    text = tokenizer([ req["content"] ])

    with torch.no_grad(), torch.autocast("cuda"):
        text_features = model.encode_text(text)
        embeddings = text_features.tolist()
        return jsonify({"embedding": embeddings[0]})

@app.route("/embeddings/image", methods=['POST'])
def embeddings_image():
    
    req = request.json
    body = base64.b64decode(req["image_data"][0]["data"])

    with tempfile.NamedTemporaryFile(delete_on_close=False, mode="wb") as wr:

        wr.write(body)
        wr.close()

        image = preprocess(Image.open(wr.name)).unsqueeze(0)
        os.remove(wr.name)
        
        with torch.no_grad(), torch.autocast("cuda"):
            image_features = model.encode_image(image)
            embeddings = image_features.tolist()
            return jsonify({"embedding": embeddings[0]})

*/
