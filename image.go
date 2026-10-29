package embeddings

import (
	"bytes"
	"image"
	_ "image/jpeg"
	"image/png"
	"log/slog"
	"math"

	"golang.org/x/image/draw"
)

// ResizeIfOversized checks image bounds against a pixel cap and shrinks if necessary
func ResizeIfOversized(body []byte, max_px int) ([]byte, error) {

	cfg, _, err := image.DecodeConfig(bytes.NewReader(body))

	if err != nil {
		return body, err
	}

	total_px := cfg.Width * cfg.Height

	slog.Debug("Image size", "total pixels", total_px, "max pixels", max_px)

	if total_px <= max_px {
		return body, nil // Fits safely inside token boundaries
	}

	ratio := math.Sqrt(float64(max_px) / float64(total_px))

	new_w := int(float64(cfg.Width) * ratio)
	new_h := int(float64(cfg.Height) * ratio)

	slog.Debug("Resize image", "width", new_w, "height", new_h)

	im, _, err := image.Decode(bytes.NewReader(body))

	if err != nil {
		return body, err
	}

	new_im := image.NewRGBA(image.Rect(0, 0, new_w, new_h))
	draw.BiLinear.Scale(new_im, new_im.Bounds(), im, im.Bounds(), draw.Over, nil)

	var buf bytes.Buffer

	err = png.Encode(&buf, new_im)

	if err != nil {
		return body, err
	}

	return buf.Bytes(), nil
}
