# Implementation of DCT and Haar wavelet gray image compression
## Comparision on 512x512 gray Lena image, with size 259KB in .gif format.

| Compression | Size | Compression ratio | PSMR | Subjective quality |
|-------------|------|-------------------|------|---------------------|
| DCT quality = 90 | 84KB | 3 | 40.62 | undetectable diff |
| DCT quality = 50 | 49KB | 5.3 | 36.15 | almost undetectable diff |
| Haar threshold = 0.5 | 160KB | 1.6 | 38.98 | visible pattern (blockines, but bearable) |
| Haar threshold = 1 | 133KB | 1.95 | 36.37 | visible pattern (blockines, but bearable) |
| Haar threshold = 3 | 49KB | 5.3 | 30.47 | very blocky |
