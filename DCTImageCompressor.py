import numpy as np
from dahuffman import HuffmanCodec
import pickle


class DCTImageCompressor:
    def __init__(self, quality = 50):
        self.block_size = 8
        self.dct_matrix = compute_dct_matrix(self.block_size)
        self.dct_matrix_transposed = np.transpose(self.dct_matrix)
        # 50% Quality JPEG Standard Quantization Matrix
        quantization_matrix_50q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                                             [12, 12, 14, 19, 26, 58, 60, 55],
                                             [14, 13, 16, 24, 40, 57, 69, 56],
                                             [14, 17, 22, 29, 51, 87, 80, 62],
                                             [18, 22, 37, 56, 68, 109, 103, 77],
                                             [24, 35, 55, 64, 81, 104, 113, 92],
                                             [49, 64, 78, 87, 103, 121, 120, 101],
                                             [72, 92, 95, 98, 112, 100, 103, 99]])
        # 90% Quality JPEG Standard Quantization Matrix
        quantization_matrix_90q = np.array([[3, 2, 2, 3, 5, 8, 10, 12],
                                            [2, 2, 3, 4, 5, 12, 12, 11],
                                            [3, 3, 3, 5, 8, 11, 14, 11],
                                            [3, 3, 4, 6, 10, 17, 16, 12],
                                            [4, 4, 7, 11, 14, 22, 21, 15],
                                            [5, 7, 11, 13, 16, 12, 23, 18],
                                            [10, 13, 16, 17, 21, 24, 24, 21],
                                            [14, 18, 19, 20, 22, 20, 20, 20]])
        if quality == 50:
            self.quantization_matrix = quantization_matrix_50q
        elif quality == 90:
            self.quantization_matrix = quantization_matrix_90q
        elif quality > 50:
            self.quantization_matrix = np.round(quantization_matrix_50q * (quality / 50.0)).astype(int)
        else:
            self.quantization_matrix = np.round(quantization_matrix_50q * (50.0 / quality)).astype(int)

    def compress(self, image, file_name_to_store, codec_name_to_store):
        rows = image.shape[0]
        columns = image.shape[1]
        zigzag_arrays = []
        for i in range(0, rows, self.block_size):
            for j in range(0, columns, self.block_size):
                shifted_img = image[i:i + self.block_size, j:j + self.block_size].astype(np.int16) - 128
                dct_block = np.matmul(np.matmul(self.dct_matrix, shifted_img), self.dct_matrix_transposed)
                dct_quantized = np.round(dct_block / self.quantization_matrix).astype(np.int16)
                if i + self.block_size >= rows and j + self.block_size >= columns:
                    print('Image snippet:')
                    print(image[i:i + self.block_size, j:j + self.block_size])
                    print('Image snipspet DCT:')
                    print(dct_block)
                    print('Image snippet DCT quantized:')
                    print(dct_quantized)
                zigzag_arrays.append(zigzag(dct_quantized))

        flattened_zigzag = np.concatenate(zigzag_arrays)
        data = np.concatenate([np.array([rows, columns], dtype=np.int16), flattened_zigzag])
        huffman_codec = HuffmanCodec.from_data(data)
        encoded_data = huffman_codec.encode(data)

        with open(file_name_to_store + '.bin', 'wb') as file:
            file.write(encoded_data)

        with open(codec_name_to_store + '.bin', 'wb') as file:
            pickle.dump(huffman_codec, file)

    def decompress(self, file_path, codec_path):
        with open(file_path+'.bin', 'rb') as file:
            encoded_data = file.read()
        with open(codec_path + '.bin', 'rb') as file:
            huffman_codec = pickle.load(file)
        decoded_data = huffman_codec.decode(encoded_data)
        rows = decoded_data[0]
        columns = decoded_data[1]
        decompressed_image = np.empty((rows, columns), dtype=np.uint8)
        count = 0
        squared_block_size = self.block_size ** 2
        for i in range(0, rows, self.block_size):
            for j in range(0, columns, self.block_size):
                start_index = 2 + i * columns + j * self.block_size
                zigzag_block = decoded_data[start_index: start_index + squared_block_size]
                dct_quantized = from_zigzag(zigzag_block, self.block_size, self.block_size)
                dct_block = dct_quantized * self.quantization_matrix
                reconstructed_block = np.matmul(np.matmul(self.dct_matrix_transposed, dct_block),
                                                            self.dct_matrix) + 128
                reconstructed_block = np.clip(reconstructed_block, 0, 255).astype(np.uint8)
                decompressed_image[i:i + self.block_size, j:j + self.block_size] = reconstructed_block
                count += 1
                if i + self.block_size >= rows and j + self.block_size >= columns:
                    print('Read from disk:')
                    print(dct_quantized)
                    print('Dequantized snippet DCT:')
                    print(dct_block)
                    print('Reconstructed image snippet:')
                    print(reconstructed_block)
        return decompressed_image


def compute_dct_matrix(n):
    dct_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == 0:
                dct_mat[i, j] = 1 / np.sqrt(n)
            else:
                dct_mat[i, j] = np.sqrt(2 / n) * np.cos((2 * j + 1) * i * np.pi / (2 * n))
    return dct_mat


def zigzag(matrix):
    m, n = matrix.shape
    result = np.zeros(m * n, dtype=matrix.dtype)
    result[0] = matrix[0][0]
    k = 1
    i = j = 0
    while (k < n * m):
        while i >= 1 and j < n - 1:
            i -= 1
            j += 1
            result[k] = matrix[i][j]
            k += 1
        if j < n - 1:
            j += 1
            result[k] = matrix[i][j]
            k += 1
        elif i < m - 1:
            i += 1
            result[k] = matrix[i][j]
            k += 1
        while i < m - 1 and j >= 1:
            i += 1
            j -= 1
            result[k] = matrix[i][j]
            k += 1
        if i < m - 1:
            i += 1
            result[k] = matrix[i][j]
            k += 1
        elif j < n - 1:
            j += 1
            result[k] = matrix[i][j]
            k += 1
    return result


def from_zigzag(zigzag_sequence, rows, columns):
    matrix = np.zeros((rows, columns))
    m, n = matrix.shape
    matrix[0][0] = zigzag_sequence[0]
    k = 1
    i = j = 0
    while (k < n * m):
        while i >= 1 and j < n - 1:
            i -= 1
            j += 1
            matrix[i][j] = zigzag_sequence[k]
            k += 1
        if j < n - 1:
            j += 1
            matrix[i][j] = zigzag_sequence[k]
            k += 1
        elif i < m - 1:
            i += 1
            matrix[i][j] = zigzag_sequence[k]
            k += 1
        while i < m - 1 and j >= 1:
            i += 1
            j -= 1
            matrix[i][j] = zigzag_sequence[k]
            k += 1
        if i < m - 1:
            i += 1
            matrix[i][j] = zigzag_sequence[k]
            k += 1
        elif j < n - 1:
            j += 1
            matrix[i][j] = zigzag_sequence[k]
            k += 1
    return matrix