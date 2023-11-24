import numpy as np
from scipy import sparse


class HaarWaveletImageCompressor:
    def __init__(self, threshold):
        self.threshold = threshold

    def compress(self, image, file_name_to_store):
        print('Image:')
        print(image)
        encoded_image = haar_encode(image)
        print('Encoded image:')
        print(encoded_image)
        truncated_encoded_image = np.where(np.abs(encoded_image) < self.threshold, 0, encoded_image)
        print('Truncated encoded image:')
        print(truncated_encoded_image)
        int_truncated_encoded_image = np.round(truncated_encoded_image).astype(np.int8)
        sparse_repr = sparse.csr_matrix(int_truncated_encoded_image)
        sparse.save_npz(file_name_to_store + '.npz', sparse_repr)

    def decompress(self, file_path):
        sparse_encoded_matrix = sparse.load_npz(file_path + '.npz')
        encoded_matrix = sparse_encoded_matrix.toarray()
        decoded_matrix = haar_decode(encoded_matrix)
        decompressed_image = np.clip(decoded_matrix, 0, 255).astype(np.uint8)
        print('Decoded image:')
        print(decompressed_image)
        return decompressed_image


def haar_encode(mat):
    number_of_steps = int(np.ceil(np.log2(mat.shape[0])))
    row_encoder = haar_transform(number_of_steps)
    return np.matmul(np.matmul(np.transpose(row_encoder), mat), row_encoder)


def haar_decode(mat):
    number_of_steps = int(np.ceil(np.log2(mat.shape[0])))
    row_decoder = np.linalg.inv(haar_transform(number_of_steps))
    return np.matmul(np.matmul(np.transpose(row_decoder), mat), row_decoder)


def haar_transform(number_of_steps):
    transform = np.eye(2 ** number_of_steps)
    for step_number in range(number_of_steps):
        transform = np.matmul(transform, haar_step(step_number, number_of_steps))
    return transform


def haar_step(step_number, number_of_steps):
    transform = np.zeros((2 ** number_of_steps, 2 ** number_of_steps))
    # Averages (1)
    for j in range(2 ** (number_of_steps - step_number - 1)):
        transform[2 * j, j] = 0.5
        transform[2 * j + 1, j] = 0.5
    # Details (2)
    offset = 2 ** (number_of_steps - step_number - 1)
    for j in range(offset):
        transform[2 * j, offset + j] = 0.5
        transform[2 * j + 1, offset + j] = -0.5
    # Identity (3)
    for j in range(2 ** (number_of_steps - step_number), 2 ** number_of_steps):
        transform[j, j] = 1
    return transform




