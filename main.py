import cv2
import numpy as np

from DCTImageCompressor import DCTImageCompressor
from HaarWaveletImageCompressor import HaarWaveletImageCompressor
from psnr import psnr

if __name__ == '__main__':
    cap = cv2.VideoCapture("lena_gray.gif")
    ret, image = cap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cap.release()
    test_image = np.array(
        [[62, 55, 55, 54, 49, 48, 47, 55],
         [62, 57, 54, 52, 48, 47, 48, 53],
         [61, 60, 52, 49, 48, 47, 49, 54],
         [63, 61, 60, 60, 63, 65, 68, 65],
         [67, 67, 70, 74, 79, 85, 91, 92],
         [82, 95, 101, 106, 114, 115, 112, 117],
         [96, 111, 115, 119, 128, 128, 130, 127],
         [109, 121, 127, 133, 139, 141, 140, 133]]
    )
    if ret:
        print('DCT:')
        quality = 50
        dct_compressor = DCTImageCompressor(quality=quality)
        image_file_name = 'lena_gray_dct_' + str(quality)
        codec_file_name = 'huffman_codec_' + str(quality)
        dct_compressor.compress(image, image_file_name, codec_file_name)
        dct_img = dct_compressor.decompress(image_file_name, codec_file_name)
        print('Haar:')
        threshold = 1
        haar_compressor = HaarWaveletImageCompressor(threshold=threshold)
        image_file_name = 'lena_gray_haar_' + str(threshold)
        haar_compressor.compress(image, image_file_name)
        haar_img = haar_compressor.decompress(image_file_name)
        print('PSNR:')
        print('DCT PSNR: ', psnr(image, dct_img))
        print('Haar PSNR: ', psnr(image, haar_img))
        result = np.concatenate((image, dct_img, haar_img), axis=1)
        cv2.imwrite('dct_img_' + str(quality) + '.png', dct_img)
        cv2.imwrite('haar_img_' + str(threshold) + '.png', haar_img)
        cv2.imshow('', result)
        cv2.waitKey(0)

