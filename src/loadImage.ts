import { createReadStream, createWriteStream, readFile } from "fs";
import { PNG } from 'pngjs';
import path from 'path';
import { Tensor4D, Tensor2D } from "@tensorflow/tfjs";
import * as tf from "@tensorflow/tfjs";

const colorType = 4;
const bitDepth = 8;

export const IMAGE_H = 28;
export const IMAGE_W = 28;
const IMAGE_SIZE = IMAGE_H * IMAGE_W;
const NUM_CLASSES = 10;
const NUM_DATASET_ELEMENTS = 65000;

const NUM_TRAIN_ELEMENTS = 55000;
const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

let imageBuffer = new Buffer([]);
let answerBuffer = new Buffer([]);

export function extractData() {
  return new Promise<{images: Tensor4D, answers: Tensor2D}>(resolve => {
    createReadStream(path.resolve(__dirname, '../mnist/mnist_images.png')).pipe(new PNG({
      colorType, bitDepth
    }))
      .on('parsed', function(this: PNG) {
        readFile(path.resolve(__dirname, '../mnist/mnist_labels_uint8'), (err, labels) => {
          // for (let index = 0; index < DATASET_ELEMENTS; index++) {
          //   // images[index] = [];
          //   // for (let offset = 0; offset < 784; offset++) {
          //   //   if (!images[index][Math.floor(offset / 28)]) images[index][Math.floor(offset / 28)] = [];
          //   //   images[index][Math.floor(offset / 28)][offset % 28] = this.data[offset * 4 + index * 784 * 4] / 255;
          //   // }
          //   const answer = Array.from(labels.subarray(index * 10, index * 10 + 10));
          //   answers[index] = answer;
          // }
          imageBuffer = this.data;
          const trainImagesArray = Array.from(this.data.subarray(0, IMAGE_SIZE * NUM_TRAIN_ELEMENTS));
          console.log("Created training images array.");
          const trainImages = tf.tensor4d(trainImagesArray, [trainImagesArray.length / 784, 28, 28, 1]);
          console.log("Created training images tensor.");

          const trainAnswersArray = labels.subarray(0, NUM_CLASSES * NUM_TRAIN_ELEMENTS);
          console.log("Created training answers array.");
          const trainAnswers = tf.tensor2d(
            trainAnswersArray, [trainAnswersArray.length / 10, 10]);
          console.log("Created training answers tensor.");
          resolve({images: trainImages, answers: trainAnswers});
        });
      });
  });
}

export function getTestData() {
  const testImagesArray = Array.from(imageBuffer.subarray(IMAGE_SIZE * NUM_TRAIN_ELEMENTS));
  console.log("Created testing images array.");
  const testImages = tf.tensor4d(testImagesArray, [testImagesArray.length / 784, 28, 28, 1]);
  console.log("Created testing images tensor.");
  const testAnswersArray = answerBuffer.subarray(NUM_CLASSES * NUM_TRAIN_ELEMENTS);
  console.log("Created testing answers array.");
  const testAnswers = tf.tensor2d(
    testAnswersArray, [testAnswersArray.length / 10, 10]);
  console.log("Created testing answers tensor.");
  return {
    images: testImages,
    answers: testAnswers
  }
}
