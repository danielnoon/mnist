import { createReadStream, createWriteStream, readFile } from "fs";
import { PNG } from 'pngjs';
import path from 'path';
import { Tensor4D, Tensor2D, Tensor } from "@tensorflow/tfjs";
import * as tf from "@tensorflow/tfjs";
import chokidar from 'chokidar';

const colorType = 4;
const bitDepth = 8;

export const IMAGE_H = 28;
export const IMAGE_W = 28;
export const IMAGE_SIZE = IMAGE_H * IMAGE_W;
export const NUM_CLASSES = 10;
export const NUM_DATASET_ELEMENTS = 65000;
export const NUM_TRAIN_ELEMENTS = 55000;
export const NUM_TEST_ELEMENTS = NUM_DATASET_ELEMENTS - NUM_TRAIN_ELEMENTS;

export class MNISTData {
  imageBuffer = new Array<number>(0);
  answerBuffer = new Array<number>(0);
  currentBatch = 0;

  load() {
    return new Promise<{images: Tensor4D, answers: Tensor2D}>(resolve => {
      createReadStream(path.resolve(__dirname, '../mnist/mnist_images.png')).pipe(new PNG({
        colorType, bitDepth
      }))
        .on('parsed', (data) => {
          readFile(path.resolve(__dirname, '../mnist/mnist_labels_uint8'), (err, labels) => {
            this.imageBuffer = Array.from(data.subarray(0, NUM_DATASET_ELEMENTS * IMAGE_SIZE * 4)
              .filter((value, index) => index % 4 === 0))
              .map(val => val / 255);
            
            this.answerBuffer = Array.from(labels.subarray(0, NUM_DATASET_ELEMENTS * NUM_CLASSES));

            resolve();
          });
        });
    });
  }

  listen(callback: (img: Tensor4D) => void) {
    this.readImage().then(img => callback(img));
    chokidar.watch(path.resolve(__dirname, '../number.png')).on('change', async () => {
      setTimeout(() => this.readImage().then(img => callback(img)), 500);
    });
  }

  readImage() {
    return new Promise<Tensor4D>(resolve => {
      createReadStream(path.resolve(__dirname, '../number.png')).pipe(new PNG({
        colorType, bitDepth
      }))
        .on('parsed', (data) => {
          const img = Array.from(data)
              .filter((value, index) => index % 4 === 0)
              .map(val => val / 255);
          
          resolve(tf.tensor4d(img, [1, IMAGE_H, IMAGE_W, 1]));
        });
    });
  }

  writeNumber(img: Tensor4D, labels: Tensor) {
    const png = new PNG({width:28,height:28});
    const arr = Array.from(img.dataSync().subarray(0, 784)).map(pixel => pixel * 255);
    const classes = Array.from(labels.dataSync()).slice(0, 10);
    let label = 0;
    let max = 0;
    for (let i = 0; i < 10; i++) {
      if (classes[i] > max) {
        max = classes[i];
        label = i;
      }
    }
    const pixels: number[] = [];
    for (let i = 0; i < 784; i++) {
      pixels[i * 4] = arr[i];
      pixels[i * 4 + 1] = arr[i];
      pixels[i * 4 + 2] = arr[i];
      pixels[i * 4 + 3] = 0;
    }
    png.data = Buffer.from(pixels);
    png.pack().pipe(createWriteStream(path.resolve(__dirname, `../mnist/pngs/${label}.${Math.floor(Math.random() * 1000)}.png`)));
  }

  getNextBatch(size: number) {
    const inputArr = this.imageBuffer.slice(this.currentBatch * IMAGE_SIZE * size, this.currentBatch * size * IMAGE_SIZE + size * IMAGE_SIZE);
    const xs = tf.tensor4d(inputArr, [inputArr.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);

    const answers = this.answerBuffer.slice(this.currentBatch * size * NUM_CLASSES, this.currentBatch * size * NUM_CLASSES + size * NUM_CLASSES);
    const labels = tf.tensor2d(answers, [answers.length / NUM_CLASSES, NUM_CLASSES]);

    this.currentBatch++;

    return {xs, labels};
  }

  getTestData(sample: number) {
    console.log(this.imageBuffer.length);
    const start = NUM_TRAIN_ELEMENTS * IMAGE_SIZE;
    const end = start + sample * IMAGE_SIZE;
    
    const startA = NUM_TRAIN_ELEMENTS * NUM_CLASSES;
    const endA = startA + sample * NUM_CLASSES;

    const inputArr = this.imageBuffer.slice(start);
    console.log("INPUT ARR: ", inputArr.length);
    const xs = tf.tensor4d(inputArr, [inputArr.length / IMAGE_SIZE, IMAGE_H, IMAGE_W, 1]);

    const answers = this.answerBuffer.slice(startA);
    const labels = tf.tensor2d(answers, [answers.length / NUM_CLASSES, NUM_CLASSES]);

    return {xs, labels};
  }
}
