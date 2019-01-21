import { extractData, getTestData } from "./loadImage";
import { writeFileSync } from "fs";
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import { Model, Tensor4D, Tensor2D } from "@tensorflow/tfjs";

async function start() {
  const model = newConvModel();

  const data = await extractData();
  
  train(model, data);
}

function newConvModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 3,
    filters: 16,
    activation: 'relu'
  }));

  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));

  model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));

  model.add(tf.layers.maxPooling2d({poolSize: 2, strides: 2}));
  
  model.add(tf.layers.conv2d({kernelSize: 3, filters: 32, activation: 'relu'}));

  model.add(tf.layers.flatten({}));

  model.add(tf.layers.dense({units: 64, activation: 'relu'}));

  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));

  return model;
}

function newDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape: [28, 28, 1]}));
  model.add(tf.layers.dense({units: 42, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  return model;
}

async function train(model: Model, data: {images: Tensor4D, answers: Tensor2D}) {
  console.log("Training model...");
  const LEARNING_RATE = 0.01;
  const optimizer = 'rmsprop';
  const batchSize = 320;
  const validationSplit = 0.15;
  const trainEpochs = 3;
  let trainBatchCount = 0;

  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  let valAcc: number = 0;
  await model.fit(data.images, data.answers), {
    batchSize,
    validationSplit,
    epochs: trainEpochs,
    onEpochEnd: async (epoch: number, logs: any) => {
      valAcc = logs.val_acc;
    }
  };

  const testData = getTestData();
  const testResult = model.evaluate(testData.images, testData.answers);
  const testAccPercent = (<any>testResult)[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  console.log(
      `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
      `Final test accuracy: ${testAccPercent.toFixed(1)}%`);
}

start();
