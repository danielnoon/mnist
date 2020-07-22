import { MNISTData } from "./loadImage";
import { writeFileSync } from "fs";
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-node';
import { Model, Tensor4D, Tensor2D, Tensor } from "@tensorflow/tfjs";
import * as readline from 'readline';

const LEARNING_RATE = 0.1;
const optimizer = tf.train.sgd(LEARNING_RATE);
const BATCH_SIZE = 64;
const TRAIN_BATCHES = 800;
const validationSplit = 0.15;
const TEST_BATCH_SIZE = 1000;
const TEST_ITERATION_FREQUENCY = 5;
const trainEpochs = 10;

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
})

async function start() {
  const model = newConvModel();

  const data = new MNISTData();
  await data.load();
  
  train(model, data);
}

function newConvModel() {
  const model = tf.sequential();

  model.add(tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 5,
    filters: 8,
    strides: 1,
    kernelInitializer: 'VarianceScaling',
    activation: 'relu'
  }));

  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }));

  model.add(tf.layers.conv2d({
    kernelSize: 5,
    filters: 16,
    strides: 1,
    activation: 'relu',
    kernelInitializer: 'VarianceScaling'
  }));
  
  model.add(tf.layers.maxPooling2d({
    poolSize: [2, 2],
    strides: [2, 2]
  }));

  model.add(tf.layers.flatten());

  model.add(tf.layers.dense({
    units: 10,
    kernelInitializer: 'VarianceScaling',
    activation: 'softmax'
  }));

  return model;
}

function newDenseModel() {
  const model = tf.sequential();
  model.add(tf.layers.flatten({inputShape: [28, 28, 1]}));
  model.add(tf.layers.dense({units: 42, activation: 'relu'}));
  model.add(tf.layers.dense({units: 10, activation: 'softmax'}));
  return model;
}

async function train(model: Model, data: MNISTData) {
  console.log("Training model...");
  let trainBatchCount = 0;

  model.compile({
    optimizer,
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy'],
  });

  let valAcc = 0;

  for (let i = 0; i < TRAIN_BATCHES; i++) {
    const batch = data.getNextBatch(BATCH_SIZE);
   
    let validationData;
   
    const history = await model.fit(
      batch.xs.reshape([BATCH_SIZE, 28, 28, 1]),
      batch.labels,
      {
        batchSize: BATCH_SIZE,
        validationData,
        epochs: 1
      }
    );
    const loss = history.history.loss[0];
    const accuracy = history.history.acc[0];
    valAcc = <number> accuracy;
  }

  const testData = data.getNextBatch(50);
  const testResult = model.evaluate(testData.xs, testData.labels);
  const testAccPercent = (<any> testResult)[1].dataSync()[0] * 100;
  const finalValAccPercent = valAcc * 100;
  console.log(
      `Final validation accuracy: ${finalValAccPercent.toFixed(1)}%; ` +
      `Final test accuracy: ${testAccPercent.toFixed(1)}%`);
  data.listen(img => {
    const classList = model.predict(img);
    const classes = (<Tensor> classList).dataSync();
    let label = 0;
    let max = 0;
    for (let i = 0; i < 10; i++) {
      if (classes[i] > max) {
        max = classes[i];
        label = i;
      }
    }
    console.log("PREDICTION: " + label);
  });
  model.save(`file://${__dirname}/model`)
}


async function listen(relearn?: boolean) {
  const model = await tf.loadModel(`file://${__dirname}/model/model.json`);
  if (relearn) {
    const optimizer = tf.train.sgd(LEARNING_RATE);
    model.compile({
      optimizer,
      loss: 'categoricalCrossentropy',
      metrics: ['accuracy'],
    });
  }
  const data = new MNISTData();
  let isWaitingForInput = false;
  data.listen(img => {
    if (!isWaitingForInput) {
      const classList = model.predict(img);
      const classes = (<Tensor> classList).dataSync();
      let label = 0;
      let max = 0;
      for (let i = 0; i < 10; i++) {
        if (classes[i] > max) {
          max = classes[i];
          label = i;
        }
      }
      console.log("PREDICTION: " + label);
      if (relearn) {
        isWaitingForInput = true;
        rl.question("Was that right? [Y/n] ", answer => {
          if (answer.toLocaleLowerCase() === 'y') {
            console.log("Great!");
            isWaitingForInput = false;
          }
          else {
            rl.question("What was the correct answer? [0, 9] ", answer => {
              const cls = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
              cls[0][parseInt(answer)] = 1;
              model.fit(img, tf.tensor2d(cls))
                .then(_ => model.save(`file://${__dirname}/model`))
                .then(_ => isWaitingForInput = false);
            });
          }
        })
      }
    }
  });
}

if (process.argv[2] === 'train') {
  start();
}
else if (process.argv[2] === 'relearn') {
  listen(true);
}
else {
  listen(false);
}
