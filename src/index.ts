import parse from 'csv-parse';
import { readFile, writeFile } from 'fs';
import path from 'path';
import { NeuralNetwork, likely } from 'brain.js';
import spinner from 'simple-spinner';

const nnConfig = {
  hiddenLayers: [100]
};

function train(num: number) {
  console.log("Loading MNIST data...");
  spinner.start();
  readFile(path.resolve(__dirname, '../mnist/mnist_train.csv'), 'utf-8', (err, csv) => {
    spinner.stop();
    console.log("Parsing MNIST data...")
    spinner.start();
    parse(csv, {auto_parse: true}, async (err, results: number[][]) => {
      spinner.stop();
      if (err) {
        console.log("There was an error parsing the MNIST csv.");
      }
      if (results) {
        console.log("Transforming input...");
        const net = new NeuralNetwork(nnConfig);
        const data = results.slice(0, num).map(image => {
          const answer = image.shift();
          const output = (new Array(10)).fill(0);
          output[answer!] = 1;
          return {input: image, output};
        });
        console.log("Training Neural Network...");
        spinner.start();
        await net.trainAsync(data);
        spinner.stop();
        console.log("Neural net trained!");
        console.log("Saving generated model...");
        const model = net.toJSON();
        writeFile(path.resolve(__dirname, '../models/numerals.json'), JSON.stringify(model), (err) => {
          if (!err) {
            console.log("Saved network model!");
          }
          else {
            console.log("There was an error saving the model: " + err);
          }
        });
      }
    });
  });
}

function test(num: number) {
  console.log("Loading MNIST tests...");
  spinner.start();
  readFile(path.resolve(__dirname, '../mnist/mnist_test.csv'), 'utf-8', (err, csv) => {
    spinner.stop();
    console.log("Parsing MNIST tests...")
    spinner.start();
    parse(csv, {auto_parse: true}, async (err, results: number[][]) => {
      spinner.stop();
      if (results) {
        console.log("Spinning up the Neural Network...");
        const net = new NeuralNetwork(nnConfig);
        readFile(path.resolve(__dirname, '../models/numerals.json'), 'utf-8', (err, json) => {
          net.fromJSON(JSON.parse(json));
          const data = results.slice(0, num).map(image => {
            const answer = image.shift();
            return {image, answer};
          });
          let correct = 0;
          for (let test of data) {
            const result = likely(test.image, net);
            console.log(`${result} should be ${test.answer}`);
            if (result === test.answer) {
              correct++;
            }
          }
          console.log("Testing done! Here are the results: ");
          console.log(`The NN got ${correct} out of ${num} correct. ${Math.floor(correct/num * 1000)/10}%`);
        });
      }
    });
  })
}

switch (process.argv[2]) {
  case 'train':
    train(parseInt(process.argv[3]));
    break;
  case 'test':
    test(parseInt(process.argv[3]));
    break;
  default:
    console.log("Please enter 'train' or 'test' subcommand.");
    process.exit(-1);
}
